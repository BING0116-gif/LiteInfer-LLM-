"""Task 12 Benchmark + Ablation 矩阵驱动器（docs/07 Task 12 + docs/05 §7/§8/§9）。

定位：measure-only。本模块只负责"跑矩阵 + 落盘 CSV/JSON"，不画图、不写
报告——图表与报告是独立的两个脚本（benchmark_charts / benchmark_report），
以 results.csv + env.json 作为它们之间的数据传输契约，避免一个脚本包办
一切、各自难以单测。

消融设计（已与用户确认，见 docs/design/benchmark.md "消融映射"）：
- 5 个真实驱动覆盖 docs/07 列出的 6 个命名消融。continuous batching（V2）
  与 paged KV（V3）在本仓库从 Task 06/08 起已合并进 EngineCore 这一个
  生产驱动，没有两套可分离的历史实现；因此 EngineCore 的双测量轴是
  "吞吐 vs 并发"（V2 效果）与 "KV utilization / 峰值块数"（V3 效果），
  在报告中分两节呈现并显式声明合并事实。
- 同模型对照（项目既定方法论）：hf 驱动独立实现，其余四个（nokv/kv/batch/
  prefix）跑在同一个 MinimalQwen 上，避免"算子实现差异"混进消融结论。

并发语义：每个 cell 提交 ``concurrency`` 个请求。顺序驱动（hf/nokv/kv）只能
逐个跑，因此并发 N 就是"拿 N 个请求排队逐个处理"；引擎驱动（batch/prefix）
同时提交 N 个，由调度器连续批处理。两种语义下吞吐都用"总产出 token / 总墙钟"
口径，cell 间才可比。

诚实性约束（docs/07 二、docs/05 §15）：
- 每个 minimal 系 cell 的生成文本必须与参照文本逐字一致（greedy 确定性），
  不一致即 FAIL；hf 文本不一致只警告（HF 与自建实现的采样边界可能不同）。
- 无 GPU 时显存指标输出 ``N/A (no GPU)``，禁止填 0。
- 脚本不做任何数值加工，全部数字来自本机真实测量。

设备/dtype：一律走 EngineConfig（补充条款 A1/A2）；本模块同样不出现任何
裸设备字面量（test_no_hardcoded_cuda 会扫描 benchmark/ 目录）。

用法（仓库根目录）：

    set "HF_HOME=D:/LiteInfer/hf_cache"
    set PYTHONPATH=d:/LiteInfer
    python benchmark/liteinfer_benchmark.py --smoke            # CPU 冒烟（验收命令）
    python benchmark/liteinfer_benchmark.py --engines hf nokv kv batch prefix ^
        --workloads decode prefill typical sp ^
        --concurrency 1 2 4 8 16 32 --outdir benchmark/results/gpu-run  # 云端全矩阵
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gc
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from liteinfer import EngineConfig
from liteinfer.config import DEFAULT_MODEL_ID, parse_dtype
from liteinfer.device import get_device, peak_memory_mb, process_rss_mb, resolve_dtype
from liteinfer.engine.core import EngineCore
from liteinfer.model.baseline import HFBaseline
from liteinfer.model.cached_generator import CachedGenerator
from liteinfer.model.eos import resolve_eos_ids
from liteinfer.model.loader import load_model_and_tokenizer
from liteinfer.model.minimal.weights import load_minimal_from_hf
from liteinfer.sampling.params import SamplingParams
from liteinfer.scheduler.config import SchedulerConfig

# --------------------------------------------------------------------------- #
# 消融驱动与 workload 定义
# --------------------------------------------------------------------------- #

#: 顺序决定的驱动顺序（同时是缺省引擎全集的展示顺序）
ENGINES: tuple[str, ...] = ("hf", "nokv", "kv", "batch", "prefix")

#: 引擎驱动（走 EngineCore 的调度器 + 块池）；其余为顺序驱动
ENGINE_DRIVERS: frozenset[str] = frozenset({"batch", "prefix"})
SEQUENTIAL_DRIVERS: frozenset[str] = frozenset(ENGINES) - ENGINE_DRIVERS

#: workload 规划文档口径（docs/05 §8）
FULL_WORKLOADS: dict[str, dict[str, int]] = {
    "decode":  {"kind": "decode-heavy",  "prompt_len": 64,   "max_tokens": 256},
    "prefill": {"kind": "prefill-heavy", "prompt_len": 2048, "max_tokens": 32},
    "typical": {"kind": "typical",       "prompt_len": 512,  "max_tokens": 128},
    "sp":      {"kind": "shared-prefix", "shared_len": 2048, "tail_len": 64,
                "max_tokens": 64},
}
#: 冒烟口径：缩放保形（形状相同、长度缩小），保证 CPU 上能在分钟级跑完
SMOKE_WORKLOADS: dict[str, dict[str, int]] = {
    "decode":  {"kind": "decode-heavy",  "prompt_len": 16,   "max_tokens": 48},
    "prefill": {"kind": "prefill-heavy", "prompt_len": 160,  "max_tokens": 16},
    "typical": {"kind": "typical",       "prompt_len": 48,   "max_tokens": 24},
    "sp":      {"kind": "shared-prefix", "shared_len": 160,  "tail_len": 16,
                "max_tokens": 16},
}

WORKLOAD_NAMES: tuple[str, ...] = tuple(FULL_WORKLOADS)


def workload_spec(name: str, smoke: bool) -> dict[str, int]:
    """取某个 workload 的口径（full 或 smoke）。未知名字 fail fast。"""
    table = SMOKE_WORKLOADS if smoke else FULL_WORKLOADS
    if name not in table:
        raise ValueError(f"未知 workload {name!r}，可选: {list(table)}")
    return dict(table[name])


def planned_prompt_len(ws: dict[str, int]) -> int:
    """workload 规划口径的总 prompt 长度（sp = shared + tail）。"""
    return ws.get("prompt_len") or (ws["shared_len"] + ws["tail_len"])


# --------------------------------------------------------------------------- #
# 提示词构造（确定性，token 数可控）
# --------------------------------------------------------------------------- #

#: 提示词宏。空间分隔的短英文词，编码稳定、跨请求可复现。
_PROMPT_MACRO = "mathematics is the study of quantity structure space and change"


def build_prompt_text(
    tokenizer: Any, n_tokens: int, macro: str = _PROMPT_MACRO
) -> str:
    """构造 token 数"恰好 n_tokens（或最接近）"的确定性提示文本。

    为什么用 encode-重复-截断-decode 而不是数空格：真实 tokenizer 的分词粒度
    不可预测，数文本长度根本控制不了 token 数。encode 一次拿到的 id 序列重复
    到目标长度再解码回文本；解码文本再编码时个别 subword 合并可能让实际长度
    差几个 token——cell 里记录的 prompt_len 是 submit 时测到的真实值，不影响
    "同 prompt 形状"的组间对比。

    ``macro`` 可注入：单测用 FakeTokenizer（只认数字字符）时传纯数字宏。
    """
    if n_tokens <= 0:
        raise ValueError(f"n_tokens 必须为正，收到 {n_tokens}")
    ids = tokenizer(macro, return_tensors="pt")["input_ids"][0].tolist()
    if not ids:
        raise ValueError(f"宏文本 {macro!r} 编码为空 token 序列")
    rep = (ids * (n_tokens // len(ids) + 1))[:n_tokens]
    return tokenizer.decode(rep, skip_special_tokens=True)


def build_sp_prompts(
    tokenizer: Any,
    shared_len: int,
    tail_len: int,
    count: int,
    tail_suffix: str = " {i} distinct query number",
    macro: str = _PROMPT_MACRO,
) -> list[str]:
    """shared-prefix 工作负载：共享前缀 + 每请求不同的尾缀。

    尾缀必须互不相同（否则 N 个请求就是同一个 prompt，测的是"重复请求"而
    不是"共享前缀"）；``tail_suffix`` 里的 ``{i}`` 会被替换成请求序号，保证
    token 一级不同。``tail_suffix``/``macro`` 可注入（单测用 FakeTokenizer
    时传纯数字）。
    """
    shared = build_prompt_text(tokenizer, shared_len, macro=macro)
    tails = [
        build_prompt_text(tokenizer, tail_len, macro=macro) + tail_suffix.format(i=i)
        for i in range(count)
    ]
    return [shared + tail for tail in tails]


# --------------------------------------------------------------------------- #
# 结果行模型 + CSV 存取
# --------------------------------------------------------------------------- #


def percentile(values: Sequence[Any], q: float) -> Optional[float]:
    """线性插值分位数；全 None/空序列返回 None（与 observability 口径一致）。"""
    vs = sorted(float(v) for v in values if v is not None)
    if not vs:
        return None
    if len(vs) == 1:
        return vs[0]
    pos = q * (len(vs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(vs) - 1)
    frac = pos - lo
    return vs[lo] * (1.0 - frac) + vs[hi] * frac


def _pct_pair(values: Sequence[Any]) -> tuple[Optional[float], Optional[float]]:
    return percentile(values, 0.5), percentile(values, 0.95)


def mean_opt(values: Sequence[Any]) -> Optional[float]:
    """None 感知均值：全 None 返回 None（"N/A 优于 0"，Task 10 口径）。"""
    vs = [float(v) for v in values if v is not None]
    return statistics.mean(vs) if vs else None


@dataclass
class CellRow:
    """一个 (engine, workload, concurrency) cell 的完整结果行。

    时序列单位统一为秒；缺失值一律 None —— CSV 里写空串，展示层渲染成
    "N/A"。这延续项目"无 GPU 显存/单 token TPOT = None 而非 0"的诚实口径。
    顺序驱动没有调度器/块池，故 max_num_seqs/max_num_batched_tokens/
    num_blocks 对它们是 None（不是 0，0 会被误读成"有配置"）。
    """

    engine: str
    workload: str
    concurrency: int
    num_requests: int
    prompt_len: int
    output_planned: int
    output_tokens_total: int
    output_tokens_mean: float
    wall_s: float
    req_per_s: float
    out_tokens_per_s: float
    # 分布指标：p50/p95 跨请求；itl 用"每请求 p50 的均值"
    ttft_p50: Optional[float]
    ttft_p95: Optional[float]
    tpot_p50: Optional[float]
    tpot_p95: Optional[float]
    itl_p50_mean: Optional[float]
    itl_p95_mean: Optional[float]
    e2e_p50: Optional[float]
    e2e_p95: Optional[float]
    # KV 观测（仅引擎驱动有值；顺序驱动为 None）
    kv_bytes_mean: Optional[float]
    kv_active_blocks_end: Optional[int]
    kv_peak_blocks: Optional[int]
    kv_blocks_used_end: Optional[int]
    kv_blocks_free_end: Optional[int]
    kv_blocks_total: Optional[int]
    kv_util_peak: Optional[float]
    # 前缀缓存观测（batch 为 None，prefix 有值）
    prefix_cached_blocks: Optional[int]
    prefix_hit_tokens_mean: Optional[float]
    # 环境/资源
    peak_gpu_mb: Optional[float]
    peak_rss_mb: Optional[float]
    # 校验 + cell 配置
    parity_ok: bool
    max_num_seqs: Optional[int]
    max_num_batched_tokens: Optional[int]
    num_blocks: Optional[int]
    block_size: int

    def to_csv(self) -> dict[str, str]:
        """序列化为 CSV 行：None -> 空串（read 侧再还原为 None）。"""
        out: dict[str, str] = {}
        for key, value in asdict(self).items():
            if value is None:
                out[key] = ""
            elif isinstance(value, bool):
                out[key] = "1" if value else "0"
            else:
                out[key] = str(value)
        return out

    @classmethod
    def from_csv(cls, values: dict[str, str]) -> "CellRow":
        """从 CSV 行的原始字符串还原（int 列严格、Optional 列容错）。"""

        def as_int(key: str) -> int:
            raw = values[key]
            if raw == "":
                raise ValueError(f"CSV 列 {key} 不能为空（int 列）")
            return int(raw)

        def as_opt_int(key: str) -> Optional[int]:
            raw = values[key]
            return int(raw) if raw not in ("", "None") else None

        def as_opt_float(key: str) -> Optional[float]:
            raw = values[key]
            return float(raw) if raw not in ("", "None") else None

        return cls(
            engine=values["engine"],
            workload=values["workload"],
            concurrency=as_int("concurrency"),
            num_requests=as_int("num_requests"),
            prompt_len=as_int("prompt_len"),
            output_planned=as_int("output_planned"),
            output_tokens_total=as_int("output_tokens_total"),
            output_tokens_mean=float(values["output_tokens_mean"]),
            wall_s=float(values["wall_s"]),
            req_per_s=float(values["req_per_s"]),
            out_tokens_per_s=float(values["out_tokens_per_s"]),
            ttft_p50=as_opt_float("ttft_p50"),
            ttft_p95=as_opt_float("ttft_p95"),
            tpot_p50=as_opt_float("tpot_p50"),
            tpot_p95=as_opt_float("tpot_p95"),
            itl_p50_mean=as_opt_float("itl_p50_mean"),
            itl_p95_mean=as_opt_float("itl_p95_mean"),
            e2e_p50=as_opt_float("e2e_p50"),
            e2e_p95=as_opt_float("e2e_p95"),
            kv_bytes_mean=as_opt_float("kv_bytes_mean"),
            kv_active_blocks_end=as_opt_int("kv_active_blocks_end"),
            kv_peak_blocks=as_opt_int("kv_peak_blocks"),
            kv_blocks_used_end=as_opt_int("kv_blocks_used_end"),
            kv_blocks_free_end=as_opt_int("kv_blocks_free_end"),
            kv_blocks_total=as_opt_int("kv_blocks_total"),
            kv_util_peak=as_opt_float("kv_util_peak"),
            prefix_cached_blocks=as_opt_int("prefix_cached_blocks"),
            prefix_hit_tokens_mean=as_opt_float("prefix_hit_tokens_mean"),
            peak_gpu_mb=as_opt_float("peak_gpu_mb"),
            peak_rss_mb=as_opt_float("peak_rss_mb"),
            parity_ok=values["parity_ok"] == "1",
            max_num_seqs=as_opt_int("max_num_seqs"),
            max_num_batched_tokens=as_opt_int("max_num_batched_tokens"),
            num_blocks=as_opt_int("num_blocks"),
            block_size=as_int("block_size"),
        )

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


def write_rows_csv(path: Path, rows: Sequence[CellRow]) -> None:
    """把 cell 行写进 CSV。列顺序 = CellRow 字段顺序（稳定契约）。"""
    fieldnames = [f.name for f in fields(CellRow)]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv())


def read_rows_csv(path: Path) -> list[CellRow]:
    """读回 CSV 为 CellRow 列表（round-trip 契约的另一半）。"""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [CellRow.from_csv(row) for row in reader]


# --------------------------------------------------------------------------- #
# 实验记录（docs/05 §14 模板）
# --------------------------------------------------------------------------- #


def env_record(
    *,
    model_id: str,
    device: Any,
    dtype: Any,
    smoke: bool,
    workload_names: Sequence[str],
    engine_names: Sequence[str],
    concurrency: Sequence[int],
    block_size: int,
    num_requests_note: str,
) -> dict[str, Any]:
    """组装 JSON 安全的实验记录，供环境复现与图表脚注使用。

    不做任何性能数字——它只回答"在什么环境、用什么配置跑的"。
    """
    dev = torch.device(device)
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "model_id": model_id,
        "device": str(dev),
        "device_type": dev.type,  # 图表脚注据此显示 "GPU" / "CPU"
        "dtype": str(dtype),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": _import_version("transformers"),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "smoke": smoke,
        "workloads": {n: workload_spec(n, smoke) for n in workload_names},
        "engines": list(engine_names),
        "concurrency": [int(c) for c in concurrency],
        "block_size": block_size,
        "num_requests_note": num_requests_note,
        "note": "CPU 上的数字仅验证流水线正确性，不进简历/汇报（docs/07 二、docs/05 §15）",
    }


def _import_version(pkg: str) -> str:
    try:
        import importlib

        return importlib.import_module(pkg).__version__
    except Exception:
        return "n/a"


# --------------------------------------------------------------------------- #
# 单 cell 执行
# --------------------------------------------------------------------------- #


def _run_sequential(
    *,
    engine: str,
    cached_gen: Optional[CachedGenerator],
    hf_baseline: Optional[HFBaseline],
    prompts: Sequence[str],
    params: SamplingParams,
) -> tuple[float, list[dict[str, Any]]]:
    """顺序驱动的 cell：N 个请求逐个 generate，返回 (wall, 每请求指标)。

    顺序驱动没有调度器与逐 token 打点：TTFT≈prefill 耗时、TPOT=decode 段
    平均每 token（HF 无拆段计时，相关列为 None，报告里标注"HF 未拆段"）。
    """
    reqs: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for prompt in prompts:
        if engine == "hf":
            if hf_baseline is None:
                raise RuntimeError("hf 驱动需要 HFBaseline")
            out = hf_baseline.generate(
                prompt, max_new_tokens=params.max_tokens, greedy=True
            )
            reqs.append(
                {
                    "text": out.text,
                    "prompt_tokens": out.prompt_tokens,
                    "output_tokens": out.output_tokens,
                    "finish_reason": "generate",
                    "ttft_s": None,  # HF 无拆段计时 -> N/A
                    "tpot_s": None,
                    "itl_p50": None,
                    "itl_p95": None,
                    "e2e_s": out.latency_s,
                    "cache_bytes": None,
                    "prefix_hit_tokens": 0,
                }
            )
        else:
            if cached_gen is None:
                raise RuntimeError(f"{engine} 驱动需要 MinimalQwen")
            out = cached_gen.generate(
                prompt, params, use_cache=(engine == "kv")
            )
            out_tok = out.output_tokens
            reqs.append(
                {
                    "text": out.text,
                    "prompt_tokens": out.prompt_tokens,
                    "output_tokens": out_tok,
                    "finish_reason": out.finish_reason,
                    # 顺序驱动 TTFT≈prefill 耗时（无排队，准入口即 prefill）
                    "ttft_s": out.prefill_latency_s if out_tok > 0 else None,
                    "tpot_s": (
                        out.decode_latency_s / (out_tok - 1) if out_tok > 1 else None
                    ),
                    "itl_p50": None,  # 顺序驱动未记录逐 token 时刻
                    "itl_p95": None,
                    "e2e_s": out.latency_s,
                    "cache_bytes": out.cache_bytes,
                    "prefix_hit_tokens": 0,
                }
            )
    wall = time.perf_counter() - t0
    return wall, reqs


def _run_engine_cell(
    core: EngineCore,
    prompts: Sequence[str],
    params: SamplingParams,
) -> tuple[float, list[dict[str, Any]], int, int, int, int]:
    """引擎驱动 cell：同时提交 N 个请求并推进到全部终态。

    返回 (wall, 每请求指标, 峰值在飞块数, 结束在飞块数, 结束时空闲块数, 总块数)。
    结束块数的用法：batch（prefix off）结束时必须 used==0（docs/02 §2 契约）；
    prefix on 时缓存块会占住池，这时验 free+used == total（块没被缓存结构吞掉）。
    """
    for prompt in prompts:
        core.submit(prompt, params)
    n = len(prompts)
    max_tokens = params.max_tokens

    # 块池观测在一个 step 内会突变（多个 prefill/decode 顺序推进），沿循环
    # 采样峰值最贴近"运行期最高占用"，比终态读数有意义
    used_peak = 0
    t0 = time.perf_counter()
    max_steps = n * (max_tokens + 2) + 4  # 每请求至多 1 次 prefill + max_tokens 次 decode
    steps = 0
    while core.registry.active():
        if steps >= max_steps:
            raise RuntimeError(
                f"cell 在 {max_steps} 步内未收敛（预算配置错误或引擎 bug），"
                f"active={len(core.registry.active())}"
            )
        core.step()
        steps += 1
        used_peak = max(used_peak, core.runner.paged.num_blocks_used)
    t1 = time.perf_counter()

    reqs: list[dict[str, Any]] = []
    for req in core.registry.all():
        m = core.metrics.get(req.request_id)  # RequestMetrics（终态已 record）
        src = core.get_request(req.request_id)
        reqs.append(
            {
                "text": src.output_text,
                "prompt_tokens": src.prompt_tokens,
                "output_tokens": src.output_tokens,
                "finish_reason": src.finish_reason or "unknown",
                "ttft_s": m.ttft_s if m else None,
                "tpot_s": m.tpot_s if m else None,
                "itl_p50": m.itl_p50_s if m else None,
                "itl_p95": m.itl_p95_s if m else None,
                "e2e_s": m.e2e_s if m else None,
                "cache_bytes": m.cache_bytes if m else None,
                "prefix_hit_tokens": m.prefix_hit_tokens if m else 0,
                "status": m.status if m else "n/a",
            }
        )
    if any(r.get("status") == "cancelled" for r in reqs):
        raise RuntimeError("cell 出现 cancelled 请求，测量无效（本流程不应有取消）")
    return (
        (t1 - t0),
        reqs,
        used_peak,
        core.runner.paged.num_blocks_used,
        core.runner.paged.num_blocks_free,
        core.runner.paged.num_blocks_total,
    )


def aggregate_cell(
    *,
    wall_s: float,
    reqs: Sequence[dict[str, Any]],
    engine: str,
    workload: str,
    concurrency: int,
    output_planned: int,
    kv_peak_blocks: Optional[int],
    kv_used_end: Optional[int],
    kv_free_end: Optional[int],
    kv_total_blocks: Optional[int],
    prefix_cached_blocks: Optional[int],
    max_num_seqs: Optional[int],
    max_num_batched_tokens: Optional[int],
    num_blocks: Optional[int],
    block_size: int,
    parity_ok: bool,
) -> CellRow:
    """把 cell 的每请求测量汇总成一行（纯函数，可直接单测）。"""
    out_tokens = [r["output_tokens"] for r in reqs]
    out_total = sum(out_tokens)
    ttft_p50, ttft_p95 = _pct_pair([r["ttft_s"] for r in reqs])
    tpot_p50, tpot_p95 = _pct_pair([r["tpot_s"] for r in reqs])
    e2e_p50, e2e_p95 = _pct_pair([r["e2e_s"] for r in reqs])
    kv_bytes = [r["cache_bytes"] for r in reqs]
    util_peak = (
        (kv_peak_blocks / kv_total_blocks)
        if kv_peak_blocks is not None and kv_total_blocks not in (None, 0)
        else None
    )
    return CellRow(
        engine=engine,
        workload=workload,
        concurrency=int(concurrency),
        num_requests=len(reqs),
        prompt_len=int(reqs[0]["prompt_tokens"]),
        output_planned=output_planned,
        output_tokens_total=out_total,
        output_tokens_mean=out_total / len(reqs) if reqs else 0.0,
        wall_s=wall_s,
        req_per_s=(len(reqs) / wall_s) if wall_s > 0 else 0.0,
        out_tokens_per_s=(out_total / wall_s) if wall_s > 0 else 0.0,
        ttft_p50=ttft_p50,
        ttft_p95=ttft_p95,
        tpot_p50=tpot_p50,
        tpot_p95=tpot_p95,
        itl_p50_mean=mean_opt([r["itl_p50"] for r in reqs]),
        itl_p95_mean=mean_opt([r["itl_p95"] for r in reqs]),
        e2e_p50=e2e_p50,
        e2e_p95=e2e_p95,
        kv_bytes_mean=mean_opt(kv_bytes),
        kv_active_blocks_end=kv_used_end,
        kv_peak_blocks=kv_peak_blocks,
        kv_blocks_used_end=kv_used_end,
        kv_blocks_free_end=kv_free_end,
        kv_blocks_total=kv_total_blocks,
        kv_util_peak=util_peak,
        prefix_cached_blocks=prefix_cached_blocks,
        prefix_hit_tokens_mean=mean_opt([r["prefix_hit_tokens"] for r in reqs]),
        peak_gpu_mb=peak_memory_mb(),
        peak_rss_mb=process_rss_mb(),
        parity_ok=parity_ok,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        num_blocks=num_blocks,
        block_size=block_size,
    )


# --------------------------------------------------------------------------- #
# 矩阵驱动器
# --------------------------------------------------------------------------- #


def _engine_cfg(
    base: EngineConfig,
    *,
    concurrency: int,
    prompt_len: int,
    max_tokens: int,
    enable_prefix_cache: bool,
) -> EngineConfig:
    """构造引擎驱动专用的 EngineConfig（每 cell 新建，块池账目干净）。

    为什么必须显式给 num_blocks：默认推导按 max_new_tokens+128 token/序列，
    长 prompt（2048）会直接撞池，alloc 抛 ValueError。这里按
    ceil((prompt+max_tokens)/block_size) 每序列块数 × 并发 + 每序列 4 块余量。
    token budget 同理：prompt ≥ max_num_batched_tokens 时 submit 直接抛
    ValueError，这里按 prompt_len 上取对齐加到预算。
    """
    bs = base.block_size
    per_seq_blocks = -(-(prompt_len + max_tokens) // bs)  # 上取整
    num_blocks = (per_seq_blocks + 4) * concurrency
    return EngineConfig(
        model_id=base.model_id,
        device=base.device,
        dtype=base.dtype,
        hf_cache_dir=base.hf_cache_dir,
        max_new_tokens=max_tokens,
        seed=base.seed,
        trust_remote_code=base.trust_remote_code,
        local_files_only=base.local_files_only,
        scheduler=SchedulerConfig(
            max_num_seqs=max(concurrency, 1),
            # prefill 一次性吃整条 prompt，budget 必须放得下最长 prompt
            max_num_batched_tokens=max(2048, prompt_len + 8),
        ),
        block_size=bs,
        num_blocks=num_blocks,
        enable_prefix_cache=enable_prefix_cache,
    )


def _cell_prompts(tokenizer: Any, ws: dict[str, int], concurrency: int) -> list[str]:
    """按 workload 类型产出本 cell 的 N 个 prompt。"""
    if ws.get("kind") == "shared-prefix":
        return build_sp_prompts(
            tokenizer, ws["shared_len"], ws["tail_len"], concurrency
        )
    return [build_prompt_text(tokenizer, ws["prompt_len"]) for _ in range(concurrency)]


def run_cell(
    *,
    engine: str,
    wname: str,
    conc: int,
    prompts: Sequence[str],
    output_planned: int,
    planned_prompt_len: int,
    max_tokens: int,
    tag: str,
    base: EngineConfig,
    minimal: Optional[Any],
    tokenizer: Any,
    eos_ids: Any,
    cached_gen: Optional[CachedGenerator],
    hf_baseline: Optional[HFBaseline],
    ref_text: dict[tuple[str, int], str],
    block_size: int,
    parity: bool,
) -> CellRow:
    """执行单个 cell（驱动选择 + 统一校验 + 汇总行）。"""
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    if engine in SEQUENTIAL_DRIVERS:
        wall, reqs = _run_sequential(
            engine=engine,
            cached_gen=cached_gen,
            hf_baseline=hf_baseline,
            prompts=prompts,
            params=params,
        )
        kv_peak = kv_used = kv_free = kv_total = None
        prefix_cached = None
        cell_cfg = (None, None, None)
    else:
        if minimal is None:
            raise RuntimeError(f"{engine} 驱动需要 MinimalQwen")
        ecfg = _engine_cfg(
            base,
            concurrency=conc,
            prompt_len=planned_prompt_len,
            max_tokens=max_tokens,
            enable_prefix_cache=(engine == "prefix"),
        )
        core = EngineCore(minimal, tokenizer, ecfg, eos_ids)
        try:
            wall, reqs, used_peak, used_end, free_end, total = _run_engine_cell(
                core, prompts, params
            )
            # prefix 结束后的可驱逐缓存块（ref==0、等在 LRU 里白捡的）。
            # 必须在 del core 之前读：PrefixCache 挂在 runner 上，删了就读不到了
            prefix_cached = (
                core.runner.prefix.num_evictable_blocks
                if engine == "prefix" and core.runner.prefix is not None
                else None
            )
            # 防泄漏契约：batch（prefix off）终态块必须全回收；prefix on 验
            # free+used==total（缓存块刻意占住池，见 docs/02 §11）
            if engine == "batch" and used_end != 0:
                raise RuntimeError(
                    f"{tag} 结束时仍有 {used_end} 个块未回收（pool 泄漏）"
                )
            if free_end + used_end != total:
                raise RuntimeError(
                    f"{tag} 块账目不平衡 free({free_end})+used({used_end}) != total({total})"
                )
        finally:
            del core
            gc.collect()
        kv_peak, kv_used, kv_free, kv_total = used_peak, used_end, free_end, total
        cell_cfg = (conc, ecfg.scheduler.max_num_batched_tokens, ecfg.num_blocks)

    # 逐字一致性校验：minimal 系必须与参照逐字节一致；hf 不一致仅警告
    # （HF 与自建实现的浮点采样边界可能不同，硬失败会冤枉一个无害差异）
    parity_ok = True
    for i, r in enumerate(reqs):
        key = (wname, i)
        if key not in ref_text:
            if engine != "hf":
                ref_text[key] = r["text"]
            continue
        if r["text"] == ref_text[key]:
            continue
        parity_ok = False
        if engine == "hf":
            print(
                f"{tag} [warn] HF 文本与参照不一致（不同实现，仅警告）："
                f"{r['text']!r} vs {ref_text[key]!r}"
            )
        elif parity:
            raise AssertionError(
                f"{tag} 输出不一致（{engine} vs 参照）：\n"
                f"  cur: {r['text']!r}\n  ref: {ref_text[key]!r}"
            )
        else:
            print(f"{tag} [warn] 输出不一致（--no-parity 跳过硬失败）")
    if not parity_ok and engine != "hf" and not parity:
        parity_ok = True  # --no-parity 时不把不一致记成 FAIL

    return aggregate_cell(
        wall_s=wall,
        reqs=reqs,
        engine=engine,
        workload=wname,
        concurrency=conc,
        output_planned=output_planned,
        kv_peak_blocks=kv_peak,
        kv_used_end=kv_used,
        kv_free_end=kv_free,
        kv_total_blocks=kv_total,
        prefix_cached_blocks=prefix_cached,
        max_num_seqs=cell_cfg[0],
        max_num_batched_tokens=cell_cfg[1],
        num_blocks=cell_cfg[2],
        block_size=block_size,
        parity_ok=parity_ok,
    )


def _print_cell(row: CellRow) -> None:
    ttft = row.ttft_p50
    util = row.kv_util_peak
    ttft_s = f"{ttft * 1000:8.1f}ms" if ttft is not None else "      N/A"
    util_s = f"{util:7.1%}" if util is not None else "     N/A"
    print(
        f"[{row.engine:>6} {row.workload:<8} conc={row.concurrency:>2}] "
        f"wall={row.wall_s:8.3f}s  out_tok/s={row.out_tokens_per_s:8.3f}  "
        f"TTFT p50={ttft_s}  kv_util_peak={util_s}"
    )


# --------------------------------------------------------------------------- #
# 矩阵编排
# --------------------------------------------------------------------------- #


def run_matrix(
    args: argparse.Namespace,
) -> tuple[list[CellRow], dict[str, Any]]:
    """跑完整矩阵，返回 (行列表, 实验记录)。

    加载策略：minimal（MinimalQwen + tokenizer + EOS）只加载一次，四个
    minimal 系驱动共享；HF 模型仅当引擎列表含 hf 时才加载。两者可同时驻留
    （0.5B 约 2+1GB，本机 16.9GB；Task 09 的教训是避免叠 8 个模型副本，
    这里只有 2 个，无此风险）。
    """
    base = EngineConfig(
        model_id=args.model_id,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        hf_cache_dir=args.hf_cache,
    )
    dev = get_device(base.device)
    dtype = resolve_dtype(base.dtype, dev)
    print(
        f"model={base.model_id} device={dev} dtype={dtype} "
        f"smoke={args.smoke} block_size={args.block_size}\n"
    )

    workloads = list(args.workloads)
    engines = list(args.engines)
    concurrency = list(args.concurrency)

    need_minimal = any(e != "hf" for e in engines)
    need_hf = "hf" in engines

    minimal: Any = None
    tokenizer: Any = None
    eos_ids: Any = None
    if need_minimal:
        loaded = load_minimal_from_hf(base)
        minimal = loaded.minimal
        tokenizer = loaded.tokenizer
        # EOS 从 HF 参照模型解析（generation_config 口径，Task 04 踩坑）
        eos_ids = resolve_eos_ids(loaded.hf_model, loaded.tokenizer)
    hf_model: Any = None
    hf_tokenizer: Any = None
    if need_hf:
        hf_loaded = load_model_and_tokenizer(base)
        hf_model, hf_tokenizer = hf_loaded.model, hf_loaded.tokenizer

    # 预热：每个底层模型的首次 forward 含线程池/算子选择冷启动（Task 04 踩坑），
    # 预热一次让矩阵第一个 cell 不用背这个锅
    if not args.no_prime:
        prime_cfg = EngineConfig(
            model_id=base.model_id,
            device=base.device,
            dtype=base.dtype,
            hf_cache_dir=base.hf_cache_dir,
            max_new_tokens=4,
        )
        if need_minimal:
            g0 = CachedGenerator(minimal, tokenizer, prime_cfg, eos_ids)
            g0.generate("The capital of France is", SamplingParams(4, temperature=0.0))
        if need_hf:
            b0 = HFBaseline(hf_model, hf_tokenizer, prime_cfg)
            b0.generate("The capital of France is", max_new_tokens=4)
        print("warmup done\n")

    # 顺序驱动共享同一个 CachedGenerator（同一 minimal 实例，零引擎态）
    cached_gen = CachedGenerator(minimal, tokenizer, base, eos_ids) if need_minimal else None
    hf_baseline = HFBaseline(hf_model, hf_tokenizer, base) if need_hf else None

    rows: list[CellRow] = []
    ref_text: dict[tuple[str, int], str] = {}  # (workload, idx) -> 首个 minimal 驱动文本
    for engine in engines:
        print(f"===== engine={engine} =====")
        for wname in workloads:
            ws = workload_spec(wname, args.smoke)
            max_tokens = ws["max_tokens"]
            output_planned = max_tokens
            for conc in concurrency:
                prompts = _cell_prompts(tokenizer, ws, conc)
                tag = f"[{engine:>6} {wname:<8} conc={conc:>2}]"
                row = run_cell(
                    engine=engine,
                    wname=wname,
                    conc=conc,
                    prompts=prompts,
                    output_planned=output_planned,
                    planned_prompt_len=planned_prompt_len(ws),
                    max_tokens=max_tokens,
                    tag=tag,
                    base=base,
                    minimal=minimal,
                    tokenizer=tokenizer,
                    eos_ids=eos_ids,
                    cached_gen=cached_gen,
                    hf_baseline=hf_baseline,
                    ref_text=ref_text,
                    block_size=args.block_size,
                    parity=not args.no_parity,
                )
                rows.append(row)
                _print_cell(row)

    # 输出行序稳定：engine → workload → concurrency，便于 diff
    rows.sort(key=lambda r: (r.engine, r.workload, r.concurrency))
    env = env_record(
        model_id=base.model_id,
        device=dev,
        dtype=dtype,
        smoke=args.smoke,
        workload_names=workloads,
        engine_names=engines,
        concurrency=concurrency,
        block_size=args.block_size,
        num_requests_note="num_requests=concurrency：顺序驱动逐个跑，引擎驱动同时跑",
    )
    return rows, env


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LiteInfer Benchmark + Ablation 矩阵驱动器")
    p.add_argument("--device", default="cpu", help="设备标识（默认 cpu；云端传 GPU 设备名）")
    p.add_argument("--dtype", default="float32", help="dtype 别名（默认 float32；CPU 强制回退）")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument(
        "--workloads", nargs="+", default=None,
        help=f"workload 子集，可选 {list(WORKLOAD_NAMES)}（缺省=全部）",
    )
    p.add_argument(
        "--concurrency", nargs="+", type=int, default=None,
        help="并发数列表（缺省全矩阵 1/2/4/8/16/32；smoke 缺省 1/2/4）",
    )
    p.add_argument(
        "--engines", nargs="+", default=None,
        help=f"驱动子集，可选 {list(ENGINES)}（缺省全 5 个；smoke 缺省 kv/batch/prefix）",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="冒烟模式：缩放保形的 workload 长度 + 缩减缺省并发/引擎",
    )
    p.add_argument("--block-size", type=int, default=16, help="分页块大小（默认 16）")
    p.add_argument(
        "--repeat", type=int, default=1,
        help="每个 cell 重复次数（取中位数；默认 1。GPU 全矩阵一轮跑完可不设）",
    )
    p.add_argument(
        "--hf-cache", default=None,
        help="HF 缓存目录；缺省走 default_hf_cache_dir 兜底链（HF_HOME 优先）",
    )
    p.add_argument(
        "--outdir", default=None, help="输出目录；缺省 benchmark/results/run-<时间戳>",
    )
    p.add_argument("--tag", default=None, help="输出目录后缀标签（便于区分跑次）")
    p.add_argument("--no-prime", action="store_true", help="关闭模型预热（调试用）")
    p.add_argument("--no-parity", action="store_true", help="关闭 minimal 系逐字校验（不推荐）")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # 缺省值补全（smoke 与 full 不同）
    if args.smoke:
        args.workloads = args.workloads or list(SMOKE_WORKLOADS)
        args.concurrency = args.concurrency or [1, 2, 4]
        args.engines = args.engines or list(("kv", "batch", "prefix"))
    else:
        args.workloads = args.workloads or list(WORKLOAD_NAMES)
        args.concurrency = args.concurrency or [1, 2, 4, 8, 16, 32]
        args.engines = args.engines or list(ENGINES)

    for name in args.workloads:
        workload_spec(name, args.smoke)  # 校验名字，fail fast
    for e in args.engines:
        if e not in ENGINES:
            raise ValueError(f"未知引擎 {e!r}，可选 {list(ENGINES)}")
    if args.repeat < 1:
        raise ValueError("--repeat 必须 >= 1")

    rows, env = run_matrix(args)
    env["repeat_cells"] = args.repeat
    if args.repeat > 1:
        rows = _median_repeat_rows(rows)

    outdir = Path(args.outdir) if args.outdir else _default_outdir(args.tag)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "results.csv"
    env_path = outdir / "env.json"
    write_rows_csv(csv_path, rows)
    with open(env_path, "w", encoding="utf-8") as fh:
        json.dump(env, fh, ensure_ascii=False, indent=2)
    print(f"\n结果: {csv_path}")
    print(f"实验记录: {env_path}")
    mem = peak_memory_mb()
    print(f"peak memory: {mem if mem is not None else 'N/A (no GPU)'}")
    return 0


def _median_repeat_rows(rows: Sequence[CellRow]) -> list[CellRow]:
    """把 repeat 次重复的 cell 行按 (engine, workload, concurrency) 合并。

    数值列取中位数；保持一致性的列（engine/文本相关）取首个。中位数比均值
    抗单次抖动（沿用 kv_cache_benchmark 的中位数方法论）。
    """
    grouped: dict[tuple[str, str, int], list[CellRow]] = {}
    for r in rows:
        grouped.setdefault((r.engine, r.workload, r.concurrency), []).append(r)
    out: list[CellRow] = []
    for group in grouped.values():
        first = group[0]
        merged: dict[str, Any] = {}
        for f in fields(first):
            if f.name == "parity_ok":
                merged[f.name] = all(getattr(r, f.name) for r in group)
                continue
            vals = [getattr(r, f.name) for r in group]
            if all(v is not None for v in vals) and all(
                isinstance(v, (int, float)) for v in vals
            ):
                med = statistics.median(float(v) for v in vals)
                merged[f.name] = int(med) if all(isinstance(v, int) for v in vals) else med
            else:
                merged[f.name] = getattr(first, f.name)
        out.append(CellRow(**merged))
    return out


def _default_outdir(tag: Optional[str]) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"run-{stamp}" + (f"-{tag}" if tag else "")
    return Path(__file__).resolve().parent / "results" / name


if __name__ == "__main__":
    sys.exit(main())