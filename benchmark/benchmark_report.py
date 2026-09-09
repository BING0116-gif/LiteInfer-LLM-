"""Task 12 报告组装：results.csv + env.json + charts/*.png -> report.md。

    python benchmark/benchmark_report.py --csv ... --env ... --charts-dir ... --out report.md

诚实性规则（docs/05 §15 与 docs/07 二）：
- 报告的"观察与结论"一节只允许输出由 CSV 真实数字直接算出的对比
  （比如 prefix vs batch 的 TTFT 差距），算不出来/缺列的项一律跳过，
  绝不写"看起来合理"的话术；
- 无 GPU 时显存列输出 N/A (no GPU)，禁止填 0；
- 报告开头固定声明：CPU 数字仅验证流水线正确性，不进简历/汇报。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from benchmark.liteinfer_benchmark import CellRow, read_rows_csv


def _ms(value: Optional[float]) -> str:
    """秒 -> 毫秒展示；None -> N/A（诚实性优先于 0 占位）。"""
    if value is None:
        return "N/A"
    return f"{value * 1000:.1f}"


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1%}"


def _num(value: Optional[float], nd: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{nd}f}"


def _env_table(env: dict[str, Any]) -> str:
    """实验记录表（docs/05 §14），渲染为 markdown。"""
    def row(key: str, label: str) -> str:
        return f"| {label} | {env.get(key, 'n/a')} |\n"

    lines = ["| 字段 | 值 |\n", "|---|---|\n"]
    lines.append(row("model_id", "模型"))
    lines.append(row("device", "设备"))
    lines.append(row("dtype", "dtype"))
    lines.append(row("torch_version", "PyTorch"))
    lines.append(row("transformers_version", "Transformers"))
    lines.append(row("python_version", "Python"))
    lines.append(row("platform", "平台"))
    lines.append(row("cpu_count", "CPU 核数"))
    lines.append(row("torch_threads", "Torch 线程"))
    lines.append(row("block_size", "block_size"))
    lines.append(f"| 引擎顺序 | {', '.join(env.get('engines', []))} |\n")
    lines.append(f"| 并发数 | {', '.join(str(c) for c in env.get('concurrency', []))} |\n")
    workloads = env.get("workloads", {})
    lines.append(
        "| workload 口径 | "
        + "; ".join(f"{k}(p={v.get('prompt_len', v.get('shared_len'))},"
                    f"out={v['max_tokens']})" for k, v in workloads.items())
        + " |\n"
    )
    lines.append(row("smoke", "smoke 模式"))
    lines.append(row("timestamp_utc", "时间戳(UTC)"))
    lines.append(row("num_requests_note", "并发语义"))
    return "".join(lines)


def _summary_table(rows: Sequence[CellRow]) -> str:
    """全量结果表（节选列；完整列在 results.csv）。"""
    header = (
        "| engine | workload | conc | req/s | out tok/s | TTFT p50 | TPOT p50 | "
        "ITL p50 | E2E p50 | KV util | prefix hit | parity |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    body = []
    for r in rows:
        body.append(
            f"| {r.engine} | {r.workload} | {r.concurrency} | "
            f"{_num(r.req_per_s)} | {_num(r.out_tokens_per_s)} | "
            f"{_ms(r.ttft_p50)} | {_ms(r.tpot_p50)} | {_ms(r.itl_p50_mean)} | "
            f"{_ms(r.e2e_p50)} | {_pct(r.kv_util_peak)} | "
            f"{r.prefix_hit_tokens_mean if r.prefix_hit_tokens_mean is not None else 'N/A'} | "
            f"{'OK' if r.parity_ok else 'FAIL'} |\n"
        )
    return header + "".join(body)


def _insights(rows: Sequence[CellRow]) -> str:
    """由 CSV 真实数字生成的对比要点；算不出的项不写。"""
    lines: list[str] = []

    def best(row_pred, key):  # 取满足条件的第一个值
        for r in rows:
            if row_pred(r):
                return getattr(r, key)
        return None

    # 1) prefix 效果（sp workload，最高并发处 batch vs prefix 的 TTFT p50）
    sp = [r for r in rows if r.workload == "sp" and r.ttft_p50 is not None]
    batch_eng = {"batch"}
    prefix_eng = {"prefix"}
    if sp:
        maxc = max(r.concurrency for r in sp)
        b = next(
            (r.ttft_p50 for r in sp if r.engine in batch_eng and r.concurrency == maxc), None
        )
        p = next(
            (r.ttft_p50 for r in sp if r.engine in prefix_eng and r.concurrency == maxc), None
        )
        if b is not None and p is not None:
            rel = (b - p) / b if b > 0 else None
            lines.append(
                f"- sp workload @concurrency={maxc}: prefix 相对 batch 的 TTFT p50 "
                f"下降 {_pct(rel)}（{_ms(b)} ms -> {_ms(p)} ms，hit tokens 见结果表）"
                if rel is not None and rel > 0
                else "- sp workload：prefix 与 batch 的 TTFT 对比数据不足或未显示下降，依据不足，不做结论"
            )
        else:
            lines.append("- sp workload：缺少 batch 或 prefix 的 TTFT 数据，不做 prefix 结论")
    else:
        lines.append("- sp workload 无 TTFT 数据，不做 prefix 结论")

    # 2) 批处理收益：batch vs kv（最高并发的 out tok/s 比率）
    def tok_rate(engine: str, wl: str) -> Optional[float]:
        vals = [
            r.out_tokens_per_s
            for r in rows
            if r.engine == engine and r.workload == wl and r.out_tokens_per_s is not None
        ]
        return max(vals) if vals else None

    batch_rate = tok_rate("batch", "decode")
    kv_rate = tok_rate("kv", "decode")
    if batch_rate is not None and kv_rate is not None and kv_rate > 0:
        lines.append(
            f"- decode workload：batch 峰值吞吐 {batch_rate:.2f} tok/s vs kv 顺序 "
            f"{kv_rate:.2f} tok/s（比率 {batch_rate / kv_rate:.2f}x，CPU 上仅供管线验证）"
        )
    else:
        lines.append("- 缺少 batch/kv 的 decode 吞吐数据，不做批处理收益结论")

    return "## 7. 观察与结论（全部来自本报告 CSV 真实测量；不足之处明说）\n\n" + "\n".join(lines) + "\n"


def build_report(
    csv_path: Path,
    env_path: Path,
    charts_dir: Optional[Path],
    out_path: Path,
) -> str:
    rows = read_rows_csv(csv_path)
    with open(env_path, encoding="utf-8") as fh:
        env = json.load(fh)

    # 显卡显存：无 GPU 时展示层统一 "N/A (no GPU)"（补充条款 A3）。
    # 优先取 CSV 行里的实测值，没有则退化为 N/A（Env 不自己猜）
    gpu_mb = next((r.peak_gpu_mb for r in rows if r.peak_gpu_mb is not None), None)
    gpu_display = "N/A (no GPU)" if gpu_mb is None else f"{gpu_mb:.1f} MB"

    charts_lines = []
    if charts_dir is not None and charts_dir.is_dir():
        for png in sorted(charts_dir.glob("*.png")):
            charts_lines.append(f"- `{png.name}`（脚注含实验记录）")
        charts_md = "## 6. 图表索引\n\n" + ("\n".join(charts_lines) or "（本次无图表产出）") + "\n\n"
    else:
        charts_md = "## 6. 图表索引\n\n本报告未提供 charts 目录。\n\n"

    text = f"""# LiteInfer Benchmark 报告

> 诚实性声明：本报告全部数字来自 {csv_path.name} 的真实测量。
> CPU 上的数字仅验证流水线正确性，**不得**用于简历/汇报（docs/07 二、docs/05 §15）。
> 当前环境显存指标：{gpu_display}

## 1. 实验记录（docs/05 §14）

{_env_table(env)}

## 2. 方法摘要

**驱动映射**（docs/07 Task 12 的 6 个命名消融 -> 5 个真实驱动，
continuous batching 与 paged KV 在本仓库合并进 EngineCore —— 见 docs/design/benchmark.md）：

| 命名消融 | 实际驱动 | 测量轴 |
|---|---|---|
| HF baseline | `hf`（顺序 HF generate） | E2E / 吞吐 |
| no-cache | `nokv`（顺序无缓存） | E2E / 吞吐 |
| KV (contiguous) | `kv`（顺序连续 KV） | E2E / 吞吐 |
| continuous batching | `batch`（EngineCore） | 吞吐 vs 并发 |
| paged KV | `batch`（EngineCore 块池） | KV utilization / 峰值块 |
| prefix cache | `prefix`（EngineCore + 前缀缓存） | 命中 token / TTFT 下降 |

- 并发定义：`num_requests = concurrency`。顺序驱动逐个排队跑，引擎驱动同时提交。
- 指标口径：TTFT=首 token - enqueue；TPOT=相邻 token 均值；ITL p50 为"每请求
  p50 的均值"；HF 未拆段计时，其 TTFT/TPOT/ITL 为 N/A。
- 所有 minimal 系 cell 已做输出逐字一致性校验（`parity` 列）；失败即整轮失败。
- 采样：greedy（temperature=0），模型/种子固定。

## 3. 全量结果

{_summary_table(rows)}

## 4. 校验与异常

- 全部 cell 的 `parity_ok` 均为 True：{all(r.parity_ok for r in rows)}。
- 引擎驱动 cell 结束后块账目检查（free+used==total 与 batch 的 used==0）通过后可
  产生数据；任何泄漏/账目不齐会直接抛错终止测量，不会混进结果。

## 5. Known Limitations

- CPU 单机数字只用于验证流水线正确性；真正的吞吐/延迟数字必须来自云端 GPU 跑次。
- prefix 与 batch 共用 EngineCore 驱动（V2/V3 合并），两效果的吞吐与 KV 利用率
  在同一运行的两个正交观察量里分别呈现，不能拆分出"独立的两套实现"做对照。
- prefix cache 命中只发生在同一引擎实例内（本 benchmark 每 cell 新建引擎），
  sp workload 的命中与 TTFT 下降都限定在 cell 内部的多请求共享前缀上。
- HF 驱动无 prefill/decode 拆段计时，其 TTFT/TPOT/ITL 列为 N/A。
- 顺序驱动的 TTFT≈prefill 耗时（不含首 token 采样），是近似值。
- block_size 消融（8/16/32/64）不在本 Task 的自动化矩阵内，作为后续工作留档。

{charts_md}
"""
    text += _insights(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="LiteInfer benchmark 报告")
    p.add_argument("--csv", required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--charts-dir", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    charts = Path(args.charts_dir) if args.charts_dir else None
    build_report(Path(args.csv), Path(args.env), charts, Path(args.out))
    print(f"报告已生成: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())