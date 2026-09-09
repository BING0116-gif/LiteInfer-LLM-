"""Task 12 快测：benchmark 工具链（无模型，全部秒级）。

覆盖三层契约（以结果文件为界，各自可独立单测）：
- 测量层（liteinfer_benchmark）：workload 解析 / 提示词构造 / 汇总纯函数 /
  CSV round-trip / 引擎配置尺寸 / 实验记录；
- 图表层（benchmark_charts）：由合成数据真实渲染 PNG（不依赖真模型）；
- 报告层（benchmark_report）：由合成 CSV/env 组装 markdown。

不加载任何模型：真模型端到端由 examples/benchmark_demo.py --smoke 承担
（可在有模型的机子上人工跑；CI 里只跑这里的快测）。
"""

from __future__ import annotations

import json

import pytest
import torch

from liteinfer import EngineConfig

from benchmark import (
    benchmark_charts,
    benchmark_report,
    liteinfer_benchmark as lb,
)
from _fakes import FakeTokenizer

# --------------------------------------------------------------------------- #
# workload 规格
# --------------------------------------------------------------------------- #


def test_workload_spec_known_names() -> None:
    for name in lb.WORKLOAD_NAMES:
        ws = lb.workload_spec(name, smoke=False)
        assert ws["kind"] in (
            "decode-heavy",
            "prefill-heavy",
            "typical",
            "shared-prefix",
        )
        assert ws["max_tokens"] >= 1


def test_workload_spec_smoke_shape_preserved() -> None:
    """smoke 必须保留 full 的形状（解码重/预填充重/常规/共享前缀）。"""
    for name in lb.WORKLOAD_NAMES:
        full = lb.workload_spec(name, smoke=False)
        smoke = lb.workload_spec(name, smoke=True)
        assert full["kind"] == smoke["kind"]
        assert smoke["max_tokens"] <= full["max_tokens"]


def test_workload_spec_unknown_fails() -> None:
    with pytest.raises(ValueError):
        lb.workload_spec("nope", smoke=False)


def test_planned_prompt_len() -> None:
    assert lb.planned_prompt_len({"prompt_len": 64, "max_tokens": 256}) == 64
    sp = lb.planned_prompt_len({"kind": "shared-prefix", "shared_len": 2048, "tail_len": 64})
    assert sp == 2112


# --------------------------------------------------------------------------- #
# 提示词构造（FakeTokenizer：单字符=单 token，只认数字）
# --------------------------------------------------------------------------- #


def test_build_prompt_text_macro_count() -> None:
    tok = FakeTokenizer()
    text = lb.build_prompt_text(tok, 8, macro="12")
    # 数字宏 "12" 重复 -> "12121212"，再编码应恰好 8 个 token
    enc = tok(text, return_tensors="pt")["input_ids"]
    assert enc.shape[1] == 8
    assert text == "12121212"


def test_build_prompt_text_requires_positive() -> None:
    with pytest.raises(ValueError):
        lb.build_prompt_text(FakeTokenizer(), 0, macro="1")


def test_build_sp_prompts_share_head_and_differ() -> None:
    tok = FakeTokenizer()
    prompts = lb.build_sp_prompts(
        tok, shared_len=4, tail_len=2, count=3, tail_suffix="5{i}",
        macro="12",
    )
    assert len(prompts) == 3
    heads = [lb.build_prompt_text(tok, 4, macro="12")] * 3
    for head, prompt in zip(heads, prompts):
        assert prompt == head + prompt[len(head):]  # 每个都以同一共享前缀开头
    assert prompts[0] != prompts[1]  # tail_suffix 里带序号 -> 尾缀互不相同


# --------------------------------------------------------------------------- #
# 纯函数：分位数 / 均值 / cell 汇总
# --------------------------------------------------------------------------- #


def test_percentile_basic() -> None:
    assert lb.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert lb.percentile([1.0, 2.0, 3.0, 4.0], 0.95) == pytest.approx(3.85)
    assert lb.percentile([5.0], 0.5) == 5.0


def test_percentile_none_aware() -> None:
    assert lb.percentile([None, 1.0, None], 0.5) == 1.0
    assert lb.percentile([], 0.5) is None
    assert lb.percentile([None, None], 0.5) is None


def test_mean_opt() -> None:
    assert lb.mean_opt([None, 2.0, None]) == 2.0
    assert lb.mean_opt([None]) is None


def _fake_reqs(n: int, ttft: float, out: int, e2e: float) -> list[dict]:
    # 每请求都是同构的合成测量；status 字段只对引擎 cell 有意义
    return [
        {
            "text": f"out-{i}",
            "prompt_tokens": 16,
            "output_tokens": out,
            "finish_reason": "length",
            "ttft_s": ttft,
            "tpot_s": 0.4,
            "itl_p50": 0.35,
            "itl_p95": 0.5,
            "e2e_s": e2e,
            "cache_bytes": 4096,
            "prefix_hit_tokens": 0,
            "status": "finished",
        }
        for i in range(n)
    ]


def test_aggregate_cell_pure() -> None:
    row = lb.aggregate_cell(
        wall_s=10.0,
        reqs=_fake_reqs(4, ttft=1.0, out=8, e2e=9.0),
        engine="kv",
        workload="decode",
        concurrency=4,
        output_planned=8,
        kv_peak_blocks=None,
        kv_used_end=None,
        kv_free_end=None,
        kv_total_blocks=None,
        prefix_cached_blocks=None,
        max_num_seqs=None,
        max_num_batched_tokens=None,
        num_blocks=None,
        block_size=16,
        parity_ok=True,
    )
    assert row.num_requests == 4
    assert row.output_tokens_total == 32
    assert row.out_tokens_per_s == pytest.approx(3.2)
    assert row.ttft_p50 == pytest.approx(1.0)
    assert row.e2e_p50 == pytest.approx(9.0)
    assert row.kv_util_peak is None  # 顺序驱动无块池


def test_aggregate_cell_engine_missing_cols_none() -> None:
    # prefix_hit_tokens 缺失的 req（顺序驱动形状）汇总时均值为 None，不是 0
    row = lb.aggregate_cell(
        wall_s=5.0,
        reqs=[dict(_fake_reqs(1, 0.5, 4, 4.0)[0], prefix_hit_tokens=None)],
        engine="nokv",
        workload="typical",
        concurrency=1,
        output_planned=4,
        kv_peak_blocks=None, kv_used_end=None, kv_free_end=None, kv_total_blocks=None,
        prefix_cached_blocks=None,
        max_num_seqs=None, max_num_batched_tokens=None, num_blocks=None,
        block_size=16, parity_ok=True,
    )
    assert row.prefix_hit_tokens_mean is None
    assert row.kv_active_blocks_end is None


# --------------------------------------------------------------------------- #
# 引擎配置尺寸（长 prompt 池必须够大、token budget 必须放得下）
# --------------------------------------------------------------------------- #


def test_engine_cfg_sizing_for_long_prompt() -> None:
    base = EngineConfig(device="cpu", dtype=torch.float32, block_size=16)
    ecfg = lb._engine_cfg(
        base, concurrency=8, prompt_len=2048, max_tokens=32, enable_prefix_cache=False
    )
    assert ecfg.scheduler.max_num_batched_tokens >= 2048
    # 每序列 ceil((2048+32)/16)+4 = 134 块；x8 -> 1072，必须正好等于 config
    per_seq = -(-(2048 + 32) // 16) + 4
    assert ecfg.num_blocks == per_seq * 8
    assert ecfg.block_size == 16


# --------------------------------------------------------------------------- #
# CSV round-trip
# --------------------------------------------------------------------------- #


def test_csv_roundtrip(tmp_path) -> None:
    rows = [
        lb.aggregate_cell(
            wall_s=3.5,
            reqs=_fake_reqs(2, ttft=0.5, out=4, e2e=3.0),
            engine="batch", workload="sp", concurrency=2, output_planned=4,
            kv_peak_blocks=12, kv_used_end=0, kv_free_end=42, kv_total_blocks=54,
            prefix_cached_blocks=None,
            max_num_seqs=2, max_num_batched_tokens=2048, num_blocks=54,
            block_size=16, parity_ok=True,
        ),
        lb.aggregate_cell(
            wall_s=6.0,
            reqs=_fake_reqs(1, ttft=None, out=4, e2e=5.5),
            engine="hf", workload="decode", concurrency=1, output_planned=4,
            kv_peak_blocks=None, kv_used_end=None, kv_free_end=None, kv_total_blocks=None,
            prefix_cached_blocks=None,
            max_num_seqs=None, max_num_batched_tokens=None, num_blocks=None,
            block_size=16, parity_ok=True,
        ),
    ]
    csv_path = tmp_path / "results.csv"
    lb.write_rows_csv(csv_path, rows)
    back = lb.read_rows_csv(csv_path)
    assert [r.to_dict() for r in back] == [r.to_dict() for r in rows]


def test_csv_fieldnames_stable(tmp_path) -> None:
    """列顺序是跨脚本契约，改名/增列要连 charts/report 一起核对。"""
    csv_path = tmp_path / "results.csv"
    lb.write_rows_csv(csv_path, [])
    with open(csv_path, encoding="utf-8") as fh:
        assert fh.readline().strip() == ",".join(f.name for f in lb.fields(lb.CellRow))


# --------------------------------------------------------------------------- #
# 实验记录
# --------------------------------------------------------------------------- #


def test_env_record_structure() -> None:
    env = lb.env_record(
        model_id="Qwen/Qwen2.5-0.5B",
        device="cpu",
        dtype=torch.float32,
        smoke=True,
        workload_names=["decode", "sp"],
        engine_names=["kv"],
        concurrency=[1, 2],
        block_size=16,
        num_requests_note="note",
    )
    assert env["model_id"] == "Qwen/Qwen2.5-0.5B"
    assert env["device"] == "cpu"
    assert env["smoke"] is True
    assert set(env["workloads"]) == {"decode", "sp"}
    assert env["concurrency"] == [1, 2]
    assert "timestamp_utc" in env


# --------------------------------------------------------------------------- #
# 图表：用合成数据真实渲染 PNG（matplotlib 已装）
# --------------------------------------------------------------------------- #


def _synthetic_rows() -> list[lb.CellRow]:
    rows = []
    for engine in ("kv", "batch", "prefix"):
        for wl in ("decode", "prefill", "typical", "sp"):
            for conc in (1, 2):
                hit = 32 if engine == "prefix" and wl == "sp" else 0
                reqs = _fake_reqs(conc, ttft=0.5 + conc * 0.1, out=8, e2e=8.0)
                for r in reqs:
                    r["prefix_hit_tokens"] = hit if hit else None
                kv_peak = 16 * conc if engine in ("batch", "prefix") else None
                rows.append(
                    lb.aggregate_cell(
                        wall_s=5.0,
                        reqs=reqs,
                        engine=engine, workload=wl, concurrency=conc, output_planned=8,
                        kv_peak_blocks=kv_peak,
                        kv_used_end=0 if engine == "batch" else kv_peak,
                        kv_free_end=54 - (kv_peak or 0),
                        kv_total_blocks=54,
                        prefix_cached_blocks=kv_peak if engine == "prefix" else None,
                        max_num_seqs=conc,
                        max_num_batched_tokens=2048,
                        num_blocks=54, block_size=16, parity_ok=True,
                    )
                )
    return rows


def test_charts_render_pngs(tmp_path) -> None:
    csv_path = tmp_path / "results.csv"
    lb.write_rows_csv(csv_path, _synthetic_rows())
    env = lb.env_record(
        model_id="Qwen/Qwen2.5-0.5B", device="cpu", dtype=torch.float32,
        smoke=True, workload_names=["decode", "prefill", "typical", "sp"],
        engine_names=["kv", "batch", "prefix"], concurrency=[1, 2], block_size=16,
        num_requests_note="",
    )
    pngs = benchmark_charts.render_charts(csv_path, env, tmp_path / "charts")
    assert len(pngs) >= 3  # 至少吞吐/TTFT/E2E + prefix 图
    for png in pngs:
        data = png.read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic，确保不是空文件


# --------------------------------------------------------------------------- #
# 报告：由合成 CSV 组装 markdown
# --------------------------------------------------------------------------- #


def test_report_generation(tmp_path) -> None:
    csv_path = tmp_path / "results.csv"
    lb.write_rows_csv(csv_path, _synthetic_rows())
    env_path = tmp_path / "env.json"
    env = lb.env_record(
        model_id="Qwen/Qwen2.5-0.5B", device="cpu", dtype=torch.float32,
        smoke=True, workload_names=["decode", "sp"],
        engine_names=["kv", "batch", "prefix"], concurrency=[1, 2], block_size=16,
        num_requests_note="",
    )
    env_path.write_text(json.dumps(env, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "report.md"
    text = benchmark_report.build_report(csv_path, env_path, None, out)
    assert "N/A (no GPU)" in text  # 无 GPU 显存列必须 N/A 而非 0
    assert "## " in text
    assert "| engine | workload |" in text
    assert "sp" in text
    assert out.exists() and out.stat().st_size > 1000


def test_report_insights_only_from_data(tmp_path) -> None:
    """不提供 batch/prefix 的 sp 数据时，结论节不能产生编造的对比。"""
    rows = [
        r
        for r in _synthetic_rows()
        if not (r.engine in ("batch", "prefix") and r.workload == "sp")
    ]
    csv_path = tmp_path / "results.csv"
    lb.write_rows_csv(csv_path, rows)
    env = lb.env_record(
        model_id="M", device="cpu", dtype=torch.float32, smoke=True,
        workload_names=["decode", "sp"], engine_names=["kv", "batch", "prefix"],
        concurrency=[1, 2], block_size=16, num_requests_note="",
    )
    env_path = tmp_path / "env.json"
    env_path.write_text(json.dumps(env, ensure_ascii=False), encoding="utf-8")
    text = benchmark_report.build_report(csv_path, env_path, None, tmp_path / "r.md")
    # 报告输出了"数据不足"行而不是拍脑袋数字
    assert "数据不足" in text or "缺少" in text