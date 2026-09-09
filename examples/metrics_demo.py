"""Task 10：Metrics + Tracing 最小示例——一次请求输出完整时间线。

运行（Windows CMD）::

    set "HF_HOME=D:/LiteInfer/hf_cache"
    set "PYTHONPATH=d:/LiteInfer"
    python examples/metrics_demo.py --max-tokens 16

内容：
1. 用 EngineCore 跑一个 greedy 请求；
2. 打印请求级指标（TTFT / TPOT / ITL p50/p95/max / E2E / tokens_per_s）；
3. 打印完整事件时间线（enqueue -> prefill -> token x N -> finished）；
4. 打印全局指标快照（waiting/running、KV utilization、吞吐、显存 N/A）。
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="LiteInfer metrics/tracing demo")
    parser.add_argument("--prompt", default="用一句话介绍 KV Cache。")
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    # 延迟导入：argparse/--help 不用加载 torch
    from liteinfer import EngineConfig, EngineCore
    from liteinfer.sampling.params import SamplingParams

    cfg = EngineConfig.from_env(
        device="cpu",  # 开发机无 GPU；上云用 LITEINFER_DEVICE=cuda 覆盖
        max_new_tokens=args.max_tokens,
    )
    print(f"[load] model={cfg.model_id} device={cfg.device} dtype={cfg.dtype}")
    core = EngineCore.from_config(cfg)

    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    rid = core.submit(args.prompt, params)
    out = core.run()[rid]

    m = core.metrics.get(rid)
    assert m is not None

    print()
    print("=" * 64)
    print(f"[text]   {out.text}")
    print(f"[finish] {m.finish_reason}  prompt_tokens={m.prompt_tokens} "
          f"output_tokens={m.output_tokens}")
    print("-" * 64)
    print("[request metrics]")
    print(f"  TTFT          = {m.ttft_s * 1000:8.2f} ms   (enqueue -> first token)")
    if m.tpot_s is not None:
        print(f"  TPOT          = {m.tpot_s * 1000:8.2f} ms   (mean, n-1 intervals)")
    else:
        print("  TPOT          =      N/A        (single-token request)")
    if m.itl_p50_s is not None:
        print(f"  ITL p50/p95/max = {m.itl_p50_s * 1000:.2f} / "
              f"{m.itl_p95_s * 1000:.2f} / {m.itl_max_s * 1000:.2f} ms")
    print(f"  E2E           = {m.e2e_s:8.4f} s")
    print(f"  tokens_per_s  = {m.tokens_per_s:8.2f}    (E2E 口径)")
    print(f"  prefill/decode = {m.prefill_latency_s:.4f} / {m.decode_latency_s:.4f} s")
    print(f"  kv bytes used = {m.cache_bytes} B")

    print("-" * 64)
    print("[request trace]  (完整时间线)")
    print(core.trace_of(rid).render())

    print("-" * 64)
    print("[global metrics snapshot]")
    snap = core.metrics_snapshot()
    print(f"  requests total/finished/cancelled = "
          f"{snap['requests_total']}/{snap['requests_finished']}/{snap['requests_cancelled']}")
    print(f"  waiting/running = {snap['num_waiting']}/{snap['num_running']}")
    print(f"  kv blocks used/total = {snap['kv_blocks_used']}/{snap['kv_blocks_total']} "
          f"(utilization={snap['kv_utilization']})")
    print(f"  output tokens/s (window {snap['throughput_window_s']:.0f}s) = "
          f"{snap['output_tokens_per_s']:.2f}")
    print(f"  gpu_memory_mb = {snap['gpu_memory_mb_display']}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    # Windows GBK 控制台：按 UTF-8 重配 stdout，避免中文/宽字符乱码
    if os.name == "nt" and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
