"""Task 04 最小运行示例：KV Cache 版生成 vs 无缓存生成。

运行方式（仓库根目录）：

    set PYTHONPATH=d:\\LiteInfer
    python examples/kv_cache_demo.py

展示四件事：
1. 按模型形状预分配一份连续 KV 缓存，并算出每 token 的 KV 开销；
2. prefill 一次 + decode 逐步复用的两段耗时；
3. 有缓存 / 无缓存两条链输出**逐字一致**（不是"看起来像"）；
4. 同样的模型下，KV 复用带来的加速比（CPU 数字仅作逻辑验证）。
"""

from __future__ import annotations

import torch

from liteinfer import CachedGenerator, EngineConfig
from liteinfer.sampling.params import SamplingParams

PROMPT = "The capital of France is"
N_TOKENS = 24


def main() -> None:
    # CPU + FP32：本机环境硬性约束，dtype/device 只从 EngineConfig 进入
    cfg = EngineConfig(device="cpu", dtype=torch.float32, max_new_tokens=N_TOKENS)
    gen = CachedGenerator.from_config(cfg)

    num_layers, num_kv_heads, head_dim = gen._kv_dims
    per_token = 2 * num_layers * num_kv_heads * head_dim * torch.empty(
        0, dtype=gen.dtype
    ).element_size()
    print(
        f"[model] layers={num_layers} kv_heads={num_kv_heads} head_dim={head_dim} "
        f"dtype={gen.dtype}"
    )
    print(f"[cache] {per_token} B/token（容量按 prompt + max_new_tokens 预分配）")

    params = None  # None -> greedy + cfg.max_new_tokens

    # 预热：首次 forward 含线程池初始化/内存分配等一次性成本（实测冷启动比
    # 热态慢 3~5 倍），不预热就计时，测到的是"谁先跑"而不是"谁更快"
    warmup = SamplingParams(max_tokens=2, temperature=0.0)
    gen.generate(PROMPT, warmup, use_cache=True)
    gen.generate(PROMPT, warmup, use_cache=False)

    kv_out = gen.generate(PROMPT, params, use_cache=True)
    no_cache_out = gen.generate(PROMPT, params, use_cache=False)

    print("\n--- KV Cache ---")
    print(f"text    : {kv_out.text!r}")
    print(
        f"tokens  : prompt={kv_out.prompt_tokens} output={kv_out.output_tokens} "
        f"finish={kv_out.finish_reason}"
    )
    print(
        f"latency : total={kv_out.latency_s:.3f}s prefill={kv_out.prefill_latency_s:.3f}s "
        f"decode={kv_out.decode_latency_s:.3f}s -> {kv_out.tokens_per_s:.2f} tok/s"
    )
    print(
        f"cache   : cached_tokens={kv_out.cached_tokens} "
        f"bytes={kv_out.cache_bytes / 1024:.1f} KiB"
    )

    print("\n--- No Cache（同模型消融基线）---")
    print(f"text    : {no_cache_out.text!r}")
    print(
        f"latency : total={no_cache_out.latency_s:.3f}s "
        f"-> {no_cache_out.tokens_per_s:.2f} tok/s"
    )

    same = kv_out.text == no_cache_out.text
    speedup = no_cache_out.latency_s / kv_out.latency_s if kv_out.latency_s > 0 else 0.0
    print(f"\n[check] 输出一致: {same}")
    print(f"[check] 加速比  : {speedup:.2f}x")
    print("[note] CPU 上的加速比仅用于验证 KV 复用逻辑，不能写进简历/汇报。")


if __name__ == "__main__":
    main()
