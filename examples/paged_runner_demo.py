"""Task 08 最小运行示例：ModelRunner 接入 Paged KV 后的多请求生成。

展示三件事：
1. 多请求并发生成，每个请求一张 BlockTable（物理块来自**同一个**共享池）；
2. 生成过程中块被真实占用，请求结束后**全部归还**（零泄漏）；
3. 分页 KV 的产出与 Task 04 的连续 KV（CachedGenerator）逐字一致——
   分页只换存储方式，不换数值。

运行（仓库根目录下）：

    set "HF_HOME=D:/LiteInfer/hf_cache"
    set PYTHONPATH=d:/LiteInfer
    python examples/paged_runner_demo.py
"""

from __future__ import annotations

import argparse

import torch

from liteinfer import EngineConfig
from liteinfer.engine import EngineCore
from liteinfer.model.cached_generator import CachedGenerator
from liteinfer.model.eos import resolve_eos_ids
from liteinfer.model.minimal.weights import load_minimal_from_hf
from liteinfer.sampling.params import SamplingParams
from liteinfer.scheduler.config import SchedulerConfig

PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "The chemical symbol for water is",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 08：Paged KV + ModelRunner 多请求示例")
    parser.add_argument("--max-tokens", type=int, default=16, help="每个请求最多生成多少 token")
    parser.add_argument("--max-num-seqs", type=int, default=2, help="并发序列上限（演示准入）")
    parser.add_argument("--block-size", type=int, default=16, help="每个物理块容纳多少 token")
    args = parser.parse_args()

    cfg = EngineConfig(
        device="cpu",
        dtype=torch.float32,  # CPU 必须 FP32（补充条款 A2）
        max_new_tokens=args.max_tokens,
        block_size=args.block_size,
        scheduler=SchedulerConfig(max_num_seqs=args.max_num_seqs, max_num_batched_tokens=2048),
    )

    loaded = load_minimal_from_hf(cfg)
    if loaded.tokenizer is None:
        raise RuntimeError("load_minimal_from_hf 未返回 tokenizer")
    eos_ids = resolve_eos_ids(loaded.hf_model, loaded.tokenizer)

    engine = EngineCore(loaded.minimal, loaded.tokenizer, cfg, eos_ids=eos_ids)
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    print(f"block_size={engine.runner.block_size} "
          f"num_blocks={engine.runner.paged.num_blocks_total} "
          f"pool={engine.runner.paged.nbytes / 1e6:.1f} MB")

    rids = [engine.submit(p, params) for p in PROMPTS]
    print(f"提交 {len(rids)} 个请求 -> waiting={engine.scheduler.num_waiting} "
          f"used_blocks={engine.runner.paged.num_blocks_used}")

    # 先跑一步：只有被准入的请求会占用物理块（演示"按需分配"）
    engine.step()
    print(f"step 1 之后: running={engine.scheduler.num_running} "
          f"used_blocks={engine.runner.paged.num_blocks_used}")

    outputs = engine.run()
    print(f"全部结束:   used_blocks={engine.runner.paged.num_blocks_used} "
          f"free_blocks={engine.runner.paged.num_blocks_free}")

    # 与连续 KV 的生成结果逐字对照
    gen = CachedGenerator(loaded.minimal, loaded.tokenizer, cfg, eos_ids=eos_ids)
    all_match = True
    for rid, prompt in zip(rids, PROMPTS):
        out = outputs[rid]
        cached = gen.generate(prompt, params)
        same = out.text == cached.text
        all_match = all_match and same
        print(f"\n[prompt] {prompt}")
        print(f"  paged     : {out.text!r}  ({out.output_tokens} tokens, {out.finish_reason})")
        print(f"  contiguous: {cached.text!r}  ({cached.output_tokens} tokens, {cached.finish_reason})")
        print(f"  identical : {same}   cache_bytes={out.cache_bytes}")

    leaked = engine.runner.paged.num_blocks_used != 0
    print(f"\n[final] 输出全部一致: {all_match} | 块泄漏: {leaked} -> "
          f"{'OK' if all_match and not leaked else 'FAIL'}")
    return 0 if all_match and not leaked else 1


if __name__ == "__main__":
    raise SystemExit(main())
