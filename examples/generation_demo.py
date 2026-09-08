"""Task 02 最小运行示例：手写自回归生成循环（不经过 generate()）。

用法：
    python examples/generation_demo.py --mode greedy
    python examples/generation_demo.py --mode sample --temperature 0.7 --top-p 0.9 --seed 42
"""

from __future__ import annotations

import argparse
import logging

from liteinfer import EngineConfig, parse_dtype
from liteinfer.model.generator import ManualGenerator
from liteinfer.sampling.params import SamplingParams


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LiteInfer Task 02: Manual Generation Loop")
    p.add_argument(
        "--prompt",
        default="用两句话解释什么是 KV Cache，以及它为什么能加速大模型推理。",
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--mode", choices=["greedy", "sample"], default="greedy")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=-1, help="-1 表示关闭")
    p.add_argument("--top-p", type=float, default=1.0, help="1.0 表示关闭")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model-id", default=None, help="覆盖默认 Qwen/Qwen2.5-0.5B")
    p.add_argument("--device", default="cpu", help="本机固定 cpu（补充条款 A1）")
    p.add_argument("--dtype", default="float32", help="本机固定 float32")
    p.add_argument("--local-only", action="store_true", help="只读缓存，不联网")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = build_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.mode == "greedy":
        params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)
    else:
        params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        )

    overrides = {}
    if args.model_id:
        overrides["model_id"] = args.model_id
    cfg = EngineConfig.from_env(
        device=args.device,
        dtype=parse_dtype(args.dtype),
        max_new_tokens=args.max_new_tokens,
        local_files_only=args.local_only,
        **overrides,
    )
    generator = ManualGenerator.from_config(cfg)
    out = generator.generate(args.prompt, params)

    print("=" * 60)
    print(f"[mode]   {args.mode}  params={params}")
    print("[prompt]", args.prompt)
    print("[output]", out.text)
    print("-" * 60)
    print(
        f"prompt_tokens={out.prompt_tokens} output_tokens={out.output_tokens} "
        f"finish_reason={out.finish_reason} latency={out.latency_s:.2f}s "
        f"tokens/s={out.tokens_per_s:.2f}"
    )
    print(f"device={out.device} dtype={out.dtype}")
    print("注：本阶段无 KV Cache（每步全序列 forward），CPU 上的 tokens/s "
          "仅用于逻辑验证，不代表 GPU serving 性能")
    print("=" * 60)


if __name__ == "__main__":
    main()
