"""Task 01 最小运行示例：HF Baseline（全项目唯一 generate() 调用点）。

用法：
    python examples/baseline_demo.py --prompt "你好" --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import logging

from liteinfer import EngineConfig, parse_dtype
from liteinfer.model.baseline import HFBaseline


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LiteInfer Task 01: HF Baseline")
    p.add_argument(
        "--prompt",
        default="用两句话解释什么是 KV Cache，以及它为什么能加速大模型推理。",
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
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
    baseline = HFBaseline.from_config(cfg)
    out = baseline.generate(args.prompt, max_new_tokens=cfg.max_new_tokens)

    print("=" * 60)
    print("[prompt]", args.prompt)
    print("[output]", out.text)
    print("-" * 60)
    print(
        f"prompt_tokens={out.prompt_tokens} output_tokens={out.output_tokens} "
        f"latency={out.latency_s:.2f}s tokens/s={out.tokens_per_s:.2f}"
    )
    print(f"device={out.device} dtype={out.dtype}")
    print("注：CPU 上的 tokens/s 仅用于逻辑验证，不代表 GPU serving 性能")
    print("=" * 60)


if __name__ == "__main__":
    main()
