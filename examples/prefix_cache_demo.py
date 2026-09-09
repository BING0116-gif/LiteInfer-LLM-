"""Task 11：Prefix Cache 最小运行示例——共享前缀 workload 中可观察 cache hit。

场景：多个请求携带相同 system prompt（共享前缀），第一个请求全量 prefill
并把完整块注册进哈希表；后续请求命中即收养物理块，只对后缀做 forward。

对照两组引擎（同一模型、同一请求序列）：
- prefix OFF：每次请求都全量 prefill；
- prefix ON ：第二个请求起 prefix_hit_tokens > 0，prefill 耗时下降。

运行（CPU）：
    set "HF_HOME=D:/LiteInfer/hf_cache"
    set "PYTHONPATH=d:/LiteInfer"
    python examples/prefix_cache_demo.py
"""

from __future__ import annotations

import argparse

import torch

from liteinfer import EngineConfig
from liteinfer.engine import EngineCore
from liteinfer.model.eos import resolve_eos_ids
from liteinfer.model.minimal.weights import load_minimal_from_hf
from liteinfer.sampling.params import SamplingParams

# 共享 system prompt：足够长（> 3 个 block），保证有完整块可缓存
SHARED_PREFIX = (
    "你是一个乐于助人的AI助手。请始终用简洁的中文回答用户的问题，"
    "并在回答末尾附上一句总结。回答时保持礼貌与专业。"
    "下面是几位用户依次提出的问题：\n"
)
QUESTIONS = ["天空为什么是蓝色的？", "什么是光合作用？", "水在几度结冰？"]


def run_engine(core: EngineCore, max_tokens: int) -> list[dict]:
    """依次提交 3 个共享前缀的请求（串行），返回每个请求的观测摘要。"""
    rows = []
    for q in QUESTIONS:
        rid = core.submit(
            SHARED_PREFIX + q, SamplingParams(max_tokens=max_tokens, temperature=0.0)
        )
        outs = core.run()
        o = outs[rid]
        rows.append(
            {
                "question": q,
                "prompt_tokens": o.prompt_tokens,
                "prefix_hit_tokens": o.prefix_hit_tokens,
                "prefill_ms": o.prefill_latency_s * 1e3,
                "text": o.text[:24],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-tokens", type=int, default=16)
    args = parser.parse_args()

    base_cfg = EngineConfig(device="cpu", dtype=torch.float32)
    loaded = load_minimal_from_hf(base_cfg)
    eos = resolve_eos_ids(loaded.hf_model, loaded.tokenizer)

    for label, enable in (("prefix OFF", False), ("prefix ON ", True)):
        # 两组用同一模型实例（省内存）；开关只差 EngineConfig.enable_prefix_cache
        cfg = EngineConfig(
            device="cpu", dtype=torch.float32,
            max_new_tokens=args.max_tokens, enable_prefix_cache=enable,
        )
        core = EngineCore(loaded.minimal, loaded.tokenizer, cfg, eos_ids=eos)
        rows = run_engine(core, args.max_tokens)
        print(f"\n===== {label} =====")
        for r in rows:
            print(
                f"[prompt={r['prompt_tokens']:>3} tok] hit={r['prefix_hit_tokens']:>3} tok | "
                f"prefill={r['prefill_ms']:>8.1f} ms | {r['question']} -> {r['text']}..."
            )
        if core.runner.prefix is not None:
            print("prefix stats:", core.runner.prefix.stats())


if __name__ == "__main__":
    main()
