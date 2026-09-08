"""Task 05 最小运行示例：EngineCore 同时维护多个请求。

运行方式（仓库根目录）：

    set PYTHONPATH=d:\\LiteInfer
    set "HF_HOME=D:\\LiteInfer\\hf_cache"
    python examples/engine_demo.py

演示四件事：
1. 一次性 submit 多个 prompt，引擎为每个请求分配专属 KV 缓存；
2. ``run()`` 用"每个 step 推进所有在飞请求一个 token"的时间片交错，
   同时把多个请求跑到结束（CPU 上不是真正的 batch 合并，属 Task 08）；
3. 每个请求独立产出结果，互不干扰；
4. 引擎产出的文本、终止原因、token 数与 CachedGenerator 逐字一致
   （这里直接复用 CachedGenerator 作为参照打印对照）。
"""

from __future__ import annotations

import torch

from liteinfer import CachedGenerator, EngineConfig, EngineCore
from liteinfer.model.eos import resolve_eos_ids
from liteinfer.model.minimal.weights import load_minimal_from_hf
from liteinfer.sampling.params import SamplingParams

PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "The chemical symbol for water is",
]
N_TOKENS = 16


def main() -> None:
    # CPU + FP32：本机环境硬性约束，device/dtype 只从 EngineConfig 进入
    cfg = EngineConfig(device="cpu", dtype=torch.float32, max_new_tokens=N_TOKENS)

    # 加载一次权重，同时构造引擎与参照生成器（避免重复加载）
    loaded = load_minimal_from_hf(cfg)
    eos_ids = resolve_eos_ids(loaded.hf_model, loaded.tokenizer)
    engine = EngineCore(loaded.minimal, loaded.tokenizer, cfg, eos_ids=eos_ids)
    ref = CachedGenerator(loaded.minimal, loaded.tokenizer, cfg, eos_ids=eos_ids)

    params = SamplingParams(max_tokens=N_TOKENS, temperature=0.0)

    print(f"[engine] 提交 {len(PROMPTS)} 个并发请求 ...")
    ids = [engine.submit(p, params) for p in PROMPTS]
    print(f"[engine] 提交完成，在飞请求: {len(engine.active_requests())}")

    outputs = engine.run()
    print(f"[engine] 全部结束，在飞请求: {len(engine.active_requests())}\n")

    all_match = True
    for rid, prompt in zip(ids, PROMPTS):
        out = outputs[rid]
        ref_out = ref.generate(prompt, params)
        match = out.text == ref_out.text
        all_match = all_match and match
        print(f"--- request {rid[:8]} ---")
        print(f"prompt : {prompt!r}")
        print(
            f"engine : {out.text!r} "
            f"(tokens={out.output_tokens}, finish={out.finish_reason}, "
            f"cached={out.cached_tokens}, {out.tokens_per_s:.2f} tok/s)"
        )
        print(f"ref    : {ref_out.text!r} (finish={ref_out.finish_reason})")
        print(f"match  : {match}\n")

    print(f"[check] 所有请求均与 CachedGenerator 逐字一致: {all_match}")
    print("[note] 本例验证引擎可同时维护多个请求并各自正确结束；真实 batch 合并见 Task 08。")


if __name__ == "__main__":
    main()
