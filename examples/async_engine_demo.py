"""Task 09 最小运行示例（asyncio 层）：AsyncEngine 的并发流式生成与取消。

展示三件事：

1. 多个请求**并发**流式生成：token 边算边吐，而不是等全部完成再返回；
2. 生成中途取消（等价于客户端断连）：请求进入 CANCELLED，物理块**立刻**归还；
3. 流式拼接的结果与 ``CachedGenerator``（同步 + 连续 KV）逐字一致——
   asyncio 只是改变了"何时把 token 交给调用方"，不改变算出来的东西。

运行（仓库根目录）：

    set "HF_HOME=D:/LiteInfer/hf_cache"
    set PYTHONPATH=d:/LiteInfer
    python examples/async_engine_demo.py --max-tokens 8
"""

from __future__ import annotations

import argparse
import asyncio

import torch

from liteinfer import EngineConfig
from liteinfer.engine import AsyncEngine, EngineCore, RequestStatus
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


async def _stream_one(engine: AsyncEngine, label: str, prompt: str, params) -> str:
    """消费一条流：拿到一个 token 就打印一个，最后返回拼接文本。"""
    stream = await engine.generate(prompt, params)
    parts: list[str] = []
    async for chunk in stream:
        if chunk.text:
            parts.append(chunk.text)
            print(f"  [{label}] +{chunk.text!r}", flush=True)
    return "".join(parts)


async def _demo_cancel(engine: AsyncEngine, params) -> bool:
    """读到第 3 个 token 就"拔网线"，验证 docs/02 §10 的 KV 回收。"""
    print("\n[取消] 生成 3 个 token 后断开连接")
    request_id = await engine.submit(PROMPTS[0], params)
    iterator = engine.stream(request_id).__aiter__()
    seen = 0
    async for chunk in iterator:
        if chunk.text:
            seen += 1
            print(f"  [cancel] +{chunk.text!r}", flush=True)
        if seen >= 3:
            break
    await iterator.aclose()  # 客户端断连

    # 取消是"投递命令 + 引擎循环下一轮执行"，给循环一点时间把它消费掉
    await asyncio.sleep(0.2)

    request = engine.core.get_request(request_id)
    blocks = engine.core.runner.paged.num_blocks_used
    print(f"  status={request.status.value} 已生成={request.output_tokens} "
          f"used_blocks={blocks}")
    return request.status == RequestStatus.CANCELLED and blocks == 0


async def amain(args: argparse.Namespace) -> int:
    cfg = EngineConfig(
        device="cpu",
        dtype=torch.float32,  # CPU 必须 FP32（补充条款 A2）
        max_new_tokens=args.max_tokens,
        scheduler=SchedulerConfig(
            max_num_seqs=args.max_num_seqs, max_num_batched_tokens=2048
        ),
    )
    loaded = load_minimal_from_hf(cfg)
    if loaded.tokenizer is None:
        raise RuntimeError("load_minimal_from_hf 未返回 tokenizer")
    eos_ids = resolve_eos_ids(loaded.hf_model, loaded.tokenizer)

    core = EngineCore(loaded.minimal, loaded.tokenizer, cfg, eos_ids=eos_ids)
    engine = AsyncEngine(core)
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    all_ok = True
    try:
        print(f"[并发流式] {len(PROMPTS)} 个请求，每请求的 token 到达时立即打印")
        texts = await asyncio.gather(
            *(_stream_one(engine, f"p{i}", p, params) for i, p in enumerate(PROMPTS))
        )
        print(f"\n[结束] used_blocks={core.runner.paged.num_blocks_used} "
              f"free_blocks={core.runner.paged.num_blocks_free}")

        all_ok = all_ok and await _demo_cancel(engine, params)

        # 与同步 + 连续 KV 的参照实现逐字对照
        gen = CachedGenerator(loaded.minimal, loaded.tokenizer, cfg, eos_ids=eos_ids)
        for prompt, text in zip(PROMPTS, texts):
            reference = gen.generate(prompt, params).text
            same = text == reference
            all_ok = all_ok and same
            print(f"\n[parity] {prompt}")
            print(f"  async stream : {text!r}")
            print(f"  CachedGen    : {reference!r}")
            print(f"  identical    : {same}")
    finally:
        await engine.shutdown()

    print(f"\n[final] 与 CachedGenerator 全部一致 + 取消后零泄漏: {all_ok} -> "
          f"{'OK' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Task 09：AsyncEngine 并发流式与取消示例")
    parser.add_argument("--max-tokens", type=int, default=8, help="每个请求最多生成多少 token")
    parser.add_argument("--max-num-seqs", type=int, default=2, help="并发序列上限")
    args = parser.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
