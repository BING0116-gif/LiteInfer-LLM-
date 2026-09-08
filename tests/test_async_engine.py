"""Task 09 测试：AsyncEngine（asyncio 流式 / 取消 / 不阻塞事件循环 / 块回收）。

两层：

1. 快测（默认 `pytest -q`）：用 ``_fakes.FakeLM``（会把 K/V 真的写进块表）驱动
   **真实的** EngineCore，验证 AsyncEngine 的四件事：
   - 流式产出的 token 顺序 / 内容与同步 ``EngineCore.run()`` 完全一致；
   - 多条流并发时互不串台；
   - 客户端断连（提前关闭生成器）会取消请求并归还物理块；
   - 阻塞前向没有焊死事件循环（必须有线程卸载）。

2. 真模型测试（marker=model）：流式拼接结果与 ``CachedGenerator`` 逐字一致。
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import pytest
import torch

from _fakes import make_core, simulate
from liteinfer import EngineConfig
from liteinfer.engine import AsyncEngine, EngineCore, RequestStatus
from liteinfer.sampling.params import SamplingParams

# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def _running(**kw):
    """起一个 AsyncEngine，退出时必定 shutdown（避免后台 task 泄漏到别的测试）。"""
    engine = AsyncEngine(make_core(**kw))
    try:
        yield engine
    finally:
        await engine.shutdown()


async def _wait_until(predicate, timeout: float = 3.0, interval: float = 0.01) -> bool:
    """轮询等待某个条件成立。

    为什么需要它：取消是"投递命令 + 引擎循环下一轮执行"，不是同步生效的。
    用 sleep 硬等会让测试要么脆弱要么很慢，轮询到成立即返回两头都照顾。
    """
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()


async def _collect(engine: AsyncEngine, prompt: str) -> str:
    stream = await engine.generate(prompt)
    return "".join([chunk.text async for chunk in stream])


def _expected(start: int, max_tokens: int) -> str:
    return "".join(str(t) for t in simulate(start, max_tokens))


# --------------------------------------------------------------------------- #
# 快测 1：流式语义
# --------------------------------------------------------------------------- #


async def test_stream_yields_tokens_then_terminal_chunk() -> None:
    async with _running(max_new_tokens=4) as engine:
        stream = await engine.generate("2")
        chunks = [chunk async for chunk in stream]

    assert [c.text for c in chunks] == ["3", "4", "5", "6"]
    assert [c.index for c in chunks] == [0, 1, 2, 3]
    assert chunks[-1].finished is True
    assert chunks[-1].finish_reason == "length"
    # 非终态 chunk 不能提前带 finish_reason，否则前端会以为生成结束了
    assert all(c.finish_reason is None for c in chunks[:-1])


async def test_stream_matches_sync_engine_run() -> None:
    """流式只是"把同步结果拆开送"，拼接后必须与同步路径逐字一致。"""
    async with _running(max_new_tokens=4) as engine:
        text = await _collect(engine, "2")

    core = make_core(max_new_tokens=4)
    request_id = core.submit("2")
    assert text == core.run()[request_id].text == _expected(2, 4)


async def test_eos_terminal_chunk_carries_no_token() -> None:
    """EOS 命中那一步不产出 token：终态 chunk 的 token_id 必须是 None。"""
    engine = AsyncEngine(make_core(max_new_tokens=8, eos={5}))
    try:
        stream = await engine.generate("2")  # 产出 3,4，下一个是 5(=eos)
        chunks = [chunk async for chunk in stream]
    finally:
        await engine.shutdown()

    assert [c.text for c in chunks] == ["3", "4", ""]
    assert chunks[-1].token_id is None
    assert chunks[-1].finish_reason == "eos"


async def test_concurrent_streams_stay_independent() -> None:
    async with _running(max_new_tokens=4) as engine:
        results = await asyncio.gather(
            *(_collect(engine, p) for p in ("2", "5", "9"))
        )
    assert list(results) == [_expected(2, 4), _expected(5, 4), _expected(9, 4)]


# --------------------------------------------------------------------------- #
# 快测 2：取消与块回收（docs/02 §10）
# --------------------------------------------------------------------------- #


async def test_early_close_cancels_and_frees_blocks() -> None:
    """客户端断连（提前关闭流）必须取消请求并归还物理块。"""
    async with _running(max_new_tokens=8) as engine:
        request_id = await engine.submit("2")
        iterator = engine.stream(request_id).__aiter__()
        first = await iterator.__anext__()
        assert first.text == "3"

        # 在生成中途"拔网线"
        await iterator.aclose()

        assert await _wait_until(
            lambda: engine.core.get_request(request_id).status
            == RequestStatus.CANCELLED
        )
        assert engine.core.runner.paged.num_blocks_used == 0


async def test_explicit_abort_while_running() -> None:
    async with _running(max_new_tokens=8) as engine:
        request_id = await engine.submit("2")
        assert await _wait_until(
            lambda: engine.core.runner.paged.num_blocks_used > 0
        ), "生成过程中应该真的占用了物理块"

        await engine.abort(request_id)
        assert engine.core.get_request(request_id).status == RequestStatus.CANCELLED
        assert engine.core.runner.paged.num_blocks_used == 0


async def test_blocks_reclaimed_after_normal_completion() -> None:
    async with _running(max_new_tokens=4) as engine:
        await _collect(engine, "2")
        assert engine.core.runner.paged.num_blocks_used == 0


async def test_shutdown_terminates_in_flight_stream() -> None:
    """关停时不能把消费方永久挂起：每条流都要收到终态 chunk。"""
    engine = AsyncEngine(make_core(max_new_tokens=32, sleep_s=0.005))
    stream = await engine.generate("2")
    iterator = stream.__aiter__()
    await iterator.__anext__()

    await engine.shutdown()

    chunks = []
    async for chunk in iterator:  # 引擎循环已停，这里必须能收敛
        chunks.append(chunk)
        if chunk.finished:
            break
    await iterator.aclose()

    assert chunks[-1].finished is True
    assert chunks[-1].finish_reason == "cancelled"
    assert engine.core.runner.paged.num_blocks_used == 0


# --------------------------------------------------------------------------- #
# 快测 3：引擎循环本身
# --------------------------------------------------------------------------- #


async def test_event_loop_not_blocked_by_forward() -> None:
    """阻塞前向必须卸载到线程：事件循环要能在生成期间继续调度别的协程。

    这是 SSE 能"逐 token 到达"的前提——若 step 直接跑在事件循环里，
    下面这个 ticker 一次都轮不到执行。
    """
    async with _running(max_new_tokens=4, sleep_s=0.02) as engine:
        ticks = 0
        stop = False

        async def ticker() -> None:
            nonlocal ticks
            while not stop:
                await asyncio.sleep(0)
                ticks += 1

        task = asyncio.create_task(ticker())
        text = await _collect(engine, "2")
        stop = True
        task.cancel()

    assert text == _expected(2, 4)
    assert ticks > 0, "前向阻塞了事件循环（step 没有被卸载到线程）"


async def test_submit_propagates_validation_error() -> None:
    """prompt 超过 token budget 时 submit 要 fail fast，不能静默排队饿死。"""
    async with _running(max_new_tokens=4, max_num_batched_tokens=2) as engine:
        with pytest.raises(ValueError):
            await engine.submit("12345")


async def test_stream_of_unknown_request_raises() -> None:
    async with _running(max_new_tokens=4) as engine:
        with pytest.raises(KeyError):
            engine.stream("not-a-request")


async def test_start_is_idempotent() -> None:
    async with _running(max_new_tokens=4) as engine:
        await engine.start()
        await engine.start()
        assert engine.is_running()
        assert engine.pending_streams == 0
        # 引擎循环空闲时不会吃掉 CPU：没有命令就阻塞在队列上
        assert await _wait_until(lambda: True)


# --------------------------------------------------------------------------- #
# 真模型（marker=model）
# --------------------------------------------------------------------------- #

PROMPT = "The capital of France is"
N_TOKENS = 8


@pytest.fixture(scope="module")
def cfg() -> EngineConfig:
    return EngineConfig(device="cpu", dtype=torch.float32, max_new_tokens=N_TOKENS)


@pytest.fixture(scope="module")
def loaded(cfg):
    from liteinfer.model.minimal.weights import load_minimal_from_hf

    return load_minimal_from_hf(cfg)


@pytest.fixture(scope="module")
def eos(loaded):
    from liteinfer.model.eos import resolve_eos_ids

    return resolve_eos_ids(loaded.hf_model, loaded.tokenizer)


@pytest.mark.model
async def test_real_model_stream_matches_cached_generator(loaded, cfg, eos) -> None:
    """Qwen2.5-0.5B：流式拼接的文本必须与 CachedGenerator（连续 KV）逐字一致。"""
    from liteinfer.model.cached_generator import CachedGenerator

    params = SamplingParams(max_tokens=N_TOKENS, temperature=0.0)
    core = EngineCore(loaded.minimal, loaded.tokenizer, cfg, eos_ids=eos)
    engine = AsyncEngine(core)
    try:
        stream = await engine.generate(PROMPT, params)
        texts = [chunk.text async for chunk in stream]
    finally:
        await engine.shutdown()

    cached = CachedGenerator(loaded.minimal, loaded.tokenizer, cfg, eos_ids=eos).generate(
        PROMPT, params
    )
    assert "".join(texts) == cached.text
    assert len(texts) > 1, "应当真的流式吐了多个 chunk（否则等于退化成一次性返回）"
    assert core.runner.paged.num_blocks_used == 0
