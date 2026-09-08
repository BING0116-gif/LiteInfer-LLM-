"""Task 09 测试：OpenAI 兼容 HTTP 接口（非流式 / SSE 流式 / 取消 / 断连回收）。

用 ``httpx.AsyncClient + ASGITransport`` 直接打 ASGI 应用，而不是
``fastapi.testclient.TestClient``：TestClient 会把应用跑在**另一个线程的**事件循环里，
而 AsyncEngine 的队列与 future 必须和引擎循环同处一个 loop——跨 loop 结算 future
会直接挂死。同 loop 驱动也顺带让"客户端断连"可以被确定性地观察。

全部为快测：模型用 ``_fakes.FakeLM``，不下载任何权重。
"""

from __future__ import annotations

import asyncio
import socket
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
import pytest

from _fakes import FakeTokenizer, make_core, simulate
from liteinfer.engine import AsyncEngine, RequestStatus
from liteinfer.sampling.params import SamplingParams
from liteinfer.server import create_app
from liteinfer.server.app import _completion_sse, _render_chat_prompt
from liteinfer.server.schemas import ChatMessage
from liteinfer.server.sse import parse_sse_frames

MAX_TOKENS = 4
MODEL_ID = "fake-lm"


@asynccontextmanager
async def _serve(max_new_tokens: int = MAX_TOKENS, **kw) -> AsyncIterator[tuple]:
    """起一个带假引擎的应用（含 lifespan），yield ``(client, engine)``。"""
    engine = AsyncEngine(make_core(max_new_tokens=max_new_tokens, **kw))
    app = create_app(engine=engine, model_id=MODEL_ID)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client, engine


async def _wait_until(predicate, timeout: float = 3.0, interval: float = 0.01) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval)
    return predicate()


def _expected(start: int, max_tokens: int = MAX_TOKENS) -> str:
    return "".join(str(t) for t in simulate(start, max_tokens))


# --------------------------------------------------------------------------- #
# 运维端点
# --------------------------------------------------------------------------- #


async def test_health_reports_device_and_stats() -> None:
    async with _serve() as (client, engine):
        response = await client.get("/health")
        assert response.status_code == 200
        body = response.json()
        # 设备与 dtype 来自 EngineConfig（补充条款 A1/A2），这里断言它们被如实上报
        assert body["device"] == "cpu"
        assert body["dtype"] == "float32"
        assert body["model"] == MODEL_ID
        assert body["engine_loop_running"] is True
        assert body["kv_blocks_total"] == engine.core.runner.paged.num_blocks_total


async def test_models_endpoint() -> None:
    async with _serve() as (client, _):
        response = await client.get("/v1/models")
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "list"
        assert body["data"][0]["id"] == MODEL_ID
        assert body["data"][0]["owned_by"] == "liteinfer"


# --------------------------------------------------------------------------- #
# /v1/completions
# --------------------------------------------------------------------------- #


async def test_completion_non_stream() -> None:
    async with _serve() as (client, _):
        response = await client.post(
            "/v1/completions",
            json={"prompt": "2", "max_tokens": MAX_TOKENS, "temperature": 0.0},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["model"] == MODEL_ID
    assert body["choices"][0]["text"] == _expected(2)
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": MAX_TOKENS,
        "total_tokens": 1 + MAX_TOKENS,
    }


async def test_completion_accepts_prompt_list() -> None:
    """OpenAI 允许 prompt 是数组，按多个 choice 返回。"""
    async with _serve() as (client, _):
        response = await client.post(
            "/v1/completions",
            json={"prompt": ["2", "5"], "max_tokens": MAX_TOKENS, "temperature": 0.0},
        )
    assert response.status_code == 200
    body = response.json()
    assert [c["text"] for c in body["choices"]] == [_expected(2), _expected(5)]
    assert [c["index"] for c in body["choices"]] == [0, 1]
    assert body["usage"]["total_tokens"] == 2 * (1 + MAX_TOKENS)


async def test_completion_stream_sse() -> None:
    async with _serve() as (client, _):
        response = await client.post(
            "/v1/completions",
            json={
                "prompt": "2",
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        raw = response.text

    frames = parse_sse_frames(raw)
    # MAX_TOKENS 个 token 帧 + 1 个只带 finish_reason 的收尾帧（OpenAI 协议）
    assert len(frames) == MAX_TOKENS + 1
    assert [f["choices"][0]["text"] for f in frames] == ["3", "4", "5", "6", ""]
    assert "".join(f["choices"][0]["text"] for f in frames) == _expected(2)
    assert frames[-1]["choices"][0]["finish_reason"] == "length"
    assert all(f["choices"][0]["finish_reason"] is None for f in frames[:-1])
    assert raw.rstrip().endswith("[DONE]")


async def test_stream_rejects_prompt_list() -> None:
    async with _serve() as (client, _):
        response = await client.post(
            "/v1/completions",
            json={"prompt": ["2", "5"], "stream": True, "max_tokens": 2},
        )
    assert response.status_code == 400


async def test_oversized_prompt_returns_400() -> None:
    """prompt 超过 token budget：必须在响应头发出前就拒绝（fail fast）。"""
    async with _serve(max_new_tokens=4, max_num_batched_tokens=2) as (client, _):
        response = await client.post(
            "/v1/completions", json={"prompt": "12345", "max_tokens": 4}
        )
    assert response.status_code == 400
    assert "max_num_batched_tokens" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# /v1/chat/completions
# --------------------------------------------------------------------------- #


async def test_chat_completion_non_stream() -> None:
    async with _serve() as (client, _):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "2"}],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": _expected(2),
    }
    assert body["choices"][0]["finish_reason"] == "length"


async def test_chat_completion_stream_first_chunk_carries_role() -> None:
    async with _serve() as (client, _):
        response = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "2"}],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "stream": True,
            },
        )
        assert response.status_code == 200
        raw = response.text

    frames = parse_sse_frames(raw)
    assert frames[0]["object"] == "chat.completion.chunk"
    # OpenAI 协议要求第一个 chunk 声明角色，否则部分客户端渲染不出消息头
    assert frames[0]["choices"][0]["delta"]["role"] == "assistant"
    contents = [f["choices"][0]["delta"].get("content", "") for f in frames]
    assert "".join(contents) == _expected(2)
    assert contents[-1] == ""  # 收尾帧不带内容，只带 finish_reason
    assert frames[-1]["choices"][0]["finish_reason"] == "length"


async def test_chat_completion_rejects_empty_messages() -> None:
    async with _serve() as (client, _):
        response = await client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 400


def test_chat_prompt_uses_tokenizer_template() -> None:
    prompt = _render_chat_prompt(
        FakeTokenizer(), [ChatMessage(role="user", content="2")]
    )
    assert prompt == "2"


def test_chat_prompt_fallback_without_template() -> None:
    """tokenizer 没有 chat template 时不能让请求失败，降级成角色拼接。"""
    prompt = _render_chat_prompt(
        object(), [ChatMessage(role="user", content="hi")]
    )
    assert prompt == "user: hi\nassistant:"


# --------------------------------------------------------------------------- #
# 取消（docs/02 §10）
# --------------------------------------------------------------------------- #


async def test_cancel_endpoint_cancels_and_frees_blocks() -> None:
    async with _serve(max_new_tokens=64) as (client, engine):
        request_id = await engine.submit("2", SamplingParams(max_tokens=64, temperature=0.0))
        assert await _wait_until(
            lambda: engine.core.runner.paged.num_blocks_used > 0
        ), "生成过程中应真的占用物理块"

        response = await client.post(f"/v1/requests/{request_id}/cancel")

        assert response.status_code == 200
        assert response.json()["cancelled"] is True
        assert engine.core.get_request(request_id).status == RequestStatus.CANCELLED
        assert engine.core.runner.paged.num_blocks_used == 0


async def test_cancel_unknown_request_returns_404() -> None:
    async with _serve() as (client, _):
        response = await client.post("/v1/requests/does-not-exist/cancel")
    assert response.status_code == 404


async def test_closing_stream_generator_cancels_request() -> None:
    """确定性版本：直接关闭 SSE 生成器（等价于响应体被关闭），请求必须被取消。"""
    async with _serve(max_new_tokens=64, sleep_s=0.01) as (_client, engine):
        request_id = await engine.submit("2", SamplingParams(max_tokens=64, temperature=0.0))
        generator = _completion_sse(engine, request_id, MODEL_ID)
        first = await generator.__anext__()
        assert first.startswith(b"data: ")

        await generator.aclose()  # 模拟响应体被关闭

        assert await _wait_until(
            lambda: engine.core.get_request(request_id).status == RequestStatus.CANCELLED
        ), "关闭流之后请求没有被取消"
        assert engine.core.runner.paged.num_blocks_used == 0


# --------------------------------------------------------------------------- #
# 真实 socket 上的客户端断连（httpx ASGITransport 形不成断连，必须上真服务器）
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def real_server():
    """后台线程里跑一个真实的 uvicorn 服务。

    为什么非它不可：实测 httpx 的 ``ASGITransport`` 会把应用跑到结束再返回响应
    （第一条 SSE 帧到达时请求已经 finished），因此**形不成"中途断开"**。
    真实 socket 才有断连语义，而"客户端断连后 KV 正确回收"是 docs/07 的硬验收项。
    """
    import threading

    import uvicorn

    engine = AsyncEngine(make_core(max_new_tokens=64, sleep_s=0.01))
    app = create_app(engine=engine, model_id="fake-lm")
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("uvicorn 未能启动")
    try:
        yield engine, f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


async def test_client_disconnect_frees_blocks(real_server) -> None:
    """硬验收：OpenAI 客户端中途断开，服务端必须取消请求并归还 KV 块。"""
    engine, base_url = real_server
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        async with client.stream(
            "POST",
            "/v1/completions",
            json={
                "prompt": "2",
                "max_tokens": 64,
                "temperature": 0.0,
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            lines = response.aiter_lines()
            first = await lines.__anext__()
            assert first.startswith("data: ")
            # 断开之前先确认生成真的启动了：否则"块为 0"可能只是还没分配，断言会假绿
            assert await _wait_until(
                lambda: engine.core.runner.paged.num_blocks_used > 0
            ), "生成尚未占用物理块就断开，本用例失去意义"
        # 离开 with 即断开连接

    assert await _wait_until(
        lambda: engine.core.runner.paged.num_blocks_used == 0, timeout=5.0
    ), "客户端断连后物理块没有归还（KV 泄漏）"
    counts = engine.core.registry.count_by_status()
    assert counts[RequestStatus.CANCELLED] >= 1, f"断连后没有请求进入 CANCELLED：{counts}"
