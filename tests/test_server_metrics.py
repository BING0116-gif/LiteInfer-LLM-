"""Task 10 测试：服务层可观测端点（/metrics、trace、SSE include_usage）。

沿用 test_server_api 的骨架：httpx.AsyncClient + ASGITransport 同 loop 驱动，
模型用 ``_fakes.FakeLM``，全部为快测（不下载权重）。
"""

from __future__ import annotations

import contextlib
from typing import AsyncIterator

import httpx
import pytest

from _fakes import make_core
from liteinfer.engine import AsyncEngine
from liteinfer.server import create_app
from liteinfer.server.sse import parse_sse_frames

MAX_TOKENS = 4
MODEL_ID = "fake-lm"


@contextlib.asynccontextmanager
async def _serve(**kw) -> AsyncIterator[tuple]:
    engine = AsyncEngine(make_core(max_new_tokens=MAX_TOKENS, **kw))
    app = create_app(engine=engine, model_id=MODEL_ID)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client, engine


async def _complete(client: httpx.AsyncClient, prompt: str = "2") -> dict:
    resp = await client.post(
        "/v1/completions",
        json={"prompt": prompt, "max_tokens": MAX_TOKENS, "temperature": 0.0},
    )
    assert resp.status_code == 200
    return resp.json()


# --------------------------------------------------------------------------- #
# /metrics
# --------------------------------------------------------------------------- #


async def test_metrics_endpoint_after_completion() -> None:
    async with _serve() as (client, engine):
        await _complete(client)
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["requests_total"] == 1
        assert body["requests_finished"] == 1
        assert body["output_tokens_total"] == MAX_TOKENS
        # 块全部归还：utilization 为 0.0 而不是 None（None 只在未知块池时出现）
        assert body["kv_blocks_used"] == 0
        assert body["kv_utilization"] == 0.0
        # num_running 不断言：调度器 running 集合惰性清理（Task 06 既有行为）
        assert body["num_waiting"] == 0
        # 补充条款 A3：CPU 下显存指标必须是 None（展示层渲染 N/A），绝不填 0
        assert body["gpu_memory_mb"] is None
        assert body["gpu_memory_mb_display"] == "N/A (no GPU)"
        # TTFT/TPOT 均值来自真实请求
        assert body["ttft_s_mean"] is not None and body["ttft_s_mean"] > 0
        assert body["tpot_s_mean"] is not None


async def test_metrics_endpoint_before_any_request() -> None:
    async with _serve() as (client, _):
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert body["requests_total"] == 0
        assert body["ttft_s_mean"] is None  # 没有请求时均值是 None，不是 0
        assert body["output_tokens_total"] == 0


async def test_health_includes_kv_utilization() -> None:
    async with _serve() as (client, engine):
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["kv_utilization"] == 0.0
        assert body["kv_blocks_total"] == engine.core.runner.paged.num_blocks_total


# --------------------------------------------------------------------------- #
# /v1/requests/{id}/trace
# --------------------------------------------------------------------------- #


async def test_trace_endpoint_full_timeline() -> None:
    async with _serve() as (client, engine):
        await _complete(client)
        rid = next(iter(engine.core.registry.all())).request_id
        resp = await client.get(f"/v1/requests/{rid}/trace")
        assert resp.status_code == 200
        body = resp.json()
        names = [e["name"] for e in body["events"]]
        assert names[0] == "enqueue" and names[-1] == "finished"
        assert names.count("token") == MAX_TOKENS
        offsets = [e["offset_s"] for e in body["events"]]
        assert offsets == sorted(offsets)
        assert offsets[0] == pytest.approx(0.0)
        assert body["ttft_s"] > 0
        assert body["output_tokens"] == MAX_TOKENS


async def test_trace_endpoint_unknown_request_404() -> None:
    async with _serve() as (client, _):
        resp = await client.get("/v1/requests/does-not-exist/trace")
    assert resp.status_code == 404


async def test_trace_available_during_generation() -> None:
    """非终态请求也可查 trace（时间线截至当前）——观测不该要求请求先结束。"""
    async with _serve(sleep_s=0.02) as (client, engine):
        from liteinfer.sampling.params import SamplingParams

        rid = await engine.submit("2", SamplingParams(max_tokens=64, temperature=0.0))
        # 不消费流，等引擎自己推进几步后查 trace
        import asyncio

        await asyncio.sleep(0.15)
        resp = await client.get(f"/v1/requests/{rid}/trace")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("decode", "prefill", "finished")
        assert any(e["name"] == "enqueue" for e in body["events"])
        # 清理：取消以免流悬挂
        await client.post(f"/v1/requests/{rid}/cancel")


# --------------------------------------------------------------------------- #
# SSE stream_options.include_usage（Task 09 遗留：流式不带 usage）
# --------------------------------------------------------------------------- #


async def test_completion_stream_include_usage() -> None:
    async with _serve() as (client, _):
        resp = await client.post(
            "/v1/completions",
            json={
                "prompt": "2",
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert resp.status_code == 200
        raw = resp.text

    frames = parse_sse_frames(raw)
    # MAX_TOKENS 个 token 帧 + finish 帧 + 1 个 usage 帧
    assert len(frames) == MAX_TOKENS + 2
    usage_frame = frames[-1]
    assert usage_frame["choices"] == [], "usage 帧的 choices 必须是空列表"
    assert usage_frame["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": MAX_TOKENS,
        "total_tokens": 1 + MAX_TOKENS,
    }
    # usage 帧之前不出现 usage
    assert all(f["usage"] is None for f in frames[:-1])
    assert raw.rstrip().endswith("[DONE]")


async def test_completion_stream_without_usage_by_default() -> None:
    async with _serve() as (client, _):
        resp = await client.post(
            "/v1/completions",
            json={
                "prompt": "2",
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "stream": True,
            },
        )
        raw = resp.text
    frames = parse_sse_frames(raw)
    assert len(frames) == MAX_TOKENS + 1  # 无 usage 帧
    assert all(f["usage"] is None for f in frames)


async def test_chat_stream_include_usage() -> None:
    async with _serve() as (client, _):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "2"}],
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert resp.status_code == 200
        raw = resp.text
    frames = parse_sse_frames(raw)
    usage_frame = frames[-1]
    assert usage_frame["choices"] == []
    assert usage_frame["usage"]["completion_tokens"] == MAX_TOKENS
    # role 帧 + token 帧 + finish 帧 + usage 帧
    assert usage_frame["object"] == "chat.completion.chunk"
