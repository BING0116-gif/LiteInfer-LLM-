"""Task 09：OpenAI 兼容 HTTP 服务层（FastAPI + SSE 流式 + 取消）。

分层（docs/02 §1）：本模块只做"协议翻译"，不含任何推理逻辑：

    HTTP / OpenAI SDK ──> 参数校验 ──> AsyncEngine.submit ──> AsyncEngine.stream ──> SSE

``create_app(engine=...)`` 支持注入引擎：测试可以塞一个跑假模型的 AsyncEngine，
``import liteinfer.server.app`` 因此不会触发模型下载——这条约束和 Task 01 以来
"快速测试不依赖模型"的约定一致。
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from liteinfer import __version__
from liteinfer.config import EngineConfig
from liteinfer.engine.async_engine import AsyncEngine, StreamChunk
from liteinfer.sampling.params import SamplingParams
from liteinfer.server.schemas import (
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    DeltaMessage,
    ModelCard,
    ModelList,
    Usage,
)
from liteinfer.server.sse import SSE_DONE, SSE_HEADERS, sse_frame

logger = logging.getLogger("liteinfer.server.app")


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #


def _now() -> int:
    return int(time.time())


def _new_id(prefix: str) -> str:
    """OpenAI 风格的对象 id（``cmpl-`` / ``chatcmpl-`` 前缀 + 随机串）。"""
    return f"{prefix}-{uuid.uuid4().hex[:24]}"


def _engine_of(request: Request) -> AsyncEngine:
    return request.app.state.engine  # type: ignore[no-any-return]


def _default_max_tokens(engine: AsyncEngine) -> int:
    cfg = getattr(engine.core, "cfg", None)
    return int(getattr(cfg, "max_new_tokens", 64))


def _build_params(body: Any, engine: AsyncEngine) -> SamplingParams:
    """把 OpenAI 请求体翻成 ``SamplingParams``。

    ``max_tokens`` 缺省时回落到引擎配置，而不是某个写死的常数——配置集中管理
    （docs/07 §五），服务层不该有第二个默认值来源。
    """
    max_tokens = (
        body.max_tokens if body.max_tokens is not None else _default_max_tokens(engine)
    )
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=body.temperature,
        top_k=body.top_k,
        top_p=body.top_p,
        seed=body.seed,
    )


def _core_stats(engine: AsyncEngine) -> dict[str, Any]:
    """给 /health 用的观测项。全部用 getattr 兜底：注入的假引擎可能没有这些字段。"""
    core = engine.core
    stats: dict[str, Any] = {
        "waiting": None,
        "running": None,
        "kv_blocks_used": None,
        "kv_blocks_total": None,
    }
    scheduler = getattr(core, "scheduler", None)
    if scheduler is not None:
        stats["waiting"] = scheduler.num_waiting
        stats["running"] = scheduler.num_running
    paged = getattr(getattr(core, "runner", None), "paged", None)
    if paged is not None:
        stats["kv_blocks_used"] = paged.num_blocks_used
        stats["kv_blocks_total"] = paged.num_blocks_total
    return stats


def _render_chat_prompt(tokenizer: Any, messages: list[ChatMessage]) -> str:
    """把 messages 渲染成引擎能吃的纯文本 prompt。

    优先用 tokenizer 自带的 chat template（Qwen2.5 有，且这才是它训练时的格式，
    直接拼 "user: xxx" 会明显掉质量）；拿不到模板时降级为简单的角色拼接，
    保证没有模板的 tokenizer（含测试里的假 tokenizer）也能跑通链路。
    """
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            return str(
                apply_chat_template(
                    [{"role": m.role, "content": m.content} for m in messages],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception:  # 无模板 / 模板不兼容：降级，不让整个请求失败
            logger.warning("tokenizer 无可用 chat template，降级为角色拼接")
    return "".join(f"{m.role}: {m.content}\n" for m in messages) + "assistant:"


# --------------------------------------------------------------------------- #
# 非流式：跑完整个请求
# --------------------------------------------------------------------------- #


async def _generate_full(
    engine: AsyncEngine, prompt: str, params: SamplingParams
) -> tuple[str, int, int, str]:
    """跑完一个请求，返回 ``(文本, prompt_tokens, completion_tokens, finish_reason)``。

    显式拿迭代器再 ``aclose()``（而不是直接 ``async for``）：保证任何异常路径下
    内层流都被确定性地关闭，从而触发 AsyncStream 的清理与块回收。
    """
    request_id = await engine.submit(prompt, params)
    parts: list[str] = []
    finish_reason: Optional[str] = None
    iterator = engine.stream(request_id).__aiter__()
    try:
        while True:
            chunk: StreamChunk = await iterator.__anext__()
            if chunk.text:
                parts.append(chunk.text)
            if chunk.finished:
                finish_reason = chunk.finish_reason
                break
    finally:
        await iterator.aclose()

    request = engine.core.get_request(request_id)
    return "".join(parts), request.prompt_tokens, request.output_tokens, (
        finish_reason or "length"
    )


# --------------------------------------------------------------------------- #
# 流式：SSE
# --------------------------------------------------------------------------- #


async def _completion_sse(
    engine: AsyncEngine, request_id: str, model_id: str
) -> AsyncIterator[bytes]:
    """``/v1/completions`` 的 SSE 体。

    为什么手写 ``__anext__`` 循环 + ``finally: await it.aclose()``，
    而不是 ``async for chunk in engine.stream(rid)``：客户端断连时，
    外层生成器被关闭，若内层只靠 GC 兜底，其 ``finally`` 的执行时机不确定，
    "断连是否回收 KV"就不可验证。显式 aclose 让清理时机确定。
    """
    created = _now()
    response_id = _new_id("cmpl")
    def _frame(text: str, finish: Optional[str]) -> bytes:
        payload = CompletionResponse(
            id=response_id,
            created=created,
            model=model_id,
            choices=[CompletionChoice(index=0, text=text, finish_reason=finish)],
        )
        return sse_frame(payload.model_dump())

    iterator = engine.stream(request_id).__aiter__()
    try:
        while True:
            chunk = await iterator.__anext__()
            if chunk.finished and chunk.token_id is None:
                # 纯终态 chunk：EOS 命中的那一步没有产出新 token，它只负责送 finish_reason
                yield _frame("", chunk.finish_reason)
                break
            yield _frame(chunk.text, None)
            if chunk.finished:
                # 引擎把"最后一个 token"和"结束标志"合并在同一个 chunk 里，而 OpenAI
                # 协议要求先送完 token、再补一个只带 finish_reason 的空帧。
                # 早期版本直接把 finished chunk 的 text 清空，结果最后一个 token 被吞掉。
                yield _frame("", chunk.finish_reason)
                break
        yield SSE_DONE
    finally:
        await iterator.aclose()


async def _chat_sse(
    engine: AsyncEngine, request_id: str, model_id: str
) -> AsyncIterator[bytes]:
    """``/v1/chat/completions`` 的 SSE 体（首个帧带 ``role=assistant``）。"""
    created = _now()
    response_id = _new_id("chatcmpl")

    def _chunk(role: Optional[str], content: str, finish: Optional[str]) -> bytes:
        payload = ChatCompletionChunk(
            id=response_id,
            created=created,
            model=model_id,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=DeltaMessage(role=role, content=content),
                    finish_reason=finish,
                )
            ],
        )
        return sse_frame(payload.model_dump())

    iterator = engine.stream(request_id).__aiter__()
    try:
        yield _chunk("assistant", "", None)
        while True:
            chunk = await iterator.__anext__()
            if chunk.finished and chunk.token_id is None:
                yield _chunk(None, "", chunk.finish_reason)
                break
            yield _chunk(None, chunk.text, None)
            if chunk.finished:
                # 与 /v1/completions 同理：token 送完之后再补一个只带 finish_reason 的空帧
                yield _chunk(None, "", chunk.finish_reason)
                break
        yield SSE_DONE
    finally:
        await iterator.aclose()


# --------------------------------------------------------------------------- #
# 应用工厂
# --------------------------------------------------------------------------- #


def create_app(
    engine: Optional[AsyncEngine] = None,
    cfg: Optional[EngineConfig] = None,
    model_id: Optional[str] = None,
) -> FastAPI:
    """构造 FastAPI 应用。

    - 传入 ``engine``：直接用它（测试 / 复用已有引擎的场景），lifespan 只负责起停循环；
    - 不传：在 lifespan 启动时按 ``cfg`` 加载模型并建引擎，保证 **import 不加载模型**。
    """
    if engine is None and cfg is None:
        cfg = EngineConfig()
    resolved_model_id = model_id or (cfg.model_id if cfg is not None else "liteinfer")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active = app.state.engine
        if active is None:
            logger.info("加载模型 %s（device=%s）", resolved_model_id, cfg.device)
            active = AsyncEngine(_make_core(cfg))  # type: ignore[arg-type]
            app.state.engine = active
            app.state.owns_engine = True
        await active.start()
        try:
            yield
        finally:
            await active.shutdown()

    app = FastAPI(title="LiteInfer", version=__version__, lifespan=lifespan)
    app.state.engine = engine
    app.state.cfg = cfg
    app.state.model_id = resolved_model_id
    app.state.owns_engine = False

    # ---- 运维端点 ----

    @app.get("/health")
    async def health(request: Request) -> dict:
        active = _engine_of(request)
        return {
            "status": "ok",
            "model": request.app.state.model_id,
            "device": str(getattr(getattr(active.core, "cfg", None), "device", "unknown")),
            "dtype": str(getattr(getattr(active.core, "cfg", None), "dtype", "unknown")).replace(
                "torch.", ""
            ),
            "engine_loop_running": active.is_running(),
            **_core_stats(active),
        }

    @app.get("/v1/models")
    async def list_models(request: Request) -> ModelList:
        return ModelList(
            data=[ModelCard(id=request.app.state.model_id, created=_now())]
        )

    # ---- 生成端点 ----

    @app.post("/v1/completions", response_model=None)
    async def create_completion(body: CompletionRequest, request: Request) -> Any:
        active = _engine_of(request)
        prompts = body.prompt if isinstance(body.prompt, list) else [body.prompt]
        try:
            params = _build_params(body, active)
        except ValueError as exc:  # SamplingParams 的构造期校验
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if body.stream:
            if len(prompts) != 1:
                raise HTTPException(
                    status_code=400,
                    detail="stream=True 时 prompt 必须是单个字符串",
                )
            try:
                # 先提交再返回响应：校验错误（如 prompt 超长）要在响应头发出之前抛出，
                # 否则只能干瞪眼看着 200 已经发出去
                request_id = await active.submit(prompts[0], params)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return StreamingResponse(
                _completion_sse(active, request_id, request.app.state.model_id),
                media_type="text/event-stream",
                headers=SSE_HEADERS,
            )

        choices: list[CompletionChoice] = []
        usage = Usage()
        for index, prompt in enumerate(prompts):
            try:
                text, prompt_tokens, completion_tokens, finish_reason = (
                    await _generate_full(active, prompt, params)
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            choices.append(
                CompletionChoice(index=index, text=text, finish_reason=finish_reason)
            )
            usage.prompt_tokens += prompt_tokens
            usage.completion_tokens += completion_tokens
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        return CompletionResponse(
            id=_new_id("cmpl"),
            created=_now(),
            model=request.app.state.model_id,
            choices=choices,
            usage=usage,
        )

    @app.post("/v1/chat/completions", response_model=None)
    async def create_chat_completion(
        body: ChatCompletionRequest, request: Request
    ) -> Any:
        active = _engine_of(request)
        if not body.messages:
            raise HTTPException(status_code=400, detail="messages 不能为空")
        prompt = _render_chat_prompt(getattr(active.core, "tokenizer", None), body.messages)
        try:
            params = _build_params(body, active)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if body.stream:
            try:
                request_id = await active.submit(prompt, params)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return StreamingResponse(
                _chat_sse(active, request_id, request.app.state.model_id),
                media_type="text/event-stream",
                headers=SSE_HEADERS,
            )

        try:
            text, prompt_tokens, completion_tokens, finish_reason = (
                await _generate_full(active, prompt, params)
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return ChatCompletionResponse(
            id=_new_id("chatcmpl"),
            created=_now(),
            model=request.app.state.model_id,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason=finish_reason,
                )
            ],
            usage=usage,
        )

    # ---- 取消端点 ----

    @app.post("/v1/requests/{request_id}/cancel", response_model=None)
    async def cancel_request(request_id: str, request: Request) -> dict:
        """显式取消（docs/02 §10）。另一条取消路径是客户端断连，由 AsyncStream 自动兜底。"""
        active = _engine_of(request)
        registry = getattr(active.core, "registry", None)
        if registry is None or request_id not in registry:
            raise HTTPException(status_code=404, detail=f"未知 request_id: {request_id}")
        await active.abort(request_id)
        return {"id": request_id, "object": "request.cancelled", "cancelled": True}

    return app


def _make_core(cfg: Optional[EngineConfig]) -> Any:
    """按配置加载模型并构造同步引擎（延迟到 lifespan 内执行）。"""
    from liteinfer.engine.core import EngineCore  # 局部导入：避免 import 期就拉起 torch 模型栈

    if cfg is None:
        cfg = EngineConfig()
    return EngineCore.from_config(cfg)


__all__ = ["create_app"]
