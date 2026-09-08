"""OpenAI 兼容的请求 / 响应模型（pydantic v2）。

只定义 LiteInfer 真正实现的那部分字段：多模态、logprobs、n>1、function calling 等
一概不声明——声明了却不支持，比直接忽略更糟（客户端会以为拿到的是真实语义）。
未声明的字段由 pydantic 默认忽略，不会报错，因此 OpenAI SDK 带的额外参数不会打挂服务。
"""

from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """聊天消息。OpenAI 的 ``content`` 可以是多模态数组，本引擎只支持纯文本。"""

    role: str = "user"
    content: str = ""


class _GenerationParams(BaseModel):
    """completions 与 chat/completions 共用的采样参数。

    默认值与 OpenAI 对齐：``temperature=1.0``、``top_p=1.0``。
    ``top_k`` 不是 OpenAI 官方字段，但 vLLM 等引擎都支持，这里一并暴露。
    """

    model: Optional[str] = None
    max_tokens: Optional[int] = None  # None -> 用 EngineConfig.max_new_tokens
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 表示关闭 top-k
    seed: Optional[int] = None
    stream: bool = False


class CompletionRequest(_GenerationParams):
    """``POST /v1/completions`` 请求体。"""

    # OpenAI 允许字符串或字符串数组；数组按"多个 choice"处理（仅非流式支持）
    prompt: Union[str, list[str]] = ""


class ChatCompletionRequest(_GenerationParams):
    """``POST /v1/chat/completions`` 请求体。"""

    messages: list[ChatMessage] = Field(default_factory=list)


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    """非流式 ``/v1/completions`` 响应；流式时每个 SSE 帧也是这个形状（text 为增量）。"""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Optional[Usage] = None  # 流式帧里不带 usage


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Optional[Usage] = None


class DeltaMessage(BaseModel):
    """流式 chat chunk 的增量内容。首个 chunk 带 ``role="assistant"``，其后只带 content。"""

    role: Optional[str] = None
    content: str = ""


class ChatCompletionStreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice]


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "liteinfer"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class ErrorPayload(BaseModel):
    """OpenAI 风格的错误体，便于 OpenAI SDK 抛出可读异常。"""

    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorPayload
