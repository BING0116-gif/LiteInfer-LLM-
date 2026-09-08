"""SSE（Server-Sent Events）帧编码。

OpenAI 的流式协议就是 SSE：每帧 ``data: <json>\n\n``，结束帧固定为 ``data: [DONE]\n\n``。
单独抽一个模块的原因：帧格式是"线上协议"，和路由逻辑混在一起会让人分不清
"这是 HTTP 的事"还是"这是生成的事"。
"""

from __future__ import annotations

import json
from typing import Any, Mapping

#: 流式结束标志（OpenAI 客户端靠它收尾，缺了客户端会一直等）
SSE_DONE = b"data: [DONE]\n\n"

#: 关闭一切中间层缓冲：nginx / 浏览器缓冲会把"逐 token 到达"变成"一次性到达"
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def sse_frame(payload: Mapping[str, Any] | str) -> bytes:
    """把一个 dict（或裸字符串）编码成 SSE 帧。

    ``ensure_ascii=False``：中文不用 ``\\uXXXX`` 转义，便于直接观察与调试，
    同时字节数也更小。
    """
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return b"data: " + data.encode("utf-8") + b"\n\n"


def parse_sse_frames(raw: str) -> list[dict]:
    """测试/调试用：把一段 SSE 文本还原成 payload 列表（``[DONE]`` 之前的部分）。

    放在这里而不是测试里，是因为它是"协议解析"，与协议编码应当同居一处。
    """
    frames: list[dict] = []
    for block in raw.split("\n\n"):
        line = block.strip()
        if not line or not line.startswith("data:"):
            continue
        body = line[len("data:") :].strip()
        if body == "[DONE]":
            break
        frames.append(json.loads(body))
    return frames
