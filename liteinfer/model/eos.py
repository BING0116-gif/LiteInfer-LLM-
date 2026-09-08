"""EOS 判定：全项目唯一真相。

Task 02 踩过的坑（见 PROGRESS.md）：Qwen2.5 的生成终止符是
``<|im_end|>``（151645），记录在 ``generation_config.eos_token_id`` 而不是
``tokenizer.eos_token_id``（后者是 ``<|endoftext|>`` = 151643）。
两条生成链（Task 02 的 ManualGenerator、Task 04 的 CachedGenerator）如果
各自解析一遍，只要有一边选错，benchmark 的"输出是否一致"断言就会以
"长度不同"的形式失败，看起来像数值问题，实际是 EOS 口径问题。
"""

from __future__ import annotations

from typing import Any


def resolve_eos_ids(model: Any, tokenizer: Any) -> "frozenset[int]":
    """解析终止 token 集合，优先 ``generation_config``，回退 tokenizer。

    取值的优先级与 HF 对齐：``generate()`` 用的是
    ``generation_config.eos_token_id``，它可能是 int 也可能是 list
    （多终止符），两种形态都归一成 ``frozenset[int]``。

    MinimalQwen 这类自建模块没有 ``generation_config``，会回退到
    tokenizer；此时建议调用方显式传入从 HF 模型解析出的集合，
    保证有缓存/无缓存两条链用同一套终止符。
    """
    raw = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if raw is None:
        raw = getattr(tokenizer, "eos_token_id", None)
    if raw is None:
        return frozenset()
    if isinstance(raw, int):
        return frozenset({raw})
    return frozenset(int(x) for x in raw)
