"""采样参数定义（Task 02）。

fail fast 原则（与 config.parse_dtype 同理）：非法参数如果在生成循环深处
才暴露（比如 temperature 传负数导致 softmax 产生 NaN），排查成本远高于
构造时直接抛错。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingParams:
    """一次生成的采样配置。

    约定（与 vLLM / OpenAI API 的通行语义对齐）：
    - ``temperature == 0.0`` 即 greedy，不进入随机采样路径；
    - ``top_k == -1`` 表示关闭 top-k；``top_p == 1.0`` 表示关闭 top-p；
    - ``seed`` 为 None 时不注入随机源（每次结果都不同），给定则可复现。

    frozen：参数在一次请求生命周期内不应被中途篡改，否则复现性无从谈起。
    """

    max_tokens: int
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens 必须 >= 1，收到 {self.max_tokens}")
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature 必须 >= 0.0（0.0 表示 greedy），收到 {self.temperature}"
            )
        if self.top_k != -1 and self.top_k < 1:
            raise ValueError(
                f"top_k 必须为 -1（关闭）或 >= 1，收到 {self.top_k}"
            )
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(
                f"top_p 必须在 (0.0, 1.0] 区间，收到 {self.top_p}"
            )

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0
