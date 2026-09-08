"""RMSNorm：Qwen2 的归一化算子（无 bias、不减均值，只除 RMS）。"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """``weight * x / sqrt(mean(x^2) + eps)``。

    计算路径刻意镜像 HF ``Qwen2RMSNorm``：先升到 float32 求方差与归一化，
    再 cast 回输入 dtype，最后才乘 weight。为什么坚持升精度：FP16 下
    ``x^2`` 的累加容易溢出/丢精度，这一步升浮点是 HF 的既定行为，
    对齐测试要求两边的数值路径一致，而不是"数学上等价"。
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # rsqrt(1/sqrt) 比先 sqrt 再除少一次舍入
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = self._norm(x)
        # weight 在 cast 回原 dtype 之后乘，与 HF 实现顺序一致
        return self.weight * x.to(input_dtype)

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"
