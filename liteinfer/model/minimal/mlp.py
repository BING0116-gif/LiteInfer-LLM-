"""SwiGLU MLP：``down( silu(gate(x)) * up(x) )``。

Qwen 的中间维度是 config.intermediate_size（0.5B 为 4864），
不是 Llama 那类 4*huge 再取整的公式，因此维度一律从 config 传入，
不在此处推算。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class QwenMLP(nn.Module):
    """三层线性投影的 SwiGLU 前馈网络。

    Qwen2 的 MLP 默认无 bias（config.mlp_bias=False），但投影维度/是否有
    bias 都由构造参数决定，不写死——checkpoint 里有什么就建什么，
    ``load_state_dict(strict=True)`` 才能对得上。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU 作用在 gate 分支上；up 分支不激活，两支逐元素相乘
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
