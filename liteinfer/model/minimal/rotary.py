"""RoPE 旋转位置编码（Qwen2 风格：rotate_half 实现，theta=1e6）。"""

from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """按 position 现算 cos/sin，不做缓存。

    本阶段刻意不缓存 cos/sin 表（每次 forward 重算）：Task 03 的目标是
    数值对齐，现算路径最短、最容易与 HF 逐位对照。cos/sin 预计算 + 增量
    追加是 KV Cache 阶段（Task 04）才需要的优化，提前做只会增加对齐噪声。
    """

    def __init__(self, head_dim: int, base: float = 1000000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim 必须是偶数才能做两两旋转，收到 {head_dim}")
        self.head_dim = head_dim
        # inv_freq 只与维度有关、与位置无关，是常量；persistent=False
        # 使其不进 state_dict（HF 同样如此，checkpoint 里没有这个张量）
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """根据位置id计算 cos/sin 表。

        Args:
            position_ids: ``[B, S]`` 任意 dtype 的位置索引。

        Returns:
            (cos, sin)，各为 ``[B, S, head_dim]``。head 维由
            ``apply_rotary_pos_emb`` 统一 unsqueeze 出来广播，
            这里不提前加（HF 的布局：rotary 只管位置，不管 head）。
        """
        # 位置必须升到 float32 参与乘法：长序列下 int32 与大 base 的
        # inv_freq 相乘会溢出；fp32 精度到 2^24 都够用
        pos = position_ids.to(torch.float32)
        # [B, S, 1] x [1, head_dim/2] -> [B, S, head_dim/2]
        freqs = pos[:, :, None] * self.inv_freq[None, None, :]
        # cat(freqs, freqs) 把"两两一组"的频率摊平成 head_dim 维，
        # 与 rotate_half 的前后半拆分方式一一对应
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把最后一维拆成前后两半，返回 ``[-x2, x1]``。

    这是 HF 的 rotate_half 写法（配合 cat(freqs, freqs) 的 cos/sin 展开），
    与"复数乘法"的写法数学等价但张量布局不同——对齐时必须选边站，
    这里选 HF 的布局。
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 ``[B, H, S, D]`` 的 q/k 施加旋转。

    cos/sin 来自 ``[B, S, 1, D]``，unsqueeze(1) 后变成 ``[B, 1, S, D]``
    向所有 head 广播——K/V 即使是 GQA 的少量 head，旋转角也只由位置决定，
    与 head 无关。
    """
    cos = cos.unsqueeze(1)  # [B, 1, S, D]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
