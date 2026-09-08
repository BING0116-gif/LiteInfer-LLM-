"""Decoder 层：Qwen2 的 pre-norm 残差块。"""

from __future__ import annotations

import torch
from torch import nn

from liteinfer.model.minimal.attention import QwenSelfAttention
from liteinfer.model.minimal.mlp import QwenMLP
from liteinfer.model.minimal.rmsnorm import RMSNorm


class QwenDecoderLayer(nn.Module):
    """``x + attn(norm1(x))`` 再 ``x' + mlp(norm2(x'))``。

    pre-norm（norm 在子层之前）而不是 post-norm：残差主干上没有非线性
    变换，深层网络的梯度可以无损穿过；这是 Qwen/Llama/GPT 系的标准结构，
    也是推理时"hidden_states 进层、hidden_states 出层"这一接口能逐层
    与 HF 中间量对齐的前提。
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        intermediate_size: int,
        rope_theta: float,
        rms_norm_eps: float,
        attention_bias: bool = True,
        use_qk_norm: bool = False,
        mlp_bias: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = QwenSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            use_qk_norm=use_qk_norm,
        )
        self.mlp = QwenMLP(hidden_size, intermediate_size, bias=mlp_bias)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_ids, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states
