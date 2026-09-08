"""Task 03 算子级单测：不依赖真实模型，纯张量手算验证。

每个算子先独立验证"数学上正确"，再进对齐测试验证"与 HF 一致"——
两层防线，算子错了对齐测试只会给出难排查的层间误差。
"""

from __future__ import annotations

import math

import pytest
import torch

from liteinfer.model.minimal.attention import QwenSelfAttention, repeat_kv
from liteinfer.model.minimal.mlp import QwenMLP
from liteinfer.model.minimal.model import build_causal_mask
from liteinfer.model.minimal.rotary import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)
from liteinfer.model.minimal.rmsnorm import RMSNorm

torch.manual_seed(0)


class TestRMSNorm:
    def test_matches_manual_formula(self) -> None:
        x = torch.randn(2, 5, 16)
        norm = RMSNorm(16, eps=1e-6)
        with torch.no_grad():
            norm.weight.normal_(1.0, 0.1)
        # 与公式直接对照：x / rms(x) * w
        expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.weight
        torch.testing.assert_close(norm(x), expected, atol=1e-6, rtol=1e-6)

    def test_zero_input_no_nan(self) -> None:
        # 全零输入靠 eps 兜底：rms=0 时除零会产生 NaN，eps 的存在意义
        norm = RMSNorm(8)
        out = norm(torch.zeros(1, 4, 8))
        assert torch.isfinite(out).all()

    def test_norm_changes_scale_not_direction(self) -> None:
        # RMSNorm 不减均值，只缩放模长：归一化后与原向量的夹角余弦保持
        x = torch.randn(1, 3, 8) + 5.0  # 均值非零，区别于 LayerNorm
        out = RMSNorm(8)(x)
        cos = torch.nn.functional.cosine_similarity(x, out, dim=-1)
        assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)


class TestRotary:
    def test_inv_freq_matches_formula(self) -> None:
        rotary = RotaryEmbedding(head_dim=8, base=10000.0)
        expected = 1.0 / (10000.0 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
        torch.testing.assert_close(rotary.inv_freq, expected)

    def test_position_zero_is_identity(self) -> None:
        # 位置 0 的 cos=1/sin=0：旋转退化为恒等映射
        rotary = RotaryEmbedding(8)
        q = torch.randn(1, 2, 1, 8)
        k = torch.randn(1, 2, 1, 8)
        cos, sin = rotary(torch.zeros(1, 1, dtype=torch.long))
        q2, k2 = apply_rotary_pos_emb(q, k, cos, sin)
        torch.testing.assert_close(q2, q, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(k2, k, atol=1e-6, rtol=1e-6)

    def test_rotation_preserves_norm(self) -> None:
        # 旋转矩阵是正交的：施加 RoPE 后每个位置的模长不变
        rotary = RotaryEmbedding(8)
        q = torch.randn(1, 4, 3, 8)
        cos, sin = rotary(torch.arange(3).unsqueeze(0))
        q2, _ = apply_rotary_pos_emb(q, q.clone(), cos, sin)
        torch.testing.assert_close(
            q2.norm(dim=-1), q.norm(dim=-1), atol=1e-5, rtol=1e-5
        )

    def test_cos_sin_values_hand_computed(self) -> None:
        # 手算一个已知值：head_dim=2, base=10000, pos=1
        # inv_freq=[1.0]，freqs=[1.0]，emb=[1,1]，cos=cos(1)≈0.5403
        rotary = RotaryEmbedding(2, base=10000.0)
        cos, sin = rotary(torch.ones(1, 1, dtype=torch.long))
        # cat(freqs, freqs) 展开后 head_dim=2 的两个分量相同，任取其一
        assert math.isclose(cos[0, 0, 0].item(), math.cos(1.0), rel_tol=1e-6)
        assert math.isclose(sin[0, 0, 0].item(), math.sin(1.0), rel_tol=1e-6)

    def test_rotate_half_layout(self) -> None:
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        # 前半 [1,2] 取负变后半，后半 [3,4] 变前半
        torch.testing.assert_close(
            rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        )


class TestGQA:
    def test_repeat_kv_grouping(self) -> None:
        # n_rep=2：KV head 0 -> Q head 0,1；KV head 1 -> Q head 2,3
        kv = torch.tensor([[[[1.0]], [[2.0]]]])  # [1, 2, 1, 1]
        out = repeat_kv(kv, 2)
        assert out.shape == (1, 4, 1, 1)
        assert out[0, :, 0, 0].tolist() == [1.0, 1.0, 2.0, 2.0]

    def test_repeat_kv_identity_when_n_rep_1(self) -> None:
        kv = torch.randn(1, 4, 2, 3)
        assert repeat_kv(kv, 1) is kv

    def test_causal_attention_locality(self) -> None:
        # 因果性：改变后面 token 的内容，前面 token 的输出必须不变
        attn = QwenSelfAttention(
            hidden_size=16, num_attention_heads=4, num_key_value_heads=2,
            head_dim=4, rope_theta=10000.0, rms_norm_eps=1e-6,
        ).eval()
        x1 = torch.randn(1, 4, 16)
        x2 = x1.clone()
        x2[0, 2:] = torch.randn(2, 16)  # 只改 token 2、3
        pos = torch.arange(4).unsqueeze(0)
        mask = build_causal_mask(4, x1.device, x1.dtype)
        with torch.inference_mode():
            o1 = attn(x1, pos, mask)
            o2 = attn(x2, pos, mask)
        torch.testing.assert_close(o1[0, :2], o2[0, :2], atol=1e-5, rtol=1e-5)
        assert not torch.allclose(o1[0, 2:], o2[0, 2:])

    def test_output_shape_and_gqa_param_dims(self) -> None:
        attn = QwenSelfAttention(
            hidden_size=16, num_attention_heads=4, num_key_value_heads=2,
            head_dim=4, rope_theta=10000.0, rms_norm_eps=1e-6,
        )
        assert attn.k_proj.weight.shape == (2 * 4, 16)
        assert attn.q_proj.weight.shape == (4 * 4, 16)
        assert attn.o_proj.weight.shape == (16, 4 * 4)
        x = torch.randn(2, 5, 16)
        pos = torch.arange(5).unsqueeze(0).expand(2, -1)
        mask = build_causal_mask(5, x.device, x.dtype)
        assert attn(x, pos, mask).shape == (2, 5, 16)

    def test_invalid_gqa_ratio_raises(self) -> None:
        with pytest.raises(ValueError, match="整除"):
            QwenSelfAttention(
                hidden_size=16, num_attention_heads=4, num_key_value_heads=3,
                head_dim=4, rope_theta=10000.0, rms_norm_eps=1e-6,
            )


class TestMLPAndMask:
    def test_swiglu_matches_manual(self) -> None:
        mlp = QwenMLP(hidden_size=8, intermediate_size=5).eval()
        x = torch.randn(1, 3, 8)
        with torch.inference_mode():
            expected = mlp.down_proj(
                torch.nn.functional.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
            )
        torch.testing.assert_close(mlp(x), expected)

    def test_causal_mask_structure(self) -> None:
        mask = build_causal_mask(4, torch.device("cpu"), torch.float32)
        assert mask.shape == (1, 1, 4, 4)
        lower = torch.tril(torch.ones(4, 4))
        # 下三角（含对角线）为 0：允许注意力；上三角为 finfo.min：屏蔽
        assert torch.all(mask[0, 0][lower.bool()] == 0)
        assert torch.all(mask[0, 0][~lower.bool()] == torch.finfo(torch.float32).min)

    def test_decoder_layer_residual_structure(self) -> None:
        # 把 o_proj / down_proj 权重清零 => attn/mlp 输出为 0，
        # 层输出应等于输入（残差主干直通），验证残差连接位置正确
        from liteinfer.model.minimal.layer import QwenDecoderLayer

        layer = QwenDecoderLayer(
            hidden_size=8, num_attention_heads=2, num_key_value_heads=1,
            head_dim=4, intermediate_size=6, rope_theta=10000.0, rms_norm_eps=1e-6,
        ).eval()
        with torch.inference_mode():
            layer.self_attn.o_proj.weight.zero_()
            layer.mlp.down_proj.weight.zero_()
        x = torch.randn(1, 3, 8)
        pos = torch.arange(3).unsqueeze(0)
        mask = build_causal_mask(3, x.device, x.dtype)
        torch.testing.assert_close(layer(x, pos, mask), x, atol=1e-6, rtol=1e-6)
