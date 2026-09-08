"""Minimal Qwen：从零实现的 Qwen2 前向计算图（embedding -> N 层 decoder -> lm_head）。

类名/属性名与 HF ``Qwen2ForCausalLM`` 对齐（``model``/``lm_head``、
``embed_tokens``/``layers``/``norm``、``self_attn``/``mlp``/两个 layernorm），
权重加载因此只需剥离 ``model.`` 前缀。所有超参从构造参数进入，
``weights.py`` 负责从 HFConfig 提取——本文件不出现 Qwen2.5-0.5B 的任何
具体数字，换任意 Qwen2 家族 checkpoint 都能直接用。
"""

from __future__ import annotations

import torch
from torch import nn

from liteinfer.model.minimal.layer import QwenDecoderLayer
from liteinfer.model.minimal.rmsnorm import RMSNorm


def build_causal_mask(
    seq_len: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """构造 additive 因果 mask ``[1, 1, S, S]``。

    用 ``finfo(dtype).min`` 而不是 ``-inf``：加法掩码里出现 -inf 与全屏蔽行
    相加会产生 NaN（softmax 的全 -inf 行），finfo.min 经 softmax 后约为 0
    且数值稳定，这也是 HF 的做法。mask 只依赖 S/dtype/device，与 batch、
    head 无关，靠广播覆盖所有 head。
    """
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
    # 严格上三角（diagonal=1）保留对角线：token 看得见自己
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None, :, :]


class MinimalQwenModel(nn.Module):
    """Qwen2 的 Transformer 主干（不含 lm_head）。"""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
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
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                QwenDecoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    rope_theta=rope_theta,
                    rms_norm_eps=rms_norm_eps,
                    attention_bias=attention_bias,
                    use_qk_norm=use_qk_norm,
                    mlp_bias=mlp_bias,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """返回最后一层输出过 final norm 之后的 hidden states ``[B, S, H]``。

        本阶段固定全序列 forward（无 KV Cache）；attention mask 在此统一
        构造并传给每一层，层与层共享同一个 mask，避免重复分配。
        """
        batch, seq_len = input_ids.shape
        if position_ids is None:
            # 单序列无 padding 时位置就是 0..S-1
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden_states = self.embed_tokens(input_ids)

        # dtype/device 跟随 embedding 输出：权重加载到哪，mask 就跟到哪
        attention_mask = build_causal_mask(seq_len, hidden_states.device, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, position_ids, attention_mask)
        return self.norm(hidden_states)


class MinimalQwenForCausalLM(nn.Module):
    """完整的因果语言模型：主干 + lm_head。

    Qwen2.5-0.5B 的 lm_head 与 embedding 共享权重
    （config.tie_word_embeddings=true，本机实测），因此 weights.py 会把
    embed_tokens 的权重同时填进 lm_head——两边各持有一份键、指向等值
    的参数，与 HF 的 tied 语义等价；不 tied 的 checkpoint（如 7B）天然
    走 strict 加载各自填充。
    """

    def __init__(self, **kwargs: object) -> None:  # noqa: ANN401
        super().__init__()
        self.model = MinimalQwenModel(**kwargs)  # type: ignore[arg-type]
        self.lm_head = nn.Linear(
            kwargs["hidden_size"], kwargs["vocab_size"], bias=False  # type: ignore[index]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """返回 ``[B, S, vocab]`` 的 logits（不做 softmax，与 HF 一致）。"""
        hidden_states = self.model(input_ids, position_ids)
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def greedy_next_token(self, input_ids: torch.Tensor) -> int:
        """取最后一步 logits 的 argmax——demo / 对齐测试用的小工具。

        放在模型上而不是 sampler：argmax 语义上属于"模型输出怎么消费"，
        但 sampling 链路（Task 02 Sampler）有完整的 temperature/top-k 流程，
        这里只是 forward 的冒烟通道，二者不混用。
        """
        logits = self(input_ids)
        return int(torch.argmax(logits[0, -1]).item())
