"""Minimal Qwen：从零实现的 Qwen2 前向计算图（embedding -> N 层 decoder -> lm_head）。

类名/属性名与 HF ``Qwen2ForCausalLM`` 对齐（``model``/``lm_head``、
``embed_tokens``/``layers``/``norm``、``self_attn``/``mlp``/两个 layernorm），
权重加载因此只需剥离 ``model.`` 前缀。所有超参从构造参数进入，
``weights.py`` 负责从 HFConfig 提取——本文件不出现 Qwen2.5-0.5B 的任何
具体数字，换任意 Qwen2 家族 checkpoint 都能直接用。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from liteinfer.model.minimal.layer import QwenDecoderLayer
from liteinfer.model.minimal.rmsnorm import RMSNorm

if TYPE_CHECKING:  # 仅类型检查期导入，见 attention.py 同款说明
    from liteinfer.model.runner import KVCacheView


def build_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    past_len: int = 0,
) -> torch.Tensor:
    """构造 additive 因果 mask ``[1, 1, S, S + past_len]``。

    ``past_len`` 是已有历史的长度：第 i 个 query 的绝对位置是
    ``past_len + i``，它能看到所有 ``key_pos <= query_pos`` 的 key。
    用"位置比较"而不是"再拼一个 triu"来表达，是因为 decode / chunked
    prefill 下 query 与 key 的长度不再相等，triu 的方阵语义会失效。

    用 ``finfo(dtype).min`` 而不是 ``-inf``：加法掩码里出现 -inf 与全屏蔽行
    相加会产生 NaN（softmax 的全 -inf 行），finfo.min 经 softmax 后约为 0
    且数值稳定，这也是 HF 的做法。mask 只依赖 S/past/dtype/device，与 batch、
    head 无关，靠广播覆盖所有 head。
    """
    total = seq_len + past_len
    mask = torch.full((seq_len, total), torch.finfo(dtype).min, device=device, dtype=dtype)
    if total > 0:
        q_pos = torch.arange(past_len, total, device=device).unsqueeze(1)  # [S, 1]
        k_pos = torch.arange(total, device=device).unsqueeze(0)  # [1, total]
        mask.masked_fill_(k_pos <= q_pos, 0.0)
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
        kv_caches: list[KVCacheView] | None = None,
        write_pos: int = 0,
    ) -> torch.Tensor:
        """返回最后一层输出过 final norm 之后的 hidden states ``[B, S, H]``。

        Args:
            input_ids: ``[B, S]``。
            position_ids: ``[B, S]`` 绝对位置；None 时按
                ``write_pos .. write_pos+S-1`` 生成（无缓存即 0..S-1）。
            kv_caches: 逐层的 KV 缓冲区视图；None 表示不复用历史。
            write_pos: 本次 token 在整条序列中的起始下标，既是缓存写入
                位置，也是 mask 的 ``past_len``。

        attention mask 在此统一构造并传给每一层，层与层共享同一个 mask，
        避免重复分配。
        """
        batch, seq_len = input_ids.shape
        if position_ids is None:
            # 单序列无 padding 时位置就是 write_pos..write_pos+S-1；
            # 带上 write_pos 偏移，chunked prefill 才能不吃掉历史位置
            position_ids = torch.arange(
                write_pos, write_pos + seq_len, device=input_ids.device
            ).unsqueeze(0)
        hidden_states = self.embed_tokens(input_ids)

        if seq_len == 1:
            # decode 的典型形态：单个 query 能看到全部历史 key，掩码全 0，
            # 加它是纯浪费（scores 形状 [B, H, 1, T]，一次 O(T) 的加法）
            attention_mask = None
        else:
            # dtype/device 跟随 embedding 输出：权重加载到哪，mask 就跟到哪
            attention_mask = build_causal_mask(
                seq_len, hidden_states.device, hidden_states.dtype, past_len=write_pos
            )
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_ids,
                attention_mask,
                kv_caches[i] if kv_caches is not None else None,
                write_pos,
            )
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
        kv_caches: list[KVCacheView] | None = None,
        write_pos: int = 0,
    ) -> torch.Tensor:
        """返回 ``[B, S, vocab]`` 的 logits（不做 softmax，与 HF 一致）。

        刻意**始终只返回 logits 张量**，不返回 ``past_key_values``：
        历史 KV 已经由外部持有的缓存承载，再返回一份既多余又会把返回类型
        变成 ``Tensor | tuple``，Task 03 那些 ``assert_close(mine, hf)`` 的
        对齐断言就全得跟着改。
        """
        hidden_states = self.model(input_ids, position_ids, kv_caches, write_pos)
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
