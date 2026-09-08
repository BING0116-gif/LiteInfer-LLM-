"""GQA 因果注意力（Qwen2 风格：QK-Norm + RoPE + grouped-query + additive mask）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from liteinfer.model.minimal.rotary import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
)
from liteinfer.model.minimal.rmsnorm import RMSNorm

if TYPE_CHECKING:  # 仅类型检查期导入：runner 与本模块互不依赖，避免包初始化顺序问题
    from liteinfer.model.runner import KVCacheView


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """把 ``[B, num_kv_heads, S, D]`` 扩展成 ``[B, num_kv_heads*n_rep, S, D]``。

    对齐 HF 的实现：每个 KV head 连续重复 n_rep 次（不是 tile 交错），
    使得展开后的第 j 个 Q head 对应第 ``j // n_rep`` 个 KV head——
    这正是 GQA 分组语义。expand+reshape 只在 reshape 处发生一次真实拷贝，
    比 repeat_interleave 少一次中间张量分配。
    """
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return expanded.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class QwenSelfAttention(nn.Module):
    """单层多头注意力，支持 GQA 与可选的 QK-Norm。

    计算顺序固定为：投影 -> [QK per-head RMSNorm] -> RoPE -> GQA 展开 ->
    缩放点积 -> additive mask -> softmax -> 加权求和 -> 输出投影。
    QK-Norm 必须在 RoPE 之前：norm 会改变向量的模长与方向，
    若先旋转再归一化，位置信息会被 norm 破坏。

    QK-Norm 由 checkpoint 决定是否启用（use_qk_norm）：Qwen2/2.5 系列的
    released 权重不含 q_norm/k_norm（本机实测 safetensors 仅 290 键），
    Qwen3 才有。结构跟着权重走，strict 加载才能对得上。
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rope_theta: float,
        rms_norm_eps: float,
        attention_bias: bool = True,
        use_qk_norm: bool = False,
    ) -> None:
        super().__init__()
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"Q 头数 {num_attention_heads} 必须被 KV 头数 "
                f"{num_key_value_heads} 整除（GQA 分组要求）"
            )
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        # GQA 乘数：每个 KV head 服务多少个 Q head
        self.num_kv_groups = num_attention_heads // num_key_value_heads
        self.scaling = head_dim**-0.5

        self.q_proj = nn.Linear(
            hidden_size, num_attention_heads * head_dim, bias=attention_bias
        )
        self.k_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=attention_bias
        )
        self.v_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=attention_bias
        )
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        # Qwen3 系的 per-head QK RMSNorm，作用于 head_dim 维（[B,H,S,D]
        # 的最后一维）；checkpoint 没有就不建，而不是建了再喂全 1
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(head_dim, base=rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_cache: KVCacheView | None = None,
        write_pos: int = 0,
    ) -> torch.Tensor:
        """单向推理前向。

        Args:
            hidden_states: ``[B, S, hidden]``，已经是 input_layernorm 之后的值。
            position_ids: ``[B, S]``。**必须是绝对位置**：decode 阶段单个
                token 的位置是它在整条序列中的下标（= 已缓存长度），
                RoPE 是绝对位置编码，从 0 开始会让续写内容全部错位。
            attention_mask: additive mask ``[B, 1, S, S+past]``（可屏蔽位置为
                finfo.min，其余 0）；None 表示不加 mask。
            kv_cache: 该层对应的 KV 缓冲区视图；None 表示不复用历史
                （Task 03 的全序列前向路径，行为与以前逐位一致）。
                实现可以是 Task 04 的 ``LayerKVCache``（连续）或 Task 08 的
                ``PagedLayerCache``（分页块表），二者接口同形，本方法无需区分。
            write_pos: 本次新算出的 token 写入缓存的起始下标。

        Returns:
            ``[B, S, hidden]`` 注意力输出（尚未加残差）。

        缓存写入的位置刻意放在 **RoPE 之后**：缓存里存的是"旋转后的 K"，
        与推理期语义一致，decode 时历史 K 无需再旋转一次；若存旋转前的 K，
        每次 decode 都得把整段历史重算一遍，缓存就白做了。
        """
        batch, seq_len, _ = hidden_states.shape

        # 投影后立刻 reshape 成 head 维度：[B, S, H*D] -> [B, S, H, D]
        q = self.q_proj(hidden_states).view(
            batch, seq_len, self.num_heads, self.head_dim
        )
        k = self.k_proj(hidden_states).view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        )
        v = self.v_proj(hidden_states).view(
            batch, seq_len, self.num_kv_heads, self.head_dim
        )

        # QK-Norm（若启用）：在 (B,S,H,D) 布局下作用于最后一维 head_dim，
        # 等价于对每个 head 单独做 RMSNorm，无需显式分头循环
        q = self.q_norm(q) if self.use_qk_norm else q
        k = self.k_norm(k) if self.use_qk_norm else k

        # [B, S, H, D] -> [B, H, S, D]：点积在 head 内做，seq 放中间
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rotary_emb(position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            # 追加式写入：只把本次新算出的 n 个 token 交给缓存，由它自己决定
            # 落到哪里（连续实现是 [max_seq, KVH, D] 的 [start:start+n] 切片，
            # 分页实现是 block table 的 (block, offset) 槽位），再把"完整历史"
            # 读回来送进注意力。契约只有 append/read 两个方法，两种实现可互换。
            #
            # 缓存的物理布局是 [T, KVH, D]，而此刻 k/v 是 [B, KVH, S, D]，
            # 所以先 transpose 回 [B, S, KVH, D] 再写；返回的完整历史
            # 同样 transpose 回来。两次 transpose 都只改步长不搬数据。
            # 注意：分页实现返回的不是视图而是 gather 出来的副本——这正是
            # "零拷贝"让步给"可共享块"的地方（见 docs/design/paged_runner.md）。
            k, v = kv_cache.append(k.transpose(1, 2), v.transpose(1, 2), write_pos)
            k, v = k.transpose(1, 2), v.transpose(1, 2)

        # GQA 展开：KV 头数补齐到 Q 头数
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # 缩放点积：[B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            # additive mask：被屏蔽位置是 finfo.min，softmax 后权重趋近 0
            scores = scores + attention_mask

        # softmax 固定升到 float32 再 cast 回去（HF eager 路径的做法）：
        # 半精度下 exp 容易上/下溢，这一步是稳定性的关键；
        # FP32 下与 HF 逐位等价
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(probs, v)  # [B, H, S, D]

        # 合并 head：transpose 回 [B, S, H, D] 再 flatten（必须先 transpose
        # 才 reshape，直接 reshape 会把 head 和 seq 的内存布局弄乱）
        out = out.transpose(1, 2).reshape(batch, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(out)
