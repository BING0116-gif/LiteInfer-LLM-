"""Task 04：Contiguous KV Cache——按层预分配的连续 K/V 缓冲区。

三个设计决策：

1. **缓存由外部持有，模型只负责读写**。
   KV 的生命周期跨越多次 forward（本阶段是一条请求的所有 decode 步），
   到 Task 07 还会跨越多个请求（同一物理 block 被不同序列共享）。
   如果缓存长在模型里，这些优化根本无从下手；因此模型侧只接收
   "单层视图"并往里写，缓存的所有权留给 generator / 未来的 engine。

2. **追加式写入 + 零拷贝读取**。
   ``LayerKVCache.append`` 只把新算出的 n 个 token 拷进缓冲区，返回
   ``[:start+n]`` 的**视图**；不做 ``torch.cat([past, new])``。
   常见的"模型返回 past_key_values、调用方拼回去"写法每步都要复制整段
   历史（decode 时长度就是 T），本方案把这一步的流量从 O(T) 降到 O(1)。

3. **物理布局 ``[num_layers, max_seq_len, num_kv_heads, head_dim]``**。
   与 docs/02 §7 的推荐一致，且天然是 Task 07 Paged KV 的前身：
   把 ``max_seq_len`` 这一维切成固定大小的 block，再配一张 block table
   做逻辑到物理的映射，就是 paged 布局。本阶段先解决 "要不要复用历史"，
   Task 07 再解决 "历史怎么分块共享"。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Union

import torch

logger = logging.getLogger("liteinfer.cache.contiguous")


def _element_size(dtype: torch.dtype) -> int:
    """单元素字节数。用真实张量问 torch，比维护一张 dtype->bytes 表可靠。"""
    return torch.empty(0, dtype=dtype).element_size()


@dataclass(frozen=True)
class KVCacheConfig:
    """KV 缓存的形状与容量描述（纯数据，可单测）。

    ``device`` 存 ``torch.device`` 而不是字符串：下游直接拿来
    ``torch.zeros(..., device=cfg.device)``，不必各自再解析一遍。
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self) -> None:
        # fail fast：形状参数写错时，错误会表现为"越界覆盖"或"静默截断"，
        # 这类 bug 在生成结果上很难与数值误差区分
        for name in ("num_layers", "num_kv_heads", "head_dim", "max_seq_len"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} 必须是正整数，收到 {value!r}")

    def bytes_per_token(self) -> int:
        """单个 token 的 KV 字节数：``2 × L × KVH × D × element_size``。

        2 是 K 与 V 两份（docs/03 模块 5 的公式）。注意按 **KV 头数** 算，
        不是 Q 头数——GQA 节省显存正是体现在这一项上。
        """
        return (
            2
            * self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * _element_size(self.dtype)
        )

    def total_bytes(self) -> int:
        return self.bytes_per_token() * self.max_seq_len

    @classmethod
    def from_hf_config(
        cls,
        hf_cfg: Any,
        max_seq_len: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
    ) -> "KVCacheConfig":
        """从 HF 的 config 对象推导形状。

        超参一律 getattr 兜底读取，与 ``minimal/weights.py`` 保持同一约定：
        不假设 Qwen2.5-0.5B 的具体数字，换 checkpoint 也能直接用。
        """
        num_heads = hf_cfg.num_attention_heads
        head_dim = getattr(hf_cfg, "head_dim", None) or hf_cfg.hidden_size // num_heads
        return cls(
            num_layers=hf_cfg.num_hidden_layers,
            num_kv_heads=hf_cfg.num_key_value_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=torch.device(device),
        )


class LayerKVCache:
    """单层 K/V 缓冲区视图：``[max_seq_len, num_kv_heads, head_dim]``。

    对外的张量一律带 batch 维（``[1, T, KVH, D]``），因为 attention 的消费
    形态就是这样；缓冲区内部省掉 batch 维，是为了让"按 token 位置切片"
    这件事在语义上最直接（第 i 个 token 的 K 就是 ``k_buf[i]``）。

    本类是无状态的（不记录已缓存长度）：写入位置由调用方的 ``write_pos``
    显式给出。这样 prefill（一次写 S 个）与 chunked prefill（分多次写）
    走的是同一条代码路径，且单测不需要"先写再重置"的顺序依赖。
    """

    def __init__(self, k_buf: torch.Tensor, v_buf: torch.Tensor, layer_idx: int) -> None:
        if k_buf.shape != v_buf.shape:
            raise ValueError(f"K/V 缓冲区形状不一致: {tuple(k_buf.shape)} vs {tuple(v_buf.shape)}")
        self.k_buf = k_buf
        self.v_buf = v_buf
        self.layer_idx = layer_idx

    @property
    def max_seq_len(self) -> int:
        return int(self.k_buf.shape[0])

    @property
    def shape(self) -> tuple:
        """单个缓冲区（K 或 V）的形状 ``[max_seq_len, KVH, D]``。"""
        return tuple(self.k_buf.shape)

    def _check_new(self, k_new: torch.Tensor, v_new: torch.Tensor, start: int) -> int:
        if k_new.dim() != 4:
            raise ValueError(
                f"k_new 必须是 [1, n, num_kv_heads, head_dim] 四维张量，收到 {tuple(k_new.shape)}"
            )
        if k_new.shape != v_new.shape:
            raise ValueError(f"k_new/v_new 形状不一致: {tuple(k_new.shape)} vs {tuple(v_new.shape)}")
        # Task 04 只服务 batch=1 的单请求生成链；batch 维留着是为了让
        # attention 的消费代码与无缓存路径完全一致（同一份 matmul 实现）
        if k_new.shape[0] != 1:
            raise ValueError(f"Task 04 只支持 batch=1，收到 batch={k_new.shape[0]}")
        if tuple(k_new.shape[2:]) != tuple(self.k_buf.shape[1:]):
            raise ValueError(
                f"k_new 的 KV 头/维度 {tuple(k_new.shape[2:])} 与缓存 {tuple(self.k_buf.shape[1:])} 不匹配"
            )
        if start < 0:
            raise ValueError(f"start 不能为负，收到 {start}")
        n = int(k_new.shape[1])
        if start + n > self.max_seq_len:
            raise ValueError(
                f"KV 缓存越界（层 {self.layer_idx}）：写入 {n} 个 token 到位置 {start}，"
                f"容量为 {self.max_seq_len}。请增大 max_seq_len 或减少生成长度"
            )
        return n

    def append(
        self, k_new: torch.Tensor, v_new: torch.Tensor, start: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """把 n 个新 token 的 K/V 写入 ``[start, start+n)``，返回完整历史视图。

        Returns:
            ``(k, v)``，各为 ``[1, start+n, num_kv_heads, head_dim]``，是缓冲区
            切片的**视图**（不复制历史），可以直接送进 attention。

        为什么用 ``copy_`` 而不是赋值：``copy_`` 支持跨 dtype/跨步长的源张量，
        且明确表达"这里是唯一一次数据搬运"。
        """
        n = self._check_new(k_new, v_new, start)
        self.k_buf[start : start + n].copy_(k_new[0])
        self.v_buf[start : start + n].copy_(v_new[0])
        return self.read(start + n)

    def read(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """读取前 ``length`` 个 token 的历史 K/V（视图，不复制）。"""
        if length < 0 or length > self.max_seq_len:
            raise ValueError(f"read 长度 {length} 超出 [0, {self.max_seq_len}]")
        return self.k_buf[:length].unsqueeze(0), self.v_buf[:length].unsqueeze(0)


class ContiguousKVCache:
    """整模型的连续 KV 缓存：所有层共享同一块预分配内存。

    一次性 ``torch.zeros`` 分配而不是逐层 append：预分配把"内存不够"这个
    问题提前到分配时刻暴露（OOM 或显式容量校验），而不是在生成到一半时
    才失败——那会让已经算出的 KV 全部作废。
    """

    def __init__(self, cfg: KVCacheConfig) -> None:
        self.cfg = cfg
        shape = (cfg.num_layers, cfg.max_seq_len, cfg.num_kv_heads, cfg.head_dim)
        # 在 inference_mode 之外创建：模式内新建的张量是 inference tensor，
        # 离开上下文后不允许再原地写（decode 每步都要 copy_，会直接抛错）
        self.k_cache = torch.zeros(shape, dtype=cfg.dtype, device=cfg.device)
        self.v_cache = torch.zeros(shape, dtype=cfg.dtype, device=cfg.device)
        self._layer_caches = [
            LayerKVCache(self.k_cache[i], self.v_cache[i], i) for i in range(cfg.num_layers)
        ]
        logger.debug(
            "分配 KV 缓存: shape=%s dtype=%s device=%s (%.2f MB)",
            shape, cfg.dtype, cfg.device, cfg.total_bytes() / 1e6,
        )

    @property
    def layer_caches(self) -> list[LayerKVCache]:
        """按层索引的视图列表，直接喂给模型的 ``kv_caches`` 参数。"""
        return self._layer_caches

    @property
    def nbytes(self) -> int:
        return self.k_cache.numel() * _element_size(self.k_cache.dtype) * 2

    def reset(self) -> None:
        """清零并复用缓冲区。

        只清零到"逻辑上已使用"的长度即可，但本阶段没有记账，直接整块清零；
        开销是一次 memset，避免残留数据被下一次生成读到。
        """
        self.k_cache.zero_()
        self.v_cache.zero_()

    def __len__(self) -> int:
        return self.cfg.num_layers

    def __repr__(self) -> str:
        return (
            f"ContiguousKVCache(layers={self.cfg.num_layers}, "
            f"max_seq_len={self.cfg.max_seq_len}, kv_heads={self.cfg.num_kv_heads}, "
            f"head_dim={self.cfg.head_dim}, dtype={self.cfg.dtype}, "
            f"device={self.cfg.device}, nbytes={self.nbytes})"
        )
