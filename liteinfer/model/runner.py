"""Task 08：ModelRunner——把 Task 07 的 Paged KV 接进模型前向路径。

边界（docs/07 任务拆分 + Task 07「下一阶段提示」）：
- Task 07 已交付「块池 + BlockTable + gather 读」，但**模型侧没有消费它**；
- 本模块把二者缝起来：每个请求一张 ``BlockTable``（共享块池），模型的第 i 层
  通过 ``PagedLayerCache(table, i)`` 写入自己的 K/V，再用 ``gather`` 读回历史；
- 调度（Task 06）与请求状态机（Task 05）不变，引擎只是把 ``self.model(...)``
  换成 ``self.runner.prefill/decode(...)``。

**这不是完整 PagedAttention CUDA Kernel**：读路径是 ``torch.gather`` 重建连续
K/V，逻辑等价、CPU 上可逐位验证，但没有 fused kernel 的访存优化；相反，CPU 上
paged 读比 Task 04 contiguous 的零拷贝视图**更慢**（每次读都要造索引、搬运）。
本阶段验收的是**正确性**与**块回收**，吞吐数字一律留到 Task 12 的 GPU benchmark。

为什么让 ``PagedLayerCache`` 伪装成 ``LayerKVCache``：
``QwenSelfAttention.forward`` 里只有一行 ``k, v = kv_cache.append(k, v, write_pos)``。
只要分页适配器提供同形的 ``append``/``read``，注意力层的计算逻辑就一个字都不用改，
Task 03 的逐层 HF 对齐断言也就不会被动到——这是"开闭原则"最省事的一次兑现。
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Protocol, runtime_checkable

import torch

from liteinfer.cache.contiguous import KVCacheConfig
from liteinfer.cache.paged import BlockTable, PagedKVCache
from liteinfer.config import EngineConfig
from liteinfer.device import resolve_dtype

logger = logging.getLogger("liteinfer.model.runner")


@runtime_checkable
class KVCacheView(Protocol):
    """模型侧看到的"单层 KV 缓存"接口。

    ``LayerKVCache``（Task 04，连续）与 ``PagedLayerCache``（本模块，分页）
    都实现它，因此 ``attention.py`` 的类型标注只需面向协议，不依赖具体实现。
    """

    def append(
        self, k_new: torch.Tensor, v_new: torch.Tensor, start: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """写入 n 个新 token（``[1, n, KVH, D]``），返回完整历史 ``(k, v)``。"""
        ...

    def read(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """读取前 ``length`` 个 token 的 ``(k, v)``，各 ``[1, length, KVH, D]``。"""
        ...


def infer_kv_dims(model: Any) -> tuple[int, int, int]:
    """从模型读出缓存形状三元组 ``(num_layers, num_kv_heads, head_dim)``。

    与 ``CachedGenerator._model_kv_dims`` 同义，但更鲁棒：既支持
    ``model.model.layers``（MinimalQwenForCausalLM 的真实结构），也支持测试用
    的假模型把 layers 直接挂在顶层。按 **KV 头数** 而非 Q 头数：GQA 下二者不等
    （0.5B 是 14 vs 2），按 Q 头数存会把缓存放大 7 倍。
    """
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) or getattr(model, "layers", None)
    if layers is None:
        raise TypeError(
            f"ModelRunner 需要可推断 KV 维度的模型（有 model.layers 或 layers），"
            f"收到 {type(model).__name__}"
        )
    attn = layers[0].self_attn
    return len(layers), attn.num_kv_heads, attn.head_dim


def _default_num_blocks(cfg: EngineConfig, block_size: int) -> int:
    """按调度预算推导块池容量：够跑满 ``max_num_seqs`` 条序列即可。

    为什么按「并发序列数 × 单序列块数」而不是拍一个常数：池太小会在生成中途
    ``alloc`` 抛错（体验极差），太大则一次性占掉几百 MB（本机 CPU 内存也吃紧）。
    单序列按 ``max_new_tokens + 128``（prompt 余量）估，128 是给长 prompt 的冗余，
    超长 prompt 仍会 fail fast 抛 ValueError——比静默 OOM 好排查。
    """
    per_seq_tokens = cfg.max_new_tokens + 128
    blocks_per_seq = max(1, math.ceil(per_seq_tokens / block_size))
    return max(16, cfg.scheduler.max_num_seqs * blocks_per_seq)


class PagedLayerCache:
    """把共享 ``BlockTable`` 的第 ``layer_idx`` 层包装成单层 KV 缓存。

    对外与 ``LayerKVCache`` 同形（``append``/``read``），让注意力层无感切换：
    - ``append`` 把 n 个 token 的 K/V 逐个写进块表（跨块自动分配），再 ``gather``
      回完整历史；
    - ``read`` 直接 ``gather``，返回的张量形状与连续实现一致 ``[1, L, KVH, D]``。

    与连续实现的**关键差异**：分页的容量不是预分配的，而是"池里还剩多少块"，
    因此 ``max_seq_len`` 语义退化为池的理论上限（仅用于越界提示）。
    """

    def __init__(self, table: BlockTable, layer_idx: int) -> None:
        self._table = table
        self._layer = layer_idx

    @property
    def layer_idx(self) -> int:
        return self._layer

    @property
    def max_seq_len(self) -> int:
        """理论容量上限（池总 token 数）。分页下真实上限受其它请求挤占。"""
        return self._table.pool.num_total * self._table.block_size

    def _check_new(self, k_new: torch.Tensor, v_new: torch.Tensor, start: int) -> int:
        if k_new.dim() != 4:
            raise ValueError(
                f"k_new 必须是 [1, n, KVH, D] 四维张量，收到 {tuple(k_new.shape)}"
            )
        if k_new.shape != v_new.shape:
            raise ValueError(
                f"k_new/v_new 形状不一致: {tuple(k_new.shape)} vs {tuple(v_new.shape)}"
            )
        if k_new.shape[0] != 1:
            raise ValueError(f"当前只支持 batch=1，收到 batch={k_new.shape[0]}")
        if start < 0:
            raise ValueError(f"start 不能为负，收到 {start}")
        return int(k_new.shape[1])

    def append(
        self, k_new: torch.Tensor, v_new: torch.Tensor, start: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """写入 n 个新 token，返回完整历史 ``(k, v)``（各 ``[1, start+n, KVH, D]``）。

        逐 token 写而不是整段写：块边界的分配判据是 token 的绝对位置
        ``(start+t) // block_size``，逐个写让"跨块"这件小事在代码里显式可见，
        也和 ``BlockTable.append`` 的既有语义保持一致。
        """
        n = self._check_new(k_new, v_new, start)
        for t in range(n):
            self._table.write_token(
                self._layer, start + t, k_new[0, t], v_new[0, t]
            )
        return self.read(start + n)

    def read(self, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """gather 出前 ``length`` 个 token 的 K/V（各 ``[1, length, KVH, D]``）。"""
        return (
            self._table.gather(self._layer, length),
            self._table.gather_v(self._layer, length),
        )

    def __repr__(self) -> str:
        return f"PagedLayerCache(layer={self._layer}, tokens={self._table.num_tokens})"


class ModelRunner:
    """模型执行器：持有共享块池，按请求执行 prefill / decode。

    职责（docs/07 §五：Scheduler / Cache / ModelRunner 解耦）：
    - 只负责"给一个请求跑一次前向，并把 KV 落进它的 block table"；
    - 不做调度决策（Task 06）、不做采样（Sampler 在引擎侧）、不持有请求状态；
    - 设备/dtype 全部来自 ``EngineConfig``（补充条款 A1/A2）。
    """

    def __init__(
        self,
        model: Any,
        cfg: EngineConfig,
        block_size: Optional[int] = None,
        num_blocks: Optional[int] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.num_layers, num_kv_heads, head_dim = infer_kv_dims(model)
        # 设备/dtype 一律从 EngineConfig 进入，不读模型参数的所在设备：
        # 避免"模型忘了 .to(device)"这类隐患被悄悄吞掉
        self._device = torch.device(cfg.device)
        self._dtype = resolve_dtype(cfg.dtype, cfg.device)

        self.block_size = int(block_size if block_size is not None else cfg.block_size)
        resolved_blocks = num_blocks if num_blocks is not None else cfg.num_blocks
        resolved_blocks = (
            _default_num_blocks(cfg, self.block_size)
            if resolved_blocks is None
            else int(resolved_blocks)
        )
        self.num_blocks = resolved_blocks

        # max_seq_len 传池的总容量：让 KVCacheConfig 的 total_bytes() 有意义
        self._cache_cfg = KVCacheConfig(
            num_layers=self.num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=self.block_size * self.num_blocks,
            dtype=self._dtype,
            device=self._device,
        )
        self.paged = PagedKVCache(self._cache_cfg, self.block_size, self.num_blocks)
        logger.debug(
            "ModelRunner 就绪: layers=%d block_size=%d num_blocks=%d (%.2f MB)",
            self.num_layers, self.block_size, self.num_blocks, self.paged.nbytes / 1e6,
        )

    # ---- block table 生命周期（引擎按请求调用） ----

    def new_block_table(self) -> BlockTable:
        """为该请求开一张空块表（物理块在写入时按需 lazy 分配）。"""
        return self.paged.new_block_table()

    def free_table(self, table: Optional[BlockTable]) -> None:
        """归还该请求占用的全部物理块；None 表示尚未分配，直接忽略。"""
        if table is None:
            return
        self.paged.free_table(table)

    def bytes_for(self, table: Optional[BlockTable]) -> int:
        """该请求实际占用的 KV 字节数（按已写入 token 计，不按块计）。"""
        if table is None:
            return 0
        return table.num_tokens * self._cache_cfg.bytes_per_token()

    def _handles(self, table: BlockTable) -> list[PagedLayerCache]:
        """逐层的缓存句柄列表，形状与 Task 04 的 ``layer_caches`` 一致。"""
        return [PagedLayerCache(table, i) for i in range(self.num_layers)]

    # ---- 前向 ----

    def prefill(self, input_ids: torch.Tensor, table: BlockTable) -> torch.Tensor:
        """处理整个 prompt，返回 logits ``[1, P, vocab]``。

        一次写 P 个 token 的 K/V（跨块自动分配），随后由调用方取 ``logits[0, -1]``。
        """
        with torch.inference_mode():
            return self.model(
                input_ids, kv_caches=self._handles(table), write_pos=0
            )

    def decode(
        self, token_id: int, position: int, table: BlockTable
    ) -> torch.Tensor:
        """推进一个 token，返回 logits ``[1, 1, vocab]``。

        ``position`` 必须是**绝对位置**（= 已缓存长度）：RoPE 是绝对位置编码，
        同时也是本次写入块表的偏移，两个语义在此恰好统一（Task 04 踩坑）。
        """
        step_input = torch.tensor([[token_id]], device=self._device)
        position_ids = torch.tensor([[position]], device=self._device)
        with torch.inference_mode():
            return self.model(
                step_input,
                position_ids=position_ids,
                kv_caches=self._handles(table),
                write_pos=position,
            )
