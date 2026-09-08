"""Task 07：BlockPool + Paged KV Cache——按 block 切分的 KV 内存管理器。

本阶段的定位（docs/07 任务拆分）：实现 **KV 缓存的分页内存管理**，把
Task 04 的「按请求预分配一整段连续 KV」改为「按需从物理块池里取固定大小的
block」，并引入逻辑序列到物理块的 `BlockTable` 映射。

与 Task 08 的边界（重要，禁止越界）：
- 本文件只负责「块怎么分配、KV 数据怎么落到物理块、怎么按 (block, offset)
  读回来」。模型侧的注意力如何消费这些块（gather-based PagedAttention）属于
  Task 08 的 ModelRunner，本文件不修改 ``attention.py``，也不接入
  ``EngineCore`` 的 step 循环。
- 我们提供的 ``BlockTable.gather(layer, length)`` 已经是 gather-based 的读接口
  （用 ``torch.gather`` 沿 block 维 + offset 维选槽），Task 08 直接复用它即可；
  本阶段只用它做单测/压测自检，不喂给模型。

**这不是一个完整的 PagedAttention CUDA Kernel**：真实 kernel 在 GPU 上做融合
的 gather+attention；这里只是在 CPU 上用 ``torch.gather`` 重建连续的 K/V 张量，
逻辑等价、可验证，但算力路径与 vLLM 的 CUDA kernel 完全不同。README 与设计文档
都会显式声明这一点。

物理布局对齐 docs/02 §7：``k_blocks[block_id, layer, offset, :, :]`` 形状为
``[num_kv_heads, head_dim]``。位置映射：
    logical_block = pos // block_size
    offset        = pos %  block_size
    physical      = block_table[logical_block]
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch

from liteinfer.cache.contiguous import KVCacheConfig

logger = logging.getLogger("liteinfer.cache.paged")


def _element_size(dtype: torch.dtype) -> int:
    """单元素字节数。用真实张量问 torch，比维护 dtype->bytes 表可靠。"""
    return torch.empty(0, dtype=dtype).element_size()


class FreeQueue:
    """空闲物理块下标的 FIFO 队列。

    为什么单独成类：块的分配/回收只关心「哪个下标空闲」，与块的物理内容无关；
    把这条状态独立出来，``BlockPool`` 只需关心「物理张量 + 一个 FreeQueue」，
    职责更清晰，也便于单测分配不变量（如双重释放检测）。
    """

    def __init__(self, total: int) -> None:
        # 初始所有块都空闲：0..total-1
        self._queue: deque[int] = deque(range(total))
        # 用集合做 O(1) 成员判定，杜绝「同一块被释放两次」（双重释放会让
        # free 计数虚高，掩盖真实泄漏，是分页系统最阴险的 bug 之一）
        self._free_set: set[int] = set(range(total))
        self._total = total

    def pop(self) -> int:
        """取一个空闲块下标；空了抛错（调用方应先用 ``size`` 预判）。"""
        if not self._queue:
            raise IndexError("FreeQueue 已空：没有可用物理块（KV 块耗尽）")
        idx = self._queue.popleft()
        self._free_set.discard(idx)
        return idx

    def push(self, idx: int) -> None:
        """归还一个块下标。若它本就空闲（双重释放），直接抛错。"""
        if idx in self._free_set:
            raise ValueError(f"块 {idx} 已在空闲集合，发生双重释放（double free）")
        if idx < 0 or idx >= self._total:
            raise ValueError(f"块下标越界: {idx} 不在 [0, {self._total})")
        self._queue.append(idx)
        self._free_set.add(idx)

    @property
    def size(self) -> int:
        return len(self._queue)

    @property
    def empty(self) -> bool:
        return not self._queue

    @property
    def total(self) -> int:
        return self._total


class BlockPool:
    """物理块池：持有所有层的 K/V 张量，按块分配/回收。

    与 Task 04 ``ContiguousKVCache`` 一样，**在 inference_mode 之外**分配张量：
    模式内新建的张量是 inference tensor，离开上下文后不允许再原地 copy_
    （Task 04 踩坑），而 decode/append 每步都要往物理块里写。

    一次 ``torch.zeros`` 预分配整池（而非逐块分配）：把「内存不够」提前到
    构造时刻暴露，而不是生成到一半才 OOM。
    """

    def __init__(self, cfg: KVCacheConfig, block_size: int, num_blocks: int) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size 必须为正，收到 {block_size}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks 必须为正，收到 {num_blocks}")
        self.cfg = cfg
        self.block_size = block_size
        self.num_blocks = num_blocks
        shape = (num_blocks, cfg.num_layers, block_size, cfg.num_kv_heads, cfg.head_dim)
        # device/dtype 一律来自 KVCacheConfig（源头是 EngineConfig），本模块不出现设备决策
        self.k_blocks = torch.zeros(shape, dtype=cfg.dtype, device=cfg.device)
        self.v_blocks = torch.zeros(shape, dtype=cfg.dtype, device=cfg.device)
        self._free = FreeQueue(num_blocks)
        logger.debug(
            "分配 Paged KV 块池: num_blocks=%d block_size=%d layers=%d kv_heads=%d head_dim=%d "
            "(%.2f MB total)",
            num_blocks, block_size, cfg.num_layers, cfg.num_kv_heads, cfg.head_dim,
            self.nbytes / 1e6,
        )

    # ---- 分配/回收：只动 FreeQueue，不动张量内容 ----

    def alloc(self, n: int) -> list[int]:
        """从池里取 ``n`` 个空闲物理块下标。不足时抛错 fail fast。"""
        if n > self.num_free:
            raise ValueError(
                f"KV 块不足：请求 {n} 块，仅剩 {self.num_free} 块（共 {self.num_blocks}）"
            )
        return [self._free.pop() for _ in range(n)]

    def free(self, block_ids: list[int]) -> None:
        """归还若干块。双重释放会在 FreeQueue.push 内抛错。"""
        for bid in block_ids:
            self._free.push(bid)

    @property
    def num_total(self) -> int:
        return self.num_blocks

    @property
    def num_free(self) -> int:
        return self._free.size

    @property
    def num_used(self) -> int:
        return self.num_blocks - self._free.size

    def usage(self) -> float:
        """已用块占比 [0, 1]。"""
        return self.num_used / self.num_blocks

    @property
    def nbytes(self) -> int:
        """K+V 两块池的总字节数。"""
        elem = _element_size(self.cfg.dtype)
        per = (
            self.num_blocks
            * self.cfg.num_layers
            * self.block_size
            * self.cfg.num_kv_heads
            * self.cfg.head_dim
        )
        return per * elem * 2


class KVBlock:
    """单物理块的句柄：``(pool, block_id)``，提供该块内的按层/偏移读写。

    为什么用句柄而不是裸下标：把「块 id」与「它属于哪个池」绑在一起，调用方
    拿到的就是一个有行为的对象（``write_seq`` / ``k_view``），不会把下标传到
    错误的池里；也比到处传 (pool, id) 二元组更不易出错。
    """

    def __init__(self, pool: BlockPool, block_id: int) -> None:
        self.pool = pool
        self.block_id = block_id

    def write_token(
        self,
        layer: int,
        offset: int,
        k_vec: torch.Tensor,
        v_vec: torch.Tensor,
    ) -> None:
        """把单个 token 的 K/V（``[KVH, D]``）写进该块 ``offset`` 处。

        ``copy_`` 而非赋值：明确表达"这是唯一一次数据搬运"，且兼容跨 dtype/
        跨步长的源张量（与 Task 04 ``LayerKVCache.append`` 同一动机）。
        按 token 逐个写（而非整段），是因为解码期每步只新增一个 token，
        逐 token 写让"块满即换块"的语义最直接、最易单测。
        """
        if offset < 0 or offset >= self.pool.block_size:
            raise ValueError(
                f"块内偏移越界：块 {self.block_id} 写到 offset={offset}，"
                f"block_size={self.pool.block_size}"
            )
        if k_vec.shape != v_vec.shape:
            raise ValueError(f"k/v 形状不一致: {tuple(k_vec.shape)} vs {tuple(v_vec.shape)}")
        self.pool.k_blocks[self.block_id, layer, offset].copy_(k_vec)
        self.pool.v_blocks[self.block_id, layer, offset].copy_(v_vec)

    def k_view(self, layer: int, offset: int) -> torch.Tensor:
        """该块某层、某偏移处的 K 视图 ``[KVH, D]``（只读/原地写都行）。"""
        return self.pool.k_blocks[self.block_id, layer, offset]

    def v_view(self, layer: int, offset: int) -> torch.Tensor:
        return self.pool.v_blocks[self.block_id, layer, offset]

    def __repr__(self) -> str:
        return f"KVBlock(pool blocks={self.pool.num_blocks}, block_id={self.block_id})"


@dataclass
class BlockTable:
    """单请求的「逻辑 block 列表」+ 已写 token 数。

    逻辑序列按 ``block_size`` 切成若干逻辑块，每个逻辑块映射到一个物理块
    （``KVBlock``）。写入时按需从池里取新块（lazy allocation），读时按
    (logical_block, offset) 用 gather 重建连续张量。
    """

    pool: BlockPool
    block_size: int
    blocks: list[KVBlock] = field(default_factory=list)
    num_tokens: int = 0

    def append(
        self, tokens_k: torch.Tensor, tokens_v: torch.Tensor
    ) -> int:
        """写入 ``n`` 个新 token 的 K/V（跨块自动分配新物理块）。

        Args:
            tokens_k, tokens_v: 形状 ``[num_layers, n, num_kv_heads, head_dim]``。
        Returns:
            本次新分配的物理块数量（用于单测断言分配行为）。

        为什么 lazy allocation：序列长度在生成前不可知，vLLM 也是随序列增长逐块
        向池里要块；这样既能自然触发 FreeQueue.alloc，又避免为每个请求预占满
        整段容量（这正是 Task 04 contiguous 的碎片来源，正是分页要解决的）。
        """
        if tokens_k.dim() != 4:
            raise ValueError(
                f"tokens_k 必须是 [num_layers, n, KVH, D]，收到 {tuple(tokens_k.shape)}"
            )
        if tokens_k.shape != tokens_v.shape:
            raise ValueError(f"tokens_k/v 形状不一致: {tuple(tokens_k.shape)} vs {tuple(tokens_v.shape)}")
        n = tokens_k.shape[1]
        # 逐 token 写入：保证「块满即换块」的语义清晰、可单测
        allocated = 0
        for t in range(n):
            if self.num_tokens % self.block_size == 0:
                # 当前块已满（或还没有块），从池里取一个新物理块
                (bid,) = self.pool.alloc(1)
                self.blocks.append(KVBlock(self.pool, bid))
                allocated += 1
            logical = self.num_tokens
            block = self.blocks[logical // self.block_size]
            offset = logical % self.block_size
            # 把第 t 个 token、所有层的 K/V（每层层内为 [KVH, D]）写入该块
            for layer in range(tokens_k.shape[0]):
                block.write_token(layer, offset, tokens_k[layer, t], tokens_v[layer, t])
            self.num_tokens += 1
        return allocated

    def gather(self, layer: int, length: int) -> torch.Tensor:
        """读回前 ``length`` 个 token 的 K（gather-based），返回 ``[1, length, KVH, D]``。

        用 ``torch.gather`` 沿 block 维（选出每个 token 所属物理块）再沿 offset
        维（选出块内偏移）取槽——这正是 PagedAttention 的 KV 读取逻辑，但落在
        CPU 的 ``torch.gather`` 上而非 CUDA kernel。Task 08 的 ModelRunner 直接复用。
        """
        if length <= 0 or length > self.num_tokens:
            raise ValueError(f"gather 长度 {length} 超出 [1, {self.num_tokens}]")
        K_src = self.pool.k_blocks[:, layer]  # [num_blocks, block_size, KVH, D]
        block_size = self.block_size
        dev = K_src.device
        tok = torch.arange(length, device=dev)
        # 每个 token 的逻辑块、块内偏移
        logical_blocks = (tok // block_size)  # [L]
        offsets = (tok % block_size)  # [L]
        # 逻辑块 -> 物理块 id
        phys = torch.tensor(
            [b.block_id for b in self.blocks], device=dev, dtype=torch.long
        )
        sel_b = phys[logical_blocks]  # [L] 物理块 id
        # 沿 block 维 gather：输出形状 [L, block_size, KVH, D]，每个 token 选其物理块。
        # 注意 expand 的目标维是 (L, block_size, KVH, D) 而非池总块数——
        # 输出样本维是 length，不是 num_blocks。
        idx_b = sel_b.view(-1, 1, 1, 1).expand(
            length, K_src.shape[1], K_src.shape[2], K_src.shape[3]
        )
        g1 = torch.gather(K_src, 0, idx_b)
        # 沿 offset 维 gather：在每块内取出正确的块内偏移 -> [L, 1, KVH, D]
        idx_o = offsets.view(-1, 1, 1, 1).expand(
            length, 1, K_src.shape[2], K_src.shape[3]
        )
        g2 = torch.gather(g1, 1, idx_o)
        return g2.squeeze(1).unsqueeze(0)  # [1, L, KVH, D]

    def free(self) -> None:
        """归还本请求占用的所有物理块，并清空逻辑表。"""
        if self.blocks:
            self.pool.free([b.block_id for b in self.blocks])
        self.blocks = []
        self.num_tokens = 0

    def usage(self) -> int:
        return self.num_tokens

    def __len__(self) -> int:
        return self.num_tokens


class PagedKVCache:
    """顶层分页 KV 缓存管理器：组合 ``BlockPool`` 与每请求的 ``BlockTable``。

    构造参数：``KVCacheConfig``（形状/device/dtype，复用 Task 04）+ ``block_size``
    + ``num_blocks``（池容量）。提供 new_block_table / append_tokens / free_table
    以及整池的 usage 观测，供 Task 08 的 ModelRunner 与 Task 10 的指标直接消费。
    """

    def __init__(
        self, cfg: KVCacheConfig, block_size: int, num_blocks: int
    ) -> None:
        self.cfg = cfg
        self.block_size = block_size
        self.pool = BlockPool(cfg, block_size, num_blocks)

    def new_block_table(self) -> BlockTable:
        """新建一个空的请求级 block table（块按需 lazy 分配）。"""
        return BlockTable(pool=self.pool, block_size=self.block_size)

    def append_tokens(
        self, table: BlockTable, tokens_k: torch.Tensor, tokens_v: torch.Tensor
    ) -> int:
        """往某请求的 table 写入新 token 的 K/V，返回新分配块数。"""
        return table.append(tokens_k, tokens_v)

    def free_table(self, table: BlockTable) -> None:
        """释放某请求占用的全部块。"""
        table.free()

    @property
    def num_blocks_total(self) -> int:
        return self.pool.num_total

    @property
    def num_blocks_free(self) -> int:
        return self.pool.num_free

    @property
    def num_blocks_used(self) -> int:
        return self.pool.num_used

    def usage(self) -> float:
        return self.pool.usage()

    @property
    def nbytes(self) -> int:
        return self.pool.nbytes

    def __repr__(self) -> str:
        return (
            f"PagedKVCache(block_size={self.block_size}, "
            f"num_blocks={self.num_blocks_total}, used={self.num_blocks_used}, "
            f"free={self.num_blocks_free}, nbytes={self.nbytes})"
        )
