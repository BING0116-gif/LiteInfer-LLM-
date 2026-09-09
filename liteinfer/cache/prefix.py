"""Task 11：Prefix Cache——共享前缀的 KV 物理块复用。

定位（docs/07 Task 11 + docs/02 §11）：让携带相同前缀（system prompt、few-shot
指令等）的请求复用先前请求已算好的 KV 物理块，跳过重复 prefill，直接降低
TTFT。实现 vLLM "Automatic Prefix Caching" 的核心思路，但刻意保持最小集：

- **block hash**：链式内容哈希 ``h_i = H(h_{i-1}, 第 i 个完整块的 token ids)``。
  父哈希进链保证"前缀敏感"——两个块内容相同但前缀不同，哈希必然不同；
  反之内容与前缀全同则哈希必同，这正是"可安全共享物理块"的判据。
- **hash table**：``hash -> 物理块 id``，挂在 ``PagedKVCache`` 的块池之上
  （Task 10 遗留提示预留的位置），只缓存**完整块**（docs/02 §11 的取舍：
  半块的上下文尚未"封口"，且最后 token 的 logits 必须现算）。
- **ref count**：被缓存的物理块带引用计数。请求释放时 ref-1；ref 归 0 后
  **不立即归还 FreeQueue**，而是进 LRU——内容还在，等后续请求白捡。
- **reuse**：prefill 前 ``lookup``，命中的物理块被新请求的 ``BlockTable``
  "收养"（``adopt``），forward 只跑未命中的后缀（``write_pos=hit_len``）。
- **eviction**：块池耗尽时按 LRU 驱逐 ref==0 的缓存块；无可驱逐则抛错
  （fail fast，与 Task 08 的块耗尽语义一致）。

正确性依据：同一模型、同一 token 前缀在 CPU FP32 下前向是确定的，命中块的
KV 与现算逐位相同，因此复用不改变任何输出——测试用"输出逐字一致"钉死。

设备/dtype 不在本模块出现：物理块池（``BlockPool``）的形状与设备由上游
``KVCacheConfig``（源头 ``EngineConfig``）决定，本模块只搬运块 id。
"""

from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Optional, Sequence

from liteinfer.cache.paged import BlockPool

if TYPE_CHECKING:  # 只作类型标注：paged.BlockTable 反向引用本模块（prefix_cache 字段），避免导入环
    from liteinfer.cache.paged import BlockTable

#: 链式哈希的根。0 即可：所有链都从它出发，进程内自洽。
ROOT_HASH = 0


def block_hash(parent: int, tokens: Sequence[int]) -> int:
    """计算一个完整块的链式哈希 ``H(parent, tokens) -> int``。

    用 blake2b（64 位摘要）而不是裸 ``hash()``：Python 的 str/bytes 哈希有
    进程级随机盐（PYTHONHASHSEED），跨进程不可复现，且 int 大对象的哈希碰撞
    行为不受控；哈希是"两个物理块能否共享"的唯一判据，必须稳定、确定。

    父哈希按**无符号** 64 位打包（``>Q``）：摘要本身就是无符号数，用有符号
    ``>q`` 会在链式第二层起溢出（踩过：struct.error）。
    """
    data = struct.pack(">Q", int(parent))
    data += b"".join(struct.pack(">q", int(t)) for t in tokens)
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


class PrefixCache:
    """前缀缓存：哈希表 + 引用计数 + LRU 驱逐，管理"可共享的物理块"。

    生命周期（每个被缓存的物理块）：
        prefill 后 register（ref=1，属主请求持有）
        → 其它请求 lookup 命中 → ref+1（多请求共享同一物理块）
        → 各属主释放 → ref 逐次 -1
        → ref==0：不还池，进 LRU 等待复用
        → 池耗尽时被 LRU 驱逐 → 挪给新块（从哈希表摘除）

    未被注册的块（如不满一块的尾巴、内容撞哈希已有人缓存的重复块）不进入
    本模块的任何结构，由 ``BlockTable.free`` 直接归还 FreeQueue——两种释放
    路径在 ``release_block`` 里按"是否被追踪"分流。
    """

    def __init__(self, pool: BlockPool, block_size: int) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size 必须为正，收到 {block_size}")
        self.pool = pool
        self.block_size = block_size
        # hash -> 物理块 id。链式哈希保证同一内容只有一个物理块代表。
        self._by_hash: dict[int, int] = {}
        # 物理块 id -> 被追踪块的哈希（O(1) 反查，驱逐时摘表用）
        self._hash_of: dict[int, int] = {}
        # 物理块 id -> 引用计数（仅被追踪的块在此）
        self._refs: dict[int, int] = {}
        # ref==0 的可驱逐块，按"最近使用"排序：队尾最新，队首最先驱逐
        self._lru: OrderedDict[int, None] = OrderedDict()
        # 观测计数：验收标准就是"可观察 cache hit"
        self.n_lookups: int = 0
        self.n_hits: int = 0
        self.n_hit_tokens: int = 0
        self.n_evictions: int = 0

    # ---- 复用入口：prefill 前查询 ----

    def lookup(self, token_ids: Sequence[int]) -> tuple[list[int], int]:
        """按块匹配 token 前缀，返回 ``(命中的物理块 id 列表, 命中 token 数)``。

        从第 0 块起逐块链哈希、逐块查表，遇到第一个未命中即停（链式哈希的
        性质决定了后续块也不可能命中）。命中块立即 ``ref+1`` 并移出 LRU
        （"收养"它的请求即将持有它）。

        ``hit_len`` 至少给请求留 1 个 token 现算：prompt 恰好是 N 个完整块且
        全命中时，最后一个块也要放弃复用——下一个 token 的 logits 只能由
        forward 产出，没有"零 token prefill"这回事。
        """
        tokens = [int(t) for t in token_ids]
        num_full = len(tokens) // self.block_size
        bs = self.block_size
        self.n_lookups += 1

        ids: list[int] = []
        h = ROOT_HASH
        for i in range(num_full):
            h = block_hash(h, tokens[i * bs:(i + 1) * bs])
            bid = self._by_hash.get(h)
            if bid is None:
                break
            ids.append(bid)

        # 至少保留一个 token 现算（见 docstring）
        if len(ids) * bs >= len(tokens) and ids:
            ids.pop()

        if not ids:
            return [], 0

        hit_len = len(ids) * bs
        for bid in ids:
            self._acquire(bid)
        self.n_hits += 1
        self.n_hit_tokens += hit_len
        return ids, hit_len

    # ---- 注册：prefill/decode 写满一个完整块后调用 ----

    def register(self, token_ids: Sequence[int], table: "BlockTable") -> int:
        """把 token 序列对应的完整块注册进哈希表，返回本次新注册的块数。

        - 已有同哈希条目则跳过（并发同前缀请求各自算出重复块时，先注册者
          获胜，后者的块不被追踪、释放时直接还池——不浪费哈希表空间）；
        - 已被追踪的物理块也跳过（adopt 来的块哈希必已在表中，防御重复计数）；
        - 从头重算哈希链是 O(块数) 的纯 CPU 整数运算，块数最多几十个，
          不值得为此维护每请求的增量状态。
        """
        tokens = [int(t) for t in token_ids]
        # 只注册"已写进 table 的完整块"：min 兜底调用方多给 token 的异常情形
        num_full = min(len(tokens) // self.block_size, len(table.blocks))
        bs = self.block_size
        registered = 0
        h = ROOT_HASH
        for i in range(num_full):
            h = block_hash(h, tokens[i * bs:(i + 1) * bs])
            if h in self._by_hash:
                continue
            bid = table.blocks[i].block_id
            if bid in self._refs:
                continue
            self._by_hash[h] = bid
            self._hash_of[bid] = h
            self._refs[bid] = 1  # ref=1：当前的属主请求
            registered += 1
        return registered

    # ---- 释放：请求终态归还块时调用 ----

    def release_block(self, block_id: int) -> None:
        """归还一个物理块：被追踪的块 ref-1（归 0 进 LRU），否则直接还池。"""
        if block_id in self._refs:
            self._refs[block_id] -= 1
            if self._refs[block_id] <= 0:
                self._refs[block_id] = 0
                # 不还池：内容保留，等后续 lookup 白捡。放到队尾=最近使用
                self._lru[block_id] = None
        else:
            self.pool.free([block_id])

    # ---- 分配：块池耗尽时驱逐缓存块 ----

    def alloc_new(self) -> int:
        """取一个空闲物理块 id：优先走 FreeQueue，耗尽则 LRU 驱逐缓存块。"""
        if self.pool.num_free == 0:
            return self._evict_one()
        (bid,) = self.pool.alloc(1)
        return bid

    def _evict_one(self) -> int:
        """驱逐最久未使用的 ref==0 缓存块，把它的物理 id 挪给调用方。

        被驱逐块从哈希表/引用表彻底摘除——之后不再有人能 lookup 到它，
        也不会再有人 release 它（它的引用已经清零），物理 id 直接归新属主，
        不经过 FreeQueue（FreeQueue 从未见过它，账目仍然平衡）。
        """
        if not self._lru:
            raise ValueError(
                "KV 块耗尽且无可驱逐的前缀缓存块（池容量不足或缓存块仍被共享）"
            )
        bid, _ = self._lru.popitem(last=False)
        h = self._hash_of.pop(bid, None)
        if h is not None:
            self._by_hash.pop(h, None)
        self._refs.pop(bid, None)
        self.n_evictions += 1
        return bid

    # ---- 内部 ----

    def _acquire(self, block_id: int) -> None:
        """lookup 命中后收养：ref+1 并移出 LRU（在飞请求持有中，不可驱逐）。"""
        self._refs[block_id] = self._refs.get(block_id, 0) + 1
        self._lru.pop(block_id, None)

    # ---- 观测 ----

    @property
    def num_tracked_blocks(self) -> int:
        """被缓存追踪的物理块数（含仍被在飞请求共享的）。"""
        return len(self._refs)

    @property
    def num_evictable_blocks(self) -> int:
        """ref==0、可驱逐/可白捡的缓存块数。"""
        return len(self._lru)

    def stats(self) -> dict[str, Any]:
        """观测快照（JSON 安全）：验收"可观察 cache hit"直接消费。"""
        return {
            "enabled": True,
            "lookups": self.n_lookups,
            "hits": self.n_hits,
            "hit_tokens": self.n_hit_tokens,
            "evictions": self.n_evictions,
            "cached_blocks": self.num_tracked_blocks,
            "evictable_blocks": self.num_evictable_blocks,
        }

    def __repr__(self) -> str:
        return (
            f"PrefixCache(block_size={self.block_size}, tracked={self.num_tracked_blocks}, "
            f"evictable={self.num_evictable_blocks}, hits={self.n_hits}/{self.n_lookups})"
        )
