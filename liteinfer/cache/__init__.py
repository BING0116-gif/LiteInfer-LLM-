"""KV Cache 管理（Task 04：Contiguous；Task 07：Paged KV）。

对外暴露：
- ``KVCacheConfig``：形状与容量的纯数据描述（字节数可核算）；
- ``LayerKVCache`` / ``ContiguousKVCache``：Task 04 连续缓存；
- ``FreeQueue`` / ``BlockPool`` / ``KVBlock`` / ``BlockTable`` / ``PagedKVCache``：
  Task 07 分页缓存（块池 + 逻辑→物理映射 + gather 读）。

设备与 dtype 一律由 ``KVCacheConfig`` 从外部传入（源头是
``EngineConfig.device / EngineConfig.dtype``），本包内不出现任何设备决策。
"""

from liteinfer.cache.contiguous import (
    ContiguousKVCache,
    KVCacheConfig,
    LayerKVCache,
)
from liteinfer.cache.paged import (
    BlockPool,
    BlockTable,
    FreeQueue,
    KVBlock,
    PagedKVCache,
)

__all__ = [
    "KVCacheConfig",
    "LayerKVCache",
    "ContiguousKVCache",
    "FreeQueue",
    "BlockPool",
    "KVBlock",
    "BlockTable",
    "PagedKVCache",
]
