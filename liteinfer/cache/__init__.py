"""KV Cache 管理（Task 04：Contiguous；Task 07 将在此包内扩展 Paged KV）。

对外只暴露三个名字：

- ``KVCacheConfig``：形状与容量的纯数据描述（字节数可核算）；
- ``LayerKVCache``：单层缓冲区的读写视图；
- ``ContiguousKVCache``：整模型（所有层）的连续缓冲区。

设备与 dtype 一律由 ``KVCacheConfig`` 从外部传入（源头是
``EngineConfig.device / EngineConfig.dtype``），本包内不出现任何设备决策。
"""

from liteinfer.cache.contiguous import (
    ContiguousKVCache,
    KVCacheConfig,
    LayerKVCache,
)

__all__ = ["KVCacheConfig", "LayerKVCache", "ContiguousKVCache"]
