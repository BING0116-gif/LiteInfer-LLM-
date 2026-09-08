"""Task 07 最小运行示例：分页 KV 缓存的分配 / 写入 / gather 读回 / 释放。

演示重点（不含真实模型，纯张量，秒级跑完）：
- 一个 PagedKVCache 池服务多个并发请求，每个请求按需 lazy 取块；
- 写入随机 K/V，gather 读回与输入逐位一致（验证分页重建正确）；
- 请求结束后 free，池的空闲块数完整复原（零泄漏）。

运行：
    set "HF_HOME=D:/LiteInfer/hf_cache"
    set PYTHONPATH=d:/LiteInfer
    python examples/paged_kv_demo.py
"""

from __future__ import annotations

import torch

from liteinfer.cache.contiguous import KVCacheConfig
from liteinfer.cache.paged import PagedKVCache


def main() -> None:
    # 用一个很小的形状演示；真实引擎里这些数字来自 HF config
    cfg = KVCacheConfig(
        num_layers=4,
        num_kv_heads=2,
        head_dim=8,
        max_seq_len=1024,  # paged 不依赖此字段做容量，仅占位
        dtype=torch.float32,
        device=torch.device("cpu"),  # 统一走 EngineConfig.device
    )
    block_size = 16
    num_blocks = 32

    paged = PagedKVCache(cfg, block_size=block_size, num_blocks=num_blocks)
    print(f"[init] {paged}")

    # 模拟 3 个并发请求，长度各不相同（含跨块情形）
    seq_lengths = [10, 40, 7]  # 40 -> 跨 3 个块
    tables = []

    for i, n in enumerate(seq_lengths):
        table = paged.new_block_table()
        # 形状 [num_layers, n, KVH, D]：模拟 n 个 token 的 K/V
        k = torch.arange(n * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim,
                         dtype=torch.float32).view(cfg.num_layers, n, cfg.num_kv_heads, cfg.head_dim)
        v = -k  # 任意确定值，便于断言
        allocated = paged.append_tokens(table, k, v)
        tables.append(table)
        # 逐层比对 gather 读回与原始输入是否逐位一致
        ok = all(torch.equal(table.gather(layer, n), k[layer].unsqueeze(0))
                 for layer in range(cfg.num_layers))
        print(f"[req {i}] seq_len={n} blocks={len(table.blocks)} allocated={allocated} "
              f"readback_ok={ok} free_blocks={paged.num_blocks_free}")

    # 释放全部请求，池应完整归还
    for i, table in enumerate(tables):
        paged.free_table(table)
        print(f"[free req {i}] used={paged.num_blocks_used} free={paged.num_blocks_free}")

    print(f"[final] {paged}")
    assert paged.num_blocks_free == paged.num_blocks_total, "KV 块泄漏！"
    print("OK: 全部请求释放后空闲块数复原，零泄漏")


if __name__ == "__main__":
    main()
