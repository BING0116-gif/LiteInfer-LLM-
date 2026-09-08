"""Task 07：Paged KV Cache 单元测试 + 10000 次无泄漏压测（无真实模型，CPU 可跑）。

设计要点：
- 全部测试不加载任何 HF 模型，用随机张量模拟 K/V，纯 CPU、秒级；
- 难度从「块分配不变量」到「跨块重建正确」再到「多请求隔离 / 双重释放 / 耗尽」；
- 压测 10000 次后断言 num_free 还原（零泄漏）为硬性验收。

命名约定：``paged`` 单测不带 ``model`` marker（不需要 tokenizer/权重），
直接 ``pytest tests/test_paged_kv.py -q`` 即可，不依赖 HF_HOME。
"""

from __future__ import annotations

import random

import pytest
import torch

from liteinfer.cache.paged import (
    BlockPool,
    BlockTable,
    FreeQueue,
    KVBlock,
    PagedKVCache,
)
from liteinfer.cache.contiguous import KVCacheConfig


# ---- 构造一个小而快的 KV 形状，专用于单元测试/压测 ----
def _cfg(device: str = "cpu", dtype: torch.dtype = torch.float32) -> KVCacheConfig:
    return KVCacheConfig(
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        max_seq_len=1024,  # 仅占位，paged 不用它作为容量
        dtype=dtype,
        device=torch.device(device),
    )


def _rand_kv(cfg: KVCacheConfig, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """造 ``[num_layers, n, KVH, D]`` 的随机 K/V，作为 n 个 token 的数据。"""
    shape = (cfg.num_layers, n, cfg.num_kv_heads, cfg.head_dim)
    return torch.randn(shape), torch.randn(shape)


# ---------- FreeQueue ----------

def test_free_queue_pop_push() -> None:
    q = FreeQueue(total=4)
    assert q.size == 4 and q.total == 4
    assert q.pop() == 0
    assert q.size == 3
    q.push(0)
    assert q.size == 4


def test_free_queue_double_push_raises() -> None:
    q = FreeQueue(total=2)
    q.pop()
    q.push(0)
    with pytest.raises(ValueError):
        q.push(0)  # 同一块释放两次


def test_free_queue_pop_empty_raises() -> None:
    q = FreeQueue(total=1)
    q.pop()
    with pytest.raises(IndexError):
        q.pop()


# ---------- BlockPool ----------

def test_block_pool_alloc_free() -> None:
    pool = BlockPool(_cfg(), block_size=16, num_blocks=8)
    assert pool.num_free == 8 and pool.num_used == 0
    ids = pool.alloc(3)
    assert len(ids) == 3
    assert pool.num_free == 5 and pool.num_used == 3
    pool.free(ids)
    assert pool.num_free == 8  # 全部归还


def test_block_pool_exhaust_raises() -> None:
    pool = BlockPool(_cfg(), block_size=16, num_blocks=2)
    pool.alloc(2)
    with pytest.raises(ValueError):
        pool.alloc(1)


def test_block_pool_double_free_raises() -> None:
    pool = BlockPool(_cfg(), block_size=16, num_blocks=4)
    (bid,) = pool.alloc(1)
    pool.free([bid])
    with pytest.raises(ValueError):
        pool.free([bid])  # double free


def test_block_pool_usage_and_nbytes() -> None:
    cfg = _cfg()
    pool = BlockPool(cfg, block_size=16, num_blocks=4)
    assert pool.usage() == 0.0
    pool.alloc(1)
    assert abs(pool.usage() - 0.25) < 1e-9
    elem = torch.empty(0, dtype=cfg.dtype).element_size()
    expected = 4 * cfg.num_layers * 16 * cfg.num_kv_heads * cfg.head_dim * elem * 2
    assert pool.nbytes == expected


# ---------- KVBlock 写入读取 ----------

def test_kvblock_write_read() -> None:
    pool = BlockPool(_cfg(), block_size=16, num_blocks=2)
    (bid,) = pool.alloc(1)
    blk = KVBlock(pool, bid)
    k = torch.randn(2, 4)  # [KVH, D]
    v = torch.randn(2, 4)
    blk.write_token(0, 0, k, v)
    assert torch.equal(blk.k_view(0, 0), k)
    assert torch.equal(blk.v_view(0, 0), v)


# ---------- BlockTable：单块 / 跨块 / 部分读 ----------

def test_block_table_single_block() -> None:
    cfg = _cfg()
    pool = BlockPool(cfg, block_size=16, num_blocks=4)
    table = BlockTable(pool, block_size=16)
    n = 10  # < block_size，只占 1 个块
    k, v = _rand_kv(cfg, n)
    allocated = table.append(k, v)
    assert allocated == 1
    assert table.num_tokens == n
    for layer in range(cfg.num_layers):
        got = table.gather(layer, n)
        assert got.shape == (1, n, cfg.num_kv_heads, cfg.head_dim)
        want = k[layer].unsqueeze(0)  # [1, n, KVH, D]
        assert torch.equal(got, want)


def test_block_table_cross_block() -> None:
    cfg = _cfg()
    pool = BlockPool(cfg, block_size=16, num_blocks=8)
    table = BlockTable(pool, block_size=16)
    n = 40  # 跨 3 个块 (16+16+8)
    k, v = _rand_kv(cfg, n)
    allocated = table.append(k, v)
    assert allocated == 3
    assert len(table.blocks) == 3
    # 跨块重建必须逐位等于原始输入
    for layer in range(cfg.num_layers):
        got = table.gather(layer, n)
        assert torch.equal(got, k[layer].unsqueeze(0))


def test_block_table_gather_partial() -> None:
    cfg = _cfg()
    pool = BlockPool(cfg, block_size=16, num_blocks=4)
    table = BlockTable(pool, block_size=16)
    n = 25
    k, v = _rand_kv(cfg, n)
    table.append(k, v)
    length = 7  # 只取前缀
    for layer in range(cfg.num_layers):
        got = table.gather(layer, length)
        assert torch.equal(got, k[layer].unsqueeze(0)[:, :length])


def test_block_table_gather_out_of_range_raises() -> None:
    cfg = _cfg()
    pool = BlockPool(cfg, block_size=16, num_blocks=4)
    table = BlockTable(pool, block_size=16)
    k, v = _rand_kv(cfg, 5)
    table.append(k, v)
    with pytest.raises(ValueError):
        table.gather(0, 6)


# ---------- 多请求隔离 ----------

def test_block_table_isolation() -> None:
    cfg = _cfg()
    pool = BlockPool(cfg, block_size=16, num_blocks=8)
    t1 = BlockTable(pool, block_size=16)
    t2 = BlockTable(pool, block_size=16)
    k1, v1 = _rand_kv(cfg, 10)
    k2, v2 = _rand_kv(cfg, 10)
    t1.append(k1, v1)
    t2.append(k2, v2)
    # 两请求占用物理块不得重叠
    ids1 = {b.block_id for b in t1.blocks}
    ids2 = {b.block_id for b in t2.blocks}
    assert ids1.isdisjoint(ids2)
    # 各自读回内容正确
    for layer in range(cfg.num_layers):
        assert torch.equal(t1.gather(layer, 10), k1[layer].unsqueeze(0))
        assert torch.equal(t2.gather(layer, 10), k2[layer].unsqueeze(0))


# ---------- PagedKVCache 顶层 + 与 contiguous 语义对齐 ----------

def test_paged_kv_manager_usage() -> None:
    cfg = _cfg()
    paged = PagedKVCache(cfg, block_size=16, num_blocks=8)
    assert paged.num_blocks_total == 8
    assert paged.num_blocks_free == 8
    table = paged.new_block_table()
    k, v = _rand_kv(cfg, 20)
    paged.append_tokens(table, k, v)
    assert paged.num_blocks_used == 2
    assert abs(paged.usage() - 0.25) < 1e-9
    paged.free_table(table)
    assert paged.num_blocks_free == 8  # 释放后全部归还


def test_paged_matches_contiguous_shape() -> None:
    """paged 的 gather 输出形状必须与 Task 04 的 LayerKVCache.read 一致
    （``[1, length, KVH, D]``），Task 08 才能无缝替换。"""
    cfg = _cfg()
    paged = PagedKVCache(cfg, block_size=16, num_blocks=4)
    table = paged.new_block_table()
    k, v = _rand_kv(cfg, 33)
    paged.append_tokens(table, k, v)
    for layer in range(cfg.num_layers):
        got = table.gather(layer, 33)
        assert got.shape == (1, 33, cfg.num_kv_heads, cfg.head_dim)


# ---------- 压力测试：10000 次无泄漏 ----------

def test_stress_10000_no_leak() -> None:
    """验收硬指标：随机 10000 次「建表→写随机长度→gather 逐位相等→释放」，
    结束后 num_free 必须还原到总块数（零泄漏），且过程不抛错。"""
    random.seed(1234)
    cfg = _cfg()
    paged = PagedKVCache(cfg, block_size=16, num_blocks=64)
    total = paged.num_blocks_total
    max_seen_used = 0
    for i in range(10000):
        seq_len = random.randint(1, 80)
        table = paged.new_block_table()
        k, v = _rand_kv(cfg, seq_len)
        paged.append_tokens(table, k, v)
        # 逐层比对 gather 与原始输入（FP32 精确相等）
        for layer in range(cfg.num_layers):
            got = table.gather(layer, seq_len)
            assert torch.equal(got, k[layer].unsqueeze(0)), f"iter {i} layer {layer} 重建不一致"
        max_seen_used = max(max_seen_used, paged.num_blocks_used)
        paged.free_table(table)
    # 零泄漏：池子被完整归还
    assert paged.num_blocks_free == total, "存在 KV 块泄漏"
    assert paged.num_blocks_used == 0
    # 过程中并发占用从未超过池容量（lazy alloc 不会越界）
    assert max_seen_used <= total
