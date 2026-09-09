"""Task 11 测试：Prefix Cache。

分三层：

1. 纯缓存层快测（无模型）：链式哈希的确定性与前缀敏感性、lookup 命中/未命中/
   部分命中、完整块注册、引用计数与 LRU、池耗尽驱逐、收养后 gather 逐位正确
   （这是"复用不改变数值"的根基：FakeLM 不读 KV，引擎级对比测不出收养块的内容错误）。
2. 引擎集成快测（FakeLM）：共享前缀的第二请求可观察命中（prefix_hit_tokens>0）、
   开关关闭时零命中且行为与 Task 08/09/10 完全一致、并发同批提交也能命中、
   多轮随机前缀压测无泄漏。
3. 真模型测试（marker=model）：真 Qwen 下第二请求命中且文本与"关闭前缀缓存"
   的引擎逐字一致——端到端钉死"复用不改输出"。
"""

from __future__ import annotations

import pytest
import torch

from liteinfer import EngineConfig
from liteinfer.cache.contiguous import KVCacheConfig
from liteinfer.cache.paged import BlockPool, BlockTable
from liteinfer.cache.prefix import ROOT_HASH, PrefixCache, block_hash
from liteinfer.engine import EngineCore
from liteinfer.sampling.params import SamplingParams
from _fakes import make_core

NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 8
BS = 4  # 测试用小块，块边界行为更容易构造


def _cfg(max_seq_len: int = 64) -> KVCacheConfig:
    return KVCacheConfig(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_seq_len=max_seq_len,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _rand_kv(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.randn(NUM_LAYERS, n, NUM_KV_HEADS, HEAD_DIM)
    v = torch.randn(NUM_LAYERS, n, NUM_KV_HEADS, HEAD_DIM)
    return k, v


# --------------------------------------------------------------------------- #
# 快测 1：链式哈希
# --------------------------------------------------------------------------- #


class TestBlockHash:
    def test_deterministic(self) -> None:
        """同内容同父哈希 -> 同哈希（这是"可共享物理块"的判据）。"""
        assert block_hash(ROOT_HASH, [1, 2, 3]) == block_hash(ROOT_HASH, [1, 2, 3])

    def test_prefix_sensitive(self) -> None:
        """同块内容、不同父哈希 -> 不同哈希（前缀不同就不可共享）。"""
        assert block_hash(1, [7, 8]) != block_hash(2, [7, 8])

    def test_content_sensitive(self) -> None:
        assert block_hash(ROOT_HASH, [1, 2, 3]) != block_hash(ROOT_HASH, [1, 2, 4])
        assert block_hash(ROOT_HASH, [1, 2]) != block_hash(ROOT_HASH, [1, 2, 3])


# --------------------------------------------------------------------------- #
# 快测 2：PrefixCache 注册 / 命中 / 引用计数 / 驱逐
# --------------------------------------------------------------------------- #


def _pool_with_table(num_blocks: int = 8) -> tuple[BlockPool, PrefixCache, BlockTable]:
    pool = BlockPool(_cfg(), block_size=BS, num_blocks=num_blocks)
    cache = PrefixCache(pool, BS)
    table = BlockTable(pool=pool, block_size=BS, prefix_cache=cache)
    return pool, cache, table


class TestPrefixCacheBasics:
    def test_lookup_miss_on_fresh_cache(self) -> None:
        _, cache, _ = _pool_with_table()
        assert cache.lookup([1, 2, 3, 4, 5]) == ([], 0)

    def test_register_then_lookup_hit(self) -> None:
        """注册过的完整块，同 token 前缀再次 lookup 必须命中同一物理块。"""
        _, cache, table = _pool_with_table()
        k, v = _rand_kv(2 * BS + 2)  # 2 个完整块 + 半块尾巴
        table.append(k, v)
        tokens = list(range(2 * BS + 2))
        assert cache.register(tokens, table) == 2

        hit_ids, hit_len = cache.lookup(tokens)
        assert hit_len == 2 * BS
        assert [b.block_id for b in table.blocks[:2]] == hit_ids

    def test_hit_stops_at_first_miss(self) -> None:
        """前缀只有第一块相同 -> 只命中一块（链式哈希从断点处失效）。"""
        _, cache, table = _pool_with_table()
        k, v = _rand_kv(3 * BS)
        table.append(k, v)
        tokens = list(range(3 * BS))
        cache.register(tokens, table)

        other = tokens[:BS] + [99] * (2 * BS)  # 第二块起不同
        hit_ids, hit_len = cache.lookup(other)
        assert hit_len == BS
        assert len(hit_ids) == 1

    def test_never_hands_back_entire_prompt(self) -> None:
        """prompt 恰好 N 个完整块且全命中：必须放弃最后一块留待现算
        （下一 token 的 logits 只能由 forward 产出）。"""
        _, cache, table = _pool_with_table()
        k, v = _rand_kv(2 * BS)
        table.append(k, v)
        tokens = list(range(2 * BS))
        cache.register(tokens, table)

        hit_ids, hit_len = cache.lookup(tokens)
        assert hit_len == BS  # 2 块命中 -> 只收养 1 块
        assert len(hit_ids) == 1

    def test_short_prompt_no_hit(self) -> None:
        """不足一个完整块的 prompt 无可缓存（docs/02 §11：只缓存完整块）。"""
        _, cache, table = _pool_with_table()
        k, v = _rand_kv(BS - 1)
        table.append(k, v)
        assert cache.register(list(range(BS - 1)), table) == 0
        assert cache.lookup(list(range(BS - 1))) == ([], 0)

    def test_duplicate_register_wins_first(self) -> None:
        """两个表写出同内容块：先注册者进表，后者的块不被追踪。"""
        pool, cache, t1 = _pool_with_table()
        t2 = BlockTable(pool=pool, block_size=BS, prefix_cache=cache)
        tokens = list(range(BS))
        for t in (t1, t2):
            k, v = _rand_kv(BS)
            t.append(k, v)
        assert cache.register(tokens, t1) == 1
        assert cache.register(tokens, t2) == 0
        # t2 的块未追踪 -> 释放时直接还池而不是进 LRU；t1 仍持有（ref=1）
        t2.free()
        assert pool.num_free == pool.num_total - 1
        assert cache.num_tracked_blocks == 1
        assert cache.num_evictable_blocks == 0


class TestRefCountAndEviction:
    def test_release_to_lru_then_reacquire(self) -> None:
        pool, cache, table = _pool_with_table()
        k, v = _rand_kv(2 * BS)
        table.append(k, v)
        cache.register(list(range(2 * BS)), table)

        table.free()  # ref 1 -> 0：不还池，进 LRU
        assert pool.num_free == pool.num_total - 2
        assert cache.num_evictable_blocks == 2

        # 2 块全命中也要放弃最后一块（留 token 现算）-> 收养 1 块
        hit_ids, hit_len = cache.lookup(list(range(2 * BS)))
        assert hit_len == BS
        assert cache.num_evictable_blocks == 1  # 收养后移出 LRU，不可驱逐

    def test_release_untracked_returns_to_pool(self) -> None:
        pool, cache, table = _pool_with_table()
        k, v = _rand_kv(BS)  # 不注册：半块/重复块路径
        table.append(k, v)
        table.free()
        assert pool.num_free == pool.num_total
        assert cache.num_tracked_blocks == 0

    def test_eviction_on_pool_exhaustion_lru_order(self) -> None:
        """池耗尽时按 LRU 驱逐 ref==0 缓存块，被驱逐块从哈希表摘除。"""
        pool, cache, t1 = _pool_with_table(num_blocks=2)
        # 块 A：先释放（更旧）；块 B：后释放（更新）
        ka, va = _rand_kv(BS)
        t1.append(ka, va)
        cache.register(list(range(BS)), t1)
        bid_a = t1.blocks[0].block_id
        t1.free()

        t2 = BlockTable(pool=pool, block_size=BS, prefix_cache=cache)
        kb, vb = _rand_kv(BS)
        t2.append(kb, vb)
        cache.register([10] * BS, t2)
        bid_b = t2.blocks[0].block_id
        t2.free()

        assert pool.num_free == 0 and cache.num_evictable_blocks == 2

        got = cache.alloc_new()  # 池空 -> 驱逐最旧的 A
        assert got == bid_a
        assert cache.lookup(list(range(BS))) == ([], 0)  # A 已从哈希表摘除

        got2 = cache.alloc_new()
        assert got2 == bid_b
        with pytest.raises(ValueError):
            cache.alloc_new()  # 无可驱逐 -> fail fast

    def test_eviction_skips_shared_inflight_blocks(self) -> None:
        """仍被在飞请求收养（ref>0）的块不可驱逐：可驱逐的耗尽后抛错。"""
        pool, cache, t1 = _pool_with_table(num_blocks=2)
        t2 = BlockTable(pool=pool, block_size=BS, prefix_cache=cache)
        k1, v1 = _rand_kv(BS)
        t1.append(k1, v1)
        cache.register([1] * BS, t1)
        k2, v2 = _rand_kv(BS)
        t2.append(k2, v2)
        cache.register([2] * BS, t2)
        bid_shared = t1.blocks[0].block_id
        bid_other = t2.blocks[0].block_id
        t1.free()
        t2.free()  # 两块都在 LRU，池 free=0

        hit_ids, hit_len = cache.lookup([1] * BS + [2] * BS)  # 收养 t1 那块
        assert hit_ids == [bid_shared]
        assert hit_len == BS

        got = cache.alloc_new()  # 先驱逐唯一可驱逐的
        assert got == bid_other
        with pytest.raises(ValueError):
            cache.alloc_new()  # 只剩被收养的块（ref=1）-> 不可驱逐 -> fail fast
        assert cache.num_tracked_blocks == 1


class TestAdoptCorrectness:
    def test_adopt_then_append_gather_bitwise(self) -> None:
        """收养 + 后缀续写后，gather 读回的历史必须逐位等于"从零写一遍"。

        FakeLM 不读 KV，引擎级对比测不出收养块的内容错误，这条是
        "复用不改变数值"的直接证据。
        """
        pool = BlockPool(_cfg(), block_size=BS, num_blocks=8)
        cache = PrefixCache(pool, BS)

        # 参照表：从零写 3 块完整 KV + 2 个新 token
        ref = BlockTable(pool=pool, block_size=BS)
        k_all, v_all = _rand_kv(3 * BS + 2)
        ref.append(k_all, v_all)

        # 被测表：先写相同内容的前 3 块并注册，lookup（放弃最后一块）后
        # 收养 2 块，再从 hit_len 续写剩余 6 个 token
        hit_table = BlockTable(pool=pool, block_size=BS, prefix_cache=cache)
        hit_table.append(k_all[:, :3 * BS], v_all[:, :3 * BS])
        cache.register(list(range(3 * BS)), hit_table)
        hit_table.free()
        hit_ids, hit_len = cache.lookup(list(range(3 * BS)))
        # 3 块全匹配也要放弃最后一块 -> 收养 2 块（物理块号与 ref 无关，内容一致即可）
        assert len(hit_ids) == 2 and hit_len == 2 * BS
        hit_table = BlockTable(pool=pool, block_size=BS, prefix_cache=cache)
        hit_table.adopt(hit_ids, hit_len)
        for layer in range(NUM_LAYERS):
            for t in range(3 * BS + 2 - hit_len):
                pos = hit_len + t
                hit_table.write_token(layer, pos, k_all[layer, pos], v_all[layer, pos])

        for layer in range(NUM_LAYERS):
            assert torch.equal(hit_table.gather(layer, 3 * BS + 2), ref.gather(layer, 3 * BS + 2))
            assert torch.equal(
                hit_table.gather_v(layer, 3 * BS + 2), ref.gather_v(layer, 3 * BS + 2)
            )

    def test_adopt_rejects_bad_args(self) -> None:
        _, cache, table = _pool_with_table()
        with pytest.raises(ValueError):
            table.adopt([0, 1], BS)  # 数量与 token 数不一致
        with pytest.raises(ValueError):
            table.adopt([0], 3)  # 不是完整块
        k, v = _rand_kv(BS)
        table.append(k, v)
        with pytest.raises(ValueError):
            table.adopt([0], BS)  # 非空表

    def test_stress_random_prefixes_no_leak(self) -> None:
        """随机共享前缀 + 随机长度请求反复建表/释放，结束时池账目平衡。"""
        pool = BlockPool(_cfg(), block_size=BS, num_blocks=16)
        cache = PrefixCache(pool, BS)
        rng = torch.Generator().manual_seed(42)
        base = list(range(BS * 3))  # 3 块的公共前缀
        for _ in range(300):
            table = BlockTable(pool=pool, block_size=BS, prefix_cache=cache)
            n = int(torch.randint(BS, 6 * BS, (1,), generator=rng))
            tokens = (base + [int(torch.randint(0, 100, (1,), generator=rng)) for _ in range(n)])
            k, v = _rand_kv(len(tokens))
            table.append(k, v)
            cache.register(tokens, table)
            # 一半请求模拟"另一个请求命中"
            hit_ids, hit_len = cache.lookup(tokens)
            if hit_ids:
                t2 = BlockTable(pool=pool, block_size=BS, prefix_cache=cache)
                t2.adopt(hit_ids, hit_len)
                k2, v2 = _rand_kv(2 * BS)
                for layer in range(NUM_LAYERS):
                    for t in range(2 * BS):
                        t2.write_token(layer, hit_len + t, k2[layer, t], v2[layer, t])
                t2.free()
            table.free()
        # 全部释放后：空闲块 + 可驱逐缓存块 == 总块数（零泄漏）
        assert pool.num_free + cache.num_evictable_blocks == pool.num_total
        assert pool.num_free == pool.num_total - cache.num_evictable_blocks


# --------------------------------------------------------------------------- #
# 快测 3：引擎集成（FakeLM）
# --------------------------------------------------------------------------- #

SHARED_PROMPT = "0123456789012345678901234567890123456789"  # 40 tokens -> 2 完整块


class TestEnginePrefixCache:
    def test_second_request_hits_and_output_identical(self) -> None:
        core = make_core(max_new_tokens=4, enable_prefix_cache=True)
        rid1 = core.submit(SHARED_PROMPT)
        outs = core.run()
        assert outs[rid1].prefix_hit_tokens == 0  # 第一个请求必然全算

        # 同一个引擎再提交一次同前缀请求（跨引擎没有共享缓存，命中必须发生在本引擎内）
        rid2 = core.submit(SHARED_PROMPT)
        outs = core.run()
        assert outs[rid2].prefix_hit_tokens == 32  # 2 个完整块命中
        assert outs[rid2].text == outs[rid1].text

    def test_disabled_zero_hit_and_pool_reclaimed(self) -> None:
        """默认关闭：零命中、结束后块全回收（Task 08/09/10 语义不变）。"""
        core = make_core(max_new_tokens=4, enable_prefix_cache=False)
        rid1 = core.submit(SHARED_PROMPT)
        core.submit(SHARED_PROMPT)
        outs = core.run()
        assert all(o.prefix_hit_tokens == 0 for o in outs.values())
        assert core.runner.paged.num_blocks_used == 0
        assert core.runner.prefix is None

        snap = core.metrics_snapshot()
        assert snap["prefix_cached_blocks"] is None
        assert snap["prefix_hit_tokens_total"] == 0

    def test_concurrent_same_prefix_same_batch(self) -> None:
        """两请求同批提交：先 prefill 的注册后，后者在同一 run 内命中。"""
        core = make_core(max_new_tokens=4, enable_prefix_cache=True, max_num_seqs=4)
        rid1 = core.submit(SHARED_PROMPT)
        rid2 = core.submit(SHARED_PROMPT)
        outs = core.run()
        assert outs[rid1].prefix_hit_tokens == 0
        assert outs[rid2].prefix_hit_tokens == 32
        assert outs[rid1].text == outs[rid2].text

    def test_decode_blocks_registered_for_second_turn(self) -> None:
        """多轮对话：第一轮生成的 token 写满块后注册，第二轮（同一引擎）命中更长前缀。"""
        core = make_core(max_new_tokens=16, enable_prefix_cache=True, block_size=16)
        rid1 = core.submit(SHARED_PROMPT)
        outs1 = core.run()
        first_text = outs1[rid1].text
        assert outs1[rid1].output_tokens == 16

        # 第二轮 prompt = 第一轮完整输入（prompt 40 + 生成 16 = 56 token）
        # -> 3 个完整块（48 token）可命中；第 3 块是 decode 期写满后注册的
        rid2 = core.submit(SHARED_PROMPT + first_text)
        outs2 = core.run()
        assert outs2[rid2].prefix_hit_tokens == 48
        assert outs2[rid2].prefix_hit_tokens > outs1[rid1].prefix_hit_tokens

    def test_metrics_snapshot_reports_prefix(self) -> None:
        core = make_core(max_new_tokens=4, enable_prefix_cache=True)
        core.submit(SHARED_PROMPT)
        core.submit(SHARED_PROMPT)
        core.run()
        snap = core.metrics_snapshot()
        assert snap["prefix_cached_blocks"] == 2  # 两块留在 LRU 等复用
        assert snap["prefix_hit_tokens_total"] == 32

    def test_partial_last_block_not_overclaimed(self) -> None:
        """40 token prompt（2.5 块）：命中上界是 2 块=32，不是 40。"""
        core = make_core(max_new_tokens=2, enable_prefix_cache=True)
        core.submit(SHARED_PROMPT)
        core.run()
        rid = core.submit(SHARED_PROMPT)
        assert core.run()[rid].prefix_hit_tokens == 32


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_default_off(self) -> None:
        assert EngineConfig().enable_prefix_cache is False

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LITEINFER_PREFIX_CACHE", "1")
        assert EngineConfig.from_env().enable_prefix_cache is True
        monkeypatch.setenv("LITEINFER_PREFIX_CACHE", "0")
        assert EngineConfig.from_env().enable_prefix_cache is False


# --------------------------------------------------------------------------- #
# 真模型（marker=model）
# --------------------------------------------------------------------------- #

REAL_PROMPT = (
    "你是一个乐于助人的AI助手。请始终用简洁的中文回答用户的问题，"
    "并在回答末尾附上一句总结。下面是用户的问题："
)


@pytest.fixture(scope="module")
def _real_model():
    """只加载一次权重，两个引擎（开/关前缀缓存）共享同一模型实例，
    避免 FP32 的 0.5B 双份加载把内存撑爆（Task 09 的教训）。"""
    from liteinfer.model.eos import resolve_eos_ids
    from liteinfer.model.minimal.weights import load_minimal_from_hf

    cfg = EngineConfig(device="cpu", dtype=torch.float32, max_new_tokens=12)
    loaded = load_minimal_from_hf(cfg)
    eos = resolve_eos_ids(loaded.hf_model, loaded.tokenizer)
    core_on = EngineCore(
        loaded.minimal,
        loaded.tokenizer,
        EngineConfig(
            device="cpu", dtype=torch.float32, max_new_tokens=12, enable_prefix_cache=True
        ),
        eos_ids=eos,
    )
    core_off = EngineCore(loaded.minimal, loaded.tokenizer, cfg, eos_ids=eos)
    return core_on, core_off


@pytest.mark.model
class TestRealModelPrefixCache:
    def test_hit_and_word_for_word_parity(self, _real_model) -> None:
        """共享前缀第二请求命中，且文本与关闭前缀缓存的引擎逐字一致。"""
        core_on, core_off = _real_model
        params = SamplingParams(max_tokens=12, temperature=0.0)

        rid1 = core_on.submit(REAL_PROMPT, params)
        outs1 = core_on.run()
        assert outs1[rid1].prefix_hit_tokens == 0

        rid2 = core_on.submit(REAL_PROMPT, params)  # 同引擎再来一次同前缀请求
        outs2 = core_on.run()
        assert outs2[rid2].prefix_hit_tokens > 0, "第二个同前缀请求应命中缓存"
        assert outs2[rid2].text == outs1[rid1].text, "命中复用后输出必须逐字一致"

        rid3 = core_off.submit(REAL_PROMPT, params)  # 对照组：关闭前缀缓存
        outs3 = core_off.run()
        assert outs3[rid3].prefix_hit_tokens == 0
        assert outs3[rid3].text == outs1[rid1].text

        # 第二次 prefill 只算后缀，prefill 耗时应显著下降（CPU 上 >2 倍余量）
        assert outs2[rid2].prefill_latency_s < outs1[rid1].prefill_latency_s
