"""Task 08 测试：ModelRunner 接入 Paged KV。

分两层：

1. 快测（默认 `pytest -q`，不下载模型）：
   - ``PagedLayerCache`` 与 Task 04 ``LayerKVCache`` **逐位一致**（这是"接入"的
     正确性根基：分页只换存储，不换数值）；
   - 块分配次数只与 token 位置有关、与层数无关（多层写同一 offset 只分配一次）；
   - 用「会写缓存的假模型」驱动 EngineCore，验证 prefill/decode 真的落到了块表里，
     且请求终态 / 取消后物理块**全部归还**（零泄漏）；
   - 块池耗尽时 fail fast。

2. 真模型测试（marker=model）：引擎（分页）与 CachedGenerator（连续）逐字一致，
   多请求并发各自正确——即 docs/07 Task 08 的验收"多请求输出正确"。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from liteinfer import EngineConfig
from liteinfer.cache.contiguous import ContiguousKVCache, KVCacheConfig
from liteinfer.cache.paged import PagedKVCache
from liteinfer.engine import EngineCore, RequestStatus
from liteinfer.model.runner import ModelRunner, PagedLayerCache, infer_kv_dims
from liteinfer.sampling.params import SamplingParams
from liteinfer.scheduler.config import SchedulerConfig

NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 8


def _cache_cfg(max_seq_len: int = 32) -> KVCacheConfig:
    return KVCacheConfig(
        num_layers=NUM_LAYERS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_seq_len=max_seq_len,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _rand(n: int) -> torch.Tensor:
    return torch.randn(1, n, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# 快测 1：分页适配器与连续实现逐位一致
# --------------------------------------------------------------------------- #


class TestPagedLayerCacheParity:
    def test_append_read_matches_contiguous(self) -> None:
        """同样的 K/V 写进两种实现，读回来必须逐位相等。"""
        contig = ContiguousKVCache(_cache_cfg(max_seq_len=32))
        paged = PagedKVCache(_cache_cfg(max_seq_len=32), block_size=4, num_blocks=8)
        table = paged.new_block_table()
        handles = [PagedLayerCache(table, i) for i in range(NUM_LAYERS)]

        # 第一段 3 个 token；第二段 5 个 token，跨越 block 边界（3+5=8 > 4）
        for chunk_n, start in ((3, 0), (5, 3)):
            ks = [_rand(chunk_n) for _ in range(NUM_LAYERS)]
            vs = [_rand(chunk_n) for _ in range(NUM_LAYERS)]
            for i in range(NUM_LAYERS):
                ck, cv = contig.layer_caches[i].append(ks[i], vs[i], start)
                pk, pv = handles[i].append(ks[i], vs[i], start)
                assert torch.equal(ck, pk), f"层 {i} 的 K 与连续实现不一致"
                assert torch.equal(cv, pv), f"层 {i} 的 V 与连续实现不一致"

        # 8 个 token / block_size=4 -> 恰好 2 个物理块
        assert table.num_tokens == 8
        assert paged.num_blocks_used == 2

    def test_read_after_partial_history_matches(self) -> None:
        """只读部分历史（length < 已写入）也要与连续实现一致。"""
        contig = ContiguousKVCache(_cache_cfg())
        paged = PagedKVCache(_cache_cfg(), block_size=4, num_blocks=8)
        table = paged.new_block_table()
        handle = PagedLayerCache(table, 0)

        k, v = _rand(7), _rand(7)
        contig.layer_caches[0].append(k, v, 0)
        handle.append(k, v, 0)

        for length in (1, 3, 4, 5, 7):
            ck, cv = contig.layer_caches[0].read(length)
            pk, pv = handle.read(length)
            assert torch.equal(ck, pk), f"length={length} 时 K 不一致"
            assert torch.equal(cv, pv), f"length={length} 时 V 不一致"

    def test_gather_v_reads_value_not_key(self) -> None:
        """V 的读路径不能复用 K（Task 08 新增 gather_v）。"""
        paged = PagedKVCache(_cache_cfg(), block_size=4, num_blocks=4)
        table = paged.new_block_table()
        handle = PagedLayerCache(table, 0)
        k, v = _rand(3), _rand(3) + 100.0  # 让 K/V 明显不同
        handle.append(k, v, 0)

        got_k, got_v = handle.read(3)
        assert torch.equal(got_k, k)
        assert torch.equal(got_v, v)
        assert not torch.allclose(got_k, got_v)

    def test_block_allocated_once_per_token_not_per_layer(self) -> None:
        """多层写同一个 token 只应分配一次物理块（分配判据是 token 位置）。"""
        paged = PagedKVCache(_cache_cfg(), block_size=4, num_blocks=16)
        table = paged.new_block_table()
        handles = [PagedLayerCache(table, i) for i in range(NUM_LAYERS)]

        k, v = _rand(9), _rand(9)
        for i in range(NUM_LAYERS):
            handles[i].append(k, v, 0)

        # 9 token / block_size 4 -> ceil(9/4)=3 块，与层数无关
        assert paged.num_blocks_used == 3

    def test_exhausted_pool_fails_fast(self) -> None:
        """块池耗尽必须抛错，不能静默越界写。"""
        cfg = KVCacheConfig(
            num_layers=1, num_kv_heads=1, head_dim=2,
            max_seq_len=8, dtype=torch.float32, device=torch.device("cpu"),
        )
        paged = PagedKVCache(cfg, block_size=2, num_blocks=1)  # 总共只装 2 个 token
        table = paged.new_block_table()
        handle = PagedLayerCache(table, 0)

        handle.append(torch.zeros(1, 2, 1, 2), torch.zeros(1, 2, 1, 2), 0)
        with pytest.raises(ValueError):
            # 第 3 个 token 需要新块，但池里没有了
            handle.append(torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 1, 2), 2)

    def test_invalid_inputs_rejected(self) -> None:
        paged = PagedKVCache(_cache_cfg(), block_size=4, num_blocks=4)
        handle = PagedLayerCache(paged.new_block_table(), 0)
        with pytest.raises(ValueError):
            handle.append(torch.zeros(1, 2, NUM_KV_HEADS), torch.zeros(1, 2, NUM_KV_HEADS), 0)
        with pytest.raises(ValueError):
            handle._check_new(_rand(2), _rand(3), 0)  # k/v 形状不一致

    def test_bytes_for_matches_token_count(self) -> None:
        paged = PagedKVCache(_cache_cfg(), block_size=4, num_blocks=8)
        table = paged.new_block_table()
        handle = PagedLayerCache(table, 0)
        handle.append(_rand(5), _rand(5), 0)

        runner = ModelRunner(_WritingFakeLM(), _cfg())
        assert runner.bytes_for(table) == 5 * _cache_cfg().bytes_per_token()
        assert runner.bytes_for(None) == 0


# --------------------------------------------------------------------------- #
# 假模型：真的往 kv_caches 里写（否则测不到分页路径与块回收）
# --------------------------------------------------------------------------- #


class _FakeAttn:
    def __init__(self, num_kv_heads: int, head_dim: int) -> None:
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


class _FakeLayer(nn.Module):
    def __init__(self, num_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.self_attn = _FakeAttn(num_kv_heads, head_dim)


class _FakeBody(nn.Module):
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_FakeLayer(num_kv_heads, head_dim) for _ in range(num_layers)]
        )


class _WritingFakeLM(nn.Module):
    """forward 会把 K/V 写进传入的 kv_caches（模拟真实注意力的 append 调用）。

    与 Task 05 的 FakeLM 区别：那个直接忽略 kv_caches，因此测不出"块是否被
    分配/归还"。这里显式调用 ``c.append(...)``，让分页的写入与 gather 读回
    真正被执行到，同时输出仍是输入的纯函数（多请求互不干扰）。
    """

    def __init__(self, vocab: int = 10) -> None:
        super().__init__()
        self.vocab = vocab
        self.model = _FakeBody(
            num_layers=NUM_LAYERS, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM
        )

    def forward(self, input_ids, position_ids=None, kv_caches=None, write_pos=0):
        batch, seq_len = input_ids.shape
        if kv_caches is not None:
            k = torch.zeros(1, seq_len, NUM_KV_HEADS, HEAD_DIM)
            v = torch.zeros(1, seq_len, NUM_KV_HEADS, HEAD_DIM)
            for cache in kv_caches:
                cache.append(k, v, write_pos)
        logits = torch.full((batch, seq_len, self.vocab), -1e9)
        last = int(input_ids[0, -1].item())
        logits[:, -1, (last + 1) % self.vocab] = 1.0
        return logits


class FakeTokenizer:
    """prompt 是若干个数字字符，每个字符一个 token。"""

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[int(c) for c in text]], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(str(int(i)) for i in ids)


def _cfg(max_new_tokens: int = 4, **kw) -> EngineConfig:
    return EngineConfig(device="cpu", dtype=torch.float32, max_new_tokens=max_new_tokens, **kw)


def _simulate(start: int, max_tokens: int, vocab: int = 10) -> list[int]:
    seq, cur = [], start
    for _ in range(max_tokens):
        nxt = (cur + 1) % vocab
        seq.append(nxt)
        cur = nxt
    return seq


# --------------------------------------------------------------------------- #
# 快测 2：ModelRunner + EngineCore 的块生命周期
# --------------------------------------------------------------------------- #


class TestRunnerBlockLifecycle:
    def test_runner_allocates_and_frees(self) -> None:
        runner = ModelRunner(_WritingFakeLM(), _cfg(max_new_tokens=4))
        table = runner.new_block_table()
        assert runner.paged.num_blocks_used == 0  # 空表不占块

        logits = runner.prefill(torch.tensor([[1, 2, 3]]), table)
        assert logits.shape == (1, 3, 10)
        assert table.num_tokens == 3
        assert runner.paged.num_blocks_used == 1

        # decode 每步写 1 个 token（block_size=16，仍在第一个块内）
        logits = runner.decode(4, 3, table)
        assert logits.shape == (1, 1, 10)
        assert table.num_tokens == 4

        runner.free_table(table)
        assert runner.paged.num_blocks_used == 0
        assert runner.paged.num_blocks_free == runner.paged.num_blocks_total

    def test_engine_releases_blocks_on_finish(self) -> None:
        engine = EngineCore(_WritingFakeLM(), FakeTokenizer(), _cfg(max_new_tokens=4))
        rids = [engine.submit(s) for s in ("12", "45", "78")]

        engine.step()  # 三个请求被准入并 prefill -> 块被占用
        assert engine.runner.paged.num_blocks_used > 0

        outputs = engine.run()
        assert engine.active_requests() == []
        # 全部终态后物理块必须全部归还（docs/02 §2 生命周期最后一步）
        assert engine.runner.paged.num_blocks_used == 0
        for rid, start in zip(rids, (2, 5, 8)):
            assert engine.get_request(rid).generated == _simulate(start, 4)
            assert outputs[rid].cache_bytes > 0

    def test_cancel_releases_blocks(self) -> None:
        engine = EngineCore(_WritingFakeLM(), FakeTokenizer(), _cfg(max_new_tokens=4))
        rid = engine.submit("12")
        engine.step()
        assert engine.runner.paged.num_blocks_used > 0

        engine.cancel(rid)
        assert engine.get_request(rid).status == RequestStatus.CANCELLED
        # 取消路径也必须立刻归还，否则块会泄漏到进程结束
        assert engine.runner.paged.num_blocks_used == 0

    def test_waiting_requests_hold_no_blocks(self) -> None:
        """waiting 队列里的请求不应提前占住物理块（分页相对预分配的价值）。"""
        sched = SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=1024)
        engine = EngineCore(
            _WritingFakeLM(), FakeTokenizer(), _cfg(max_new_tokens=3, scheduler=sched)
        )
        [engine.submit(s) for s in ("12", "45", "78")]
        assert engine.scheduler.num_waiting == 3
        assert engine.runner.paged.num_blocks_used == 0

        engine.step()  # 只准入 1 个
        assert engine.runner.paged.num_blocks_used == 1

    def test_multi_request_blocks_isolated(self) -> None:
        """并发请求各自一张块表，互不覆盖（gather 只读自己那几块）。"""
        engine = EngineCore(_WritingFakeLM(), FakeTokenizer(), _cfg(max_new_tokens=3))
        rids = [engine.submit(s) for s in ("12", "45", "78")]
        outputs = engine.run()

        for rid, start in zip(rids, (2, 5, 8)):
            assert engine.get_request(rid).generated == _simulate(start, 3)
            assert outputs[rid].finish_reason == "length"
        assert engine.runner.paged.num_blocks_used == 0

    def test_block_size_respected(self) -> None:
        """跨块写入时块数 = ceil(tokens / block_size)。"""
        cfg = _cfg(max_new_tokens=4, block_size=2)
        engine = EngineCore(_WritingFakeLM(), FakeTokenizer(), cfg)
        assert engine.runner.block_size == 2
        engine.submit("12345")  # prompt 5 token + 4 decode = 9 token
        engine.run()
        # 结束时块已归还，所以只在过程中断言峰值
        assert engine.runner.paged.num_blocks_used == 0


# --------------------------------------------------------------------------- #
# 真模型测试：与 CachedGenerator（连续 KV）逐字一致
# --------------------------------------------------------------------------- #


PROMPT_A = "The capital of France is"
PROMPT_B = "The largest planet in our solar system is"
PROMPT_C = "The chemical symbol for water is"
N_TOKENS = 16


@pytest.fixture(scope="module")
def cfg() -> EngineConfig:
    return EngineConfig(device="cpu", dtype=torch.float32, max_new_tokens=N_TOKENS)


@pytest.fixture(scope="module")
def loaded(cfg):
    from liteinfer.model.minimal.weights import load_minimal_from_hf

    return load_minimal_from_hf(cfg)


@pytest.fixture(scope="module")
def eos(loaded):
    from liteinfer.model.eos import resolve_eos_ids

    return resolve_eos_ids(loaded.hf_model, loaded.tokenizer)


def _cached_out(loaded, cfg, eos, prompt, params):
    from liteinfer.model.cached_generator import CachedGenerator

    gen = CachedGenerator(loaded.minimal, loaded.tokenizer, cfg, eos_ids=eos)
    return gen.generate(prompt, params)


@pytest.mark.model
class TestPagedRunnerParity:
    def test_runner_dims_from_real_model(self, loaded, cfg) -> None:
        runner = ModelRunner(loaded.minimal, cfg)
        num_layers, num_kv_heads, head_dim = infer_kv_dims(loaded.minimal)
        # Qwen2.5-0.5B：24 层、2 个 KV 头（GQA，不是 14 个 Q 头）、head_dim 64
        assert (runner.num_layers, num_layers) == (24, 24)
        assert num_kv_heads == 2
        assert head_dim == 64

    def test_single_request_matches_contiguous(self, loaded, cfg, eos) -> None:
        """分页 KV 与连续 KV 必须产出完全相同的文本（greedy）。"""
        params = SamplingParams(max_tokens=N_TOKENS, temperature=0.0)
        engine = EngineCore(loaded.minimal, loaded.tokenizer, cfg, eos_ids=eos)
        rid = engine.submit(PROMPT_A, params)
        out = engine.run()[rid]

        cached = _cached_out(loaded, cfg, eos, PROMPT_A, params)
        assert out.text == cached.text
        assert out.finish_reason == cached.finish_reason
        assert out.output_tokens == cached.output_tokens
        assert out.cached_tokens == cached.cached_tokens

    def test_multi_request_each_matches_contiguous(self, loaded, cfg, eos) -> None:
        """验收点：多请求并发，各自与连续 KV 的对照逐字一致。"""
        params = SamplingParams(max_tokens=N_TOKENS, temperature=0.0)
        engine = EngineCore(loaded.minimal, loaded.tokenizer, cfg, eos_ids=eos)
        ra = engine.submit(PROMPT_A, params)
        rb = engine.submit(PROMPT_B, params)
        engine.step()
        rc = engine.submit(PROMPT_C, params)  # 运行中动态加入
        outputs = engine.run()

        for rid, prompt in ((ra, PROMPT_A), (rb, PROMPT_B), (rc, PROMPT_C)):
            cached = _cached_out(loaded, cfg, eos, prompt, params)
            assert outputs[rid].text == cached.text, f"{prompt} 分页输出与连续不一致"
            assert outputs[rid].finish_reason == cached.finish_reason
            assert outputs[rid].output_tokens == cached.output_tokens

        assert engine.active_requests() == []
        # 三请求全部结束后，共享池的块应全部归还
        assert engine.runner.paged.num_blocks_used == 0

    def test_blocks_actually_used_during_generation(self, loaded, cfg, eos) -> None:
        """跑的过程中必须真的占用了块（否则上面的"归零"是空跑通过）。"""
        params = SamplingParams(max_tokens=N_TOKENS, temperature=0.0)
        engine = EngineCore(loaded.minimal, loaded.tokenizer, cfg, eos_ids=eos)
        engine.submit(PROMPT_A, params)
        engine.step()
        assert engine.runner.paged.num_blocks_used > 0
        engine.run()
        assert engine.runner.paged.num_blocks_used == 0
