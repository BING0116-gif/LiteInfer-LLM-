"""Task 06 集成测试：Scheduler 接入 EngineCore。

两层：
1. 快测（默认 `pytest -q`，不下载模型）：用 Task 05 的「假模型」（token 的纯函数）
   驱动引擎，验证：
   - 序列预算（max_num_seqs）确实限制并发在飞数；
   - 8+8=16 个请求在运行中动态加入、最终全部正确结束（连续批处理）；
   - 中途 submit 与 cancel 在调度器下依然正确；
   - token budget 过小导致单 prompt 无法准入时 fail fast 抛出清晰错误；
   - 默认（不限额）下结果与 Task 05 无调度器时一致。

2. 真模型测试（marker=model）：在 Scheduler 下跑 Qwen2.5-0.5B，产出文本与
   CachedGenerator 逐字一致（调度只改变"何时算"，不改变"算什么"）。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from liteinfer import EngineConfig
from liteinfer.engine import EngineCore, RequestStatus
from liteinfer.sampling.params import SamplingParams
from liteinfer.scheduler.config import SchedulerConfig


# --------------------------------------------------------------------------- #
# 假模型 / 假 tokenizer（与 Task 05 同形）：不下载任何权重
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


class FakeLM(nn.Module):
    """forward 忽略 KV 缓存，按输入最后一位产出下一个 token：nxt = (last+1) % vocab。"""

    def __init__(self, vocab: int = 10) -> None:
        super().__init__()
        self.vocab = vocab
        self.model = _FakeBody(num_layers=1, num_kv_heads=2, head_dim=8)

    def forward(self, input_ids, position_ids=None, kv_caches=None, write_pos=0):
        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, self.vocab), -1e9)
        last = int(input_ids[0, -1].item())
        nxt = (last + 1) % self.vocab
        logits[:, -1, nxt] = 1.0
        return logits


class FakeTokenizer:
    """prompt 直接当成单个整数 token id；decode 把 id 拼成字符串。"""

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[int(text)]], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(str(int(i)) for i in ids)


class _LongTokenizer:
    """用于验证 token budget fail-fast：把任意 prompt 编码成固定长度的张量。"""

    def __init__(self, length: int) -> None:
        self._length = length

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.zeros((1, self._length), dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(str(int(i)) for i in ids)


def _simulate(start: int, max_tokens: int, eos: int | None, vocab: int = 10) -> list[int]:
    seq: list[int] = []
    cur = start
    for _ in range(max_tokens):
        nxt = (cur + 1) % vocab
        if eos is not None and nxt == eos:
            break
        seq.append(nxt)
        cur = nxt
    return seq


def _make_engine(max_new_tokens: int = 4, scheduler: SchedulerConfig | None = None, eos=None):
    cfg = EngineConfig(
        device="cpu",
        dtype=torch.float32,
        max_new_tokens=max_new_tokens,
        scheduler=scheduler if scheduler is not None else SchedulerConfig(),
    )
    eos_set = frozenset([eos]) if eos is not None else frozenset()
    return EngineCore(FakeLM(), FakeTokenizer(), cfg, eos_ids=eos_set)


# --------------------------------------------------------------------------- #
# 快测：序列预算 / 动态批 / 中途 submit / cancel / fail-fast
# --------------------------------------------------------------------------- #


class TestSchedulerEngineFake:
    def test_seq_budget_limits_concurrency(self) -> None:
        cfg_sched = SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=1024)
        engine = _make_engine(max_new_tokens=4, scheduler=cfg_sched)
        starts = ["2", "4", "6", "8"]
        rids = [engine.submit(s) for s in starts]

        # 提交即进 waiting，尚未准入
        assert engine.scheduler.num_waiting == 4
        assert engine.scheduler.num_running == 0

        max_running = 0
        while engine.active_requests():
            engine.step()
            max_running = max(max_running, engine.scheduler.num_running)

        # 并发在飞数从不突破序列预算
        assert max_running <= 2
        for rid, st in zip(rids, starts):
            assert engine.get_request(rid).generated == _simulate(int(st), 4, None)
        assert engine.active_requests() == []

    def test_16_requests_dynamic_join_leave(self) -> None:
        """8 个先提交，运行中再动态提交 8 个；全部结束后各自正确。

        验收点：8~16 个请求动态加入退出，连续批处理正确维护。
        """
        cfg_sched = SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=1024)
        engine = _make_engine(max_new_tokens=3, scheduler=cfg_sched)

        first_batch = [str(i) for i in range(8)]
        rids = [engine.submit(s) for s in first_batch]

        # 跑几步，让第一批部分推进，此时动态加入第二批
        for _ in range(2):
            engine.step()
        second_batch = [str(i) for i in range(8, 16)]
        rids += [engine.submit(s) for s in second_batch]

        outputs = engine.run()
        assert engine.active_requests() == []
        for rid, start in zip(rids, list(range(16))):
            assert outputs[rid].finish_reason == "length"
            assert engine.get_request(rid).generated == _simulate(start, 3, None)

    def test_mid_flight_submit_with_budget(self) -> None:
        cfg_sched = SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=1024)
        engine = _make_engine(max_new_tokens=4, scheduler=cfg_sched)
        ra = engine.submit("2")
        engine.step()  # A 被准入并 prefill
        assert engine.get_request(ra).status == RequestStatus.DECODE
        rb = engine.submit("7")  # 运行中动态加入
        outputs = engine.run()
        assert engine.get_request(ra).generated == _simulate(2, 4, None)
        assert engine.get_request(rb).generated == _simulate(7, 4, None)
        assert engine.active_requests() == []

    def test_cancel_with_scheduler(self) -> None:
        cfg_sched = SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=1024)
        engine = _make_engine(max_new_tokens=4, scheduler=cfg_sched)
        rid = engine.submit("2")
        engine.step()  # prefill -> 产出 3
        assert engine.get_request(rid).generated == [3]
        engine.cancel(rid)
        req = engine.get_request(rid)
        assert req.status == RequestStatus.CANCELLED
        assert req.finish_reason == "cancelled"
        # 取消后不应再被调度
        out = engine.run()[rid]
        assert out.finish_reason == "cancelled"
        assert out.text == "3"

    def test_huge_prompt_rejected_by_token_budget(self) -> None:
        # token budget 只能容纳 3 个 token，prompt 长 100 -> 提交即失败（fail fast）
        cfg_sched = SchedulerConfig(max_num_seqs=16, max_num_batched_tokens=3)
        cfg = EngineConfig(
            device="cpu", dtype=torch.float32, max_new_tokens=4, scheduler=cfg_sched
        )
        engine = EngineCore(FakeLM(), _LongTokenizer(length=100), cfg, eos_ids=frozenset())
        with pytest.raises(ValueError):
            engine.submit("anything")

    def test_uncapped_matches_task05_behavior(self) -> None:
        """默认（不限额）配置下，结果与 Task 05 同时推进完全一致。"""
        engine = _make_engine(max_new_tokens=5)  # 默认 SchedulerConfig（足够大）
        rids = [engine.submit(s) for s in ("2", "5", "9")]
        outputs = engine.run()
        for rid, start in zip(rids, (2, 5, 9)):
            assert engine.get_request(rid).generated == _simulate(start, 5, None)


# --------------------------------------------------------------------------- #
# 真模型测试：与 CachedGenerator 逐字一致（需要 Qwen2.5-0.5B）
# --------------------------------------------------------------------------- #


PROMPT_A = "The capital of France is"
PROMPT_B = "The largest planet in our solar system is"
PROMPT_C = "The chemical symbol for water is"
N_TOKENS = 16


@pytest.fixture(scope="module")
def loaded(cfg):
    from liteinfer.model.minimal.weights import load_minimal_from_hf

    return load_minimal_from_hf(cfg)


@pytest.fixture(scope="module")
def cfg() -> EngineConfig:
    return EngineConfig(device="cpu", dtype=torch.float32, max_new_tokens=N_TOKENS)


@pytest.fixture(scope="module")
def eos(loaded):
    from liteinfer.model.eos import resolve_eos_ids

    return resolve_eos_ids(loaded.hf_model, loaded.tokenizer)


def _cached_out(loaded, cfg, eos, prompt, params):
    from liteinfer.model.cached_generator import CachedGenerator

    gen = CachedGenerator(loaded.minimal, loaded.tokenizer, cfg, eos_ids=eos)
    return gen.generate(prompt, params)


@pytest.mark.model
class TestSchedulerEngineParity:
    def test_single_request_matches_cached_generator(self, loaded, cfg, eos) -> None:
        params = SamplingParams(max_tokens=N_TOKENS, temperature=0.0)
        # 用受限的序列预算证明：预算不影响生成正确性，只影响并发节奏
        sched = SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=2048)
        sched_cfg = EngineConfig(
            device="cpu", dtype=torch.float32, max_new_tokens=N_TOKENS, scheduler=sched
        )
        engine = EngineCore(loaded.minimal, loaded.tokenizer, sched_cfg, eos_ids=eos)
        rid = engine.submit(PROMPT_A, params)
        out = engine.run()[rid]

        cached = _cached_out(loaded, cfg, eos, PROMPT_A, params)
        assert out.text == cached.text
        assert out.finish_reason == cached.finish_reason
        assert out.output_tokens == cached.output_tokens
        assert out.cached_tokens == cached.cached_tokens

    def test_concurrent_plus_mid_flight_matches(self, loaded, cfg, eos) -> None:
        params = SamplingParams(max_tokens=N_TOKENS, temperature=0.0)
        engine = EngineCore(loaded.minimal, loaded.tokenizer, cfg, eos_ids=eos)
        ra = engine.submit(PROMPT_A, params)
        rb = engine.submit(PROMPT_B, params)
        engine.step()  # 两个都已 prefill 推进
        rc = engine.submit(PROMPT_C, params)  # 运行中动态加入第三个
        outputs = engine.run()

        for rid, prompt in ((ra, PROMPT_A), (rb, PROMPT_B), (rc, PROMPT_C)):
            cached = _cached_out(loaded, cfg, eos, prompt, params)
            assert outputs[rid].text == cached.text
            assert outputs[rid].finish_reason == cached.finish_reason
            assert outputs[rid].output_tokens == cached.output_tokens
        assert engine.active_requests() == []
