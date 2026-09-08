"""Task 05 测试：Request + Engine Core。

分两层：

1. 快测（默认 `pytest -q`，不下载模型）：
   - Request / RequestStatus / RequestRegistry 纯逻辑；
   - 用「假模型」驱动 EngineCore 的状态机，验证单/多请求、EOS/length 终止、
     中途 submit、cancel 的并发维护正确性。假模型是 token 的纯函数，保证两个
     请求互不干扰、顺序无关。

2. 真模型测试（marker=model，`pytest -m model`）：
   - 单请求 EngineCore 输出与 CachedGenerator **逐字一致**（text / finish_reason
     / output_tokens），证明引擎只是把 CachedGenerator 的单请求循环升级成了
     多请求状态机，语义没变；
   - 2 个并发请求各自与各自 CachedGenerator 输出一致。
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from liteinfer import EngineConfig
from liteinfer.engine import (
    EngineCore,
    Request,
    RequestRegistry,
    RequestState,
    RequestStatus,
)
from liteinfer.engine.request import RequestStepResult
from liteinfer.sampling.params import SamplingParams


# --------------------------------------------------------------------------- #
# 假模型 / 假 tokenizer：不下载任何权重，token id 是输入最后一位的确定性函数
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
    """forward 忽略 KV 缓存，按输入最后一位产出下一个 token：nxt = (last+1) % vocab。

    之所以用"输入决定输出"而非全局计数器：引擎持有**单个**模型，多请求共享，
    若用计数器两个请求会串台；用纯函数则每个请求只取决于自己的 token 序列，
    顺序无关，能真正验证"并发维护"。
    """

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


def _simulate(start: int, max_tokens: int, eos: int | None, vocab: int = 10) -> list[int]:
    """复刻引擎在 FakeLM 下的应有产出，供断言对照。"""
    seq: list[int] = []
    cur = start
    for _ in range(max_tokens):
        nxt = (cur + 1) % vocab
        if eos is not None and nxt == eos:
            break
        seq.append(nxt)
        cur = nxt
    return seq


def _make_engine(max_new_tokens: int = 4, eos: int | None = None):
    cfg = EngineConfig(device="cpu", dtype=torch.float32, max_new_tokens=max_new_tokens)
    eos_set = frozenset([eos]) if eos is not None else frozenset()
    return EngineCore(FakeLM(), FakeTokenizer(), cfg, eos_ids=eos_set)


# --------------------------------------------------------------------------- #
# 快测：Request / RequestStatus / RequestRegistry 纯逻辑
# --------------------------------------------------------------------------- #


class TestRequestModel:
    def test_request_defaults(self) -> None:
        p = SamplingParams(max_tokens=4)
        req = Request(request_id="r1", prompt="2", params=p)
        assert req.status == RequestStatus.WAITING
        assert req.prompt_tokens == 0
        assert req.output_tokens == 0
        assert not req.is_terminal

    def test_request_status_terminal(self) -> None:
        assert RequestStatus.FINISHED.is_terminal
        assert RequestStatus.CANCELLED.is_terminal
        assert not RequestStatus.DECODE.is_terminal

    def test_registry_add_get_contains(self) -> None:
        reg = RequestRegistry()
        p = SamplingParams(max_tokens=4)
        reg.add(Request("a", "2", p))
        assert "a" in reg
        assert reg.get("a").request_id == "a"
        assert len(reg) == 1

    def test_registry_duplicate_raises(self) -> None:
        reg = RequestRegistry()
        p = SamplingParams(max_tokens=4)
        reg.add(Request("a", "2", p))
        with pytest.raises(ValueError):
            reg.add(Request("a", "2", p))

    def test_registry_active_excludes_terminal(self) -> None:
        reg = RequestRegistry()
        p = SamplingParams(max_tokens=4)
        reg.add(Request("a", "2", p))
        reg.add(Request("b", "3", p, status=RequestStatus.FINISHED))
        assert [r.request_id for r in reg.active()] == ["a"]

    def test_registry_count_by_status(self) -> None:
        reg = RequestRegistry()
        p = SamplingParams(max_tokens=4)
        reg.add(Request("a", "2", p, status=RequestStatus.WAITING))
        reg.add(Request("b", "3", p, status=RequestStatus.DECODE))
        reg.add(Request("c", "4", p, status=RequestStatus.FINISHED))
        counts = reg.count_by_status()
        assert counts[RequestStatus.WAITING] == 1
        assert counts[RequestStatus.DECODE] == 1
        assert counts[RequestStatus.FINISHED] == 1


# --------------------------------------------------------------------------- #
# 快测：EngineCore 状态机（假模型，无下载）
# --------------------------------------------------------------------------- #


class TestEngineFakeModel:
    def test_single_request_length_finish(self) -> None:
        engine = _make_engine(max_new_tokens=4)
        rid = engine.submit("2")
        out = engine.run()[rid]
        assert out.finish_reason == "length"
        assert out.output_tokens == 4
        assert engine.get_request(rid).status == RequestStatus.FINISHED
        assert engine.get_request(rid).generated == _simulate(2, 4, None)
        assert engine.active_requests() == []

    def test_single_request_eos_finish(self) -> None:
        engine = _make_engine(max_new_tokens=10, eos=5)
        rid = engine.submit("2")
        out = engine.run()[rid]
        assert out.finish_reason == "eos"
        # 2 -> 3,4,5(命中 EOS 不产出) => 仅 [3,4]
        assert engine.get_request(rid).generated == [3, 4]
        assert engine.active_requests() == []

    def test_multiple_requests_interleaved(self) -> None:
        engine = _make_engine(max_new_tokens=4)
        ra = engine.submit("2")
        rb = engine.submit("7")
        outputs = engine.run()
        assert engine.active_requests() == []
        assert engine.get_request(ra).generated == _simulate(2, 4, None)
        assert engine.get_request(rb).generated == _simulate(7, 4, None)
        assert outputs[ra].finish_reason == "length"
        assert outputs[rb].finish_reason == "length"

    def test_mid_flight_submit(self) -> None:
        engine = _make_engine(max_new_tokens=4)
        ra = engine.submit("2")
        # A 先走一步（prefill + 产出首 token），此时仍在 DECODE
        engine.step()
        assert engine.get_request(ra).status == RequestStatus.DECODE
        assert engine.active_requests()  # A 还在
        rb = engine.submit("7")
        outputs = engine.run()
        # 两个请求都正确结束，且互不影响
        assert engine.get_request(ra).generated == _simulate(2, 4, None)
        assert engine.get_request(rb).generated == _simulate(7, 4, None)
        assert engine.active_requests() == []

    def test_cancel(self) -> None:
        engine = _make_engine(max_new_tokens=4)
        rid = engine.submit("2")
        engine.step()  # prefill，产出 3
        assert engine.get_request(rid).generated == [3]
        engine.cancel(rid)
        req = engine.get_request(rid)
        assert req.status == RequestStatus.CANCELLED
        assert req.finish_reason == "cancelled"
        assert engine.active_requests() == []
        out = engine.run()[rid]
        assert out.finish_reason == "cancelled"
        assert out.text == "3"

    def test_step_results_shape(self) -> None:
        engine = _make_engine(max_new_tokens=2)
        rid = engine.submit("2")
        results = engine.step()  # prefill -> 1 个逐步结果
        assert len(results) == 1
        r0 = results[0]
        assert isinstance(r0, RequestStepResult)
        assert r0.request_id == rid
        assert r0.token_id == 3  # (2+1)%10
        assert not r0.finished


# --------------------------------------------------------------------------- #
# 真模型测试：与 CachedGenerator 逐字一致（需要 Qwen2.5-0.5B）
# --------------------------------------------------------------------------- #


PROMPT_A = "The capital of France is"
PROMPT_B = "The largest planet in our solar system is"
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
class TestEngineParityWithCachedGenerator:
    def test_single_request_matches_cached_generator(self, loaded, cfg, eos) -> None:
        params = SamplingParams(max_tokens=N_TOKENS, temperature=0.0)
        engine = EngineCore(loaded.minimal, loaded.tokenizer, cfg, eos_ids=eos)
        rid = engine.submit(PROMPT_A, params)
        out = engine.run()[rid]

        cached = _cached_out(loaded, cfg, eos, PROMPT_A, params)
        assert out.text == cached.text
        assert out.finish_reason == cached.finish_reason
        assert out.output_tokens == cached.output_tokens
        # 缓存记账也应一致：prompt + 输出 - 1（length 情形）
        assert out.cached_tokens == cached.cached_tokens

    def test_multi_request_each_matches_cached_generator(self, loaded, cfg, eos) -> None:
        params = SamplingParams(max_tokens=N_TOKENS, temperature=0.0)
        engine = EngineCore(loaded.minimal, loaded.tokenizer, cfg, eos_ids=eos)
        ra = engine.submit(PROMPT_A, params)
        rb = engine.submit(PROMPT_B, params)
        outputs = engine.run()

        for rid, prompt in ((ra, PROMPT_A), (rb, PROMPT_B)):
            cached = _cached_out(loaded, cfg, eos, prompt, params)
            assert outputs[rid].text == cached.text
            assert outputs[rid].finish_reason == cached.finish_reason
            assert outputs[rid].output_tokens == cached.output_tokens
        # 两个请求都正确结束，证明引擎同时维护了多个请求
        assert engine.active_requests() == []
