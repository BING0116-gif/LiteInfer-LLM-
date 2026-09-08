"""Sampler 纯单测：用手工构造的 logits 验证采样数学性质。

不加载任何模型，秒级完成；这是把 Sampler 做成纯函数的直接收益。
"""

from __future__ import annotations

import pytest
import torch

from liteinfer.sampling.params import SamplingParams
from liteinfer.sampling.sampler import Sampler


@pytest.fixture
def sampler() -> Sampler:
    return Sampler()


def _sample_many(
    sampler: Sampler,
    logits: torch.Tensor,
    params: SamplingParams,
    n: int = 300,
    seed: int = 1234,
) -> set[int]:
    """固定 seed 采 n 次，返回实际出现的 token id 集合。"""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    observed = {
        sampler.sample(logits, params, gen) for _ in range(n)
    }
    return observed


class TestGreedy:
    def test_temperature_zero_is_argmax(self, sampler: Sampler) -> None:
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
        params = SamplingParams(max_tokens=1, temperature=0.0)
        assert sampler.sample(logits, params) == 1

    def test_top_k_1_equals_greedy(self, sampler: Sampler) -> None:
        # top_k=1 只留最大候选，概率质量 100% 集中，必须等于 argmax
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
        params = SamplingParams(max_tokens=1, temperature=1.0, top_k=1)
        assert sampler.sample(logits, params) == 1


class TestTopK:
    def test_top_k_filters_outside_candidates(self, sampler: Sampler) -> None:
        # 8 个均匀 token，top_k=3 只允许分数最高的 {5,6,7} 出现
        logits = torch.arange(8, dtype=torch.float32)
        params = SamplingParams(max_tokens=1, temperature=1.0, top_k=3)
        observed = _sample_many(sampler, logits, params)
        assert observed == {5, 6, 7}

    def test_top_k_larger_than_vocab_is_noop(self, sampler: Sampler) -> None:
        # k >= vocab 等价于不过滤，不应报错（topk 的 k 越界防御）。
        # 用概率相近的 logits，保证 4 个 token 都能在有限次采样内出现
        logits = torch.tensor([1.0, 1.2, 0.8, 1.1])
        params = SamplingParams(max_tokens=1, temperature=1.0, top_k=100)
        observed = _sample_many(sampler, logits, params)
        assert observed == {0, 1, 2, 3}


class TestTopP:
    def test_top_p_keeps_minimal_set(self, sampler: Sampler) -> None:
        # softmax([10,9,8,0]) ≈ [0.665, 0.245, 0.090, ~0]
        # top_p=0.7：候选 0 的前置累计 = 0 < 0.7 保留；候选 1 的前置累计
        # = 0.665 < 0.7 保留（恰好跨越阈值）；候选 2 前置累计 0.910 >= 0.7 淘汰
        logits = torch.tensor([10.0, 9.0, 8.0, 0.0])
        params = SamplingParams(max_tokens=1, temperature=1.0, top_p=0.7)
        observed = _sample_many(sampler, logits, params)
        assert observed == {0, 1}

    def test_small_top_p_keeps_only_top1(self, sampler: Sampler) -> None:
        # top_p=0.5：候选 1 的前置累计 0.665 已 >= 0.5，只剩 token 0
        logits = torch.tensor([10.0, 9.0, 8.0, 0.0])
        params = SamplingParams(max_tokens=1, temperature=1.0, top_p=0.5)
        observed = _sample_many(sampler, logits, params)
        assert observed == {0}


class TestTemperature:
    def test_low_temperature_concentrates_on_argmax(
        self, sampler: Sampler
    ) -> None:
        # T=0.01 时 argmax 概率≈1，300 次采样应几乎全部命中 argmax
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
        params = SamplingParams(max_tokens=1, temperature=0.01)
        observed = _sample_many(sampler, logits, params)
        assert observed == {1}

    def test_high_temperature_explores_more(self, sampler: Sampler) -> None:
        # T 越大分布越平，4 个候选都应出现
        logits = torch.tensor([1.0, 5.0, 3.0, 2.0])
        params = SamplingParams(max_tokens=1, temperature=100.0)
        observed = _sample_many(sampler, logits, params)
        assert observed == {0, 1, 2, 3}


class TestSeed:
    def test_same_seed_reproducible(self, sampler: Sampler) -> None:
        # 同 seed 的随机序列必须逐 token 一致——这是采样可复现性的合同
        logits = torch.randn(1000)
        p = SamplingParams(max_tokens=1, temperature=1.0)

        gen1 = torch.Generator(device="cpu")
        gen1.manual_seed(42)
        seq1 = [sampler.sample(logits, p, gen1) for _ in range(20)]

        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(42)
        seq2 = [sampler.sample(logits, p, gen2) for _ in range(20)]

        assert seq1 == seq2

    def test_different_seed_differs(self, sampler: Sampler) -> None:
        # 1000 维高熵分布下，不同 seed 生成 20 个 token 完全相同的概率约
        # (1/1000)^20，工程上视为不可能
        logits = torch.randn(1000)
        p = SamplingParams(max_tokens=1, temperature=1.0)

        gen1 = torch.Generator(device="cpu")
        gen1.manual_seed(1)
        seq1 = [sampler.sample(logits, p, gen1) for _ in range(20)]

        gen2 = torch.Generator(device="cpu")
        gen2.manual_seed(2)
        seq2 = [sampler.sample(logits, p, gen2) for _ in range(20)]

        assert seq1 != seq2


class TestParamValidation:
    """非法参数必须在构造期 fail fast，而不是在采样深处产生 NaN。"""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_tokens": 0},
            {"max_tokens": -5},
            {"max_tokens": 1, "temperature": -0.1},
            {"max_tokens": 1, "top_k": 0},
            {"max_tokens": 1, "top_k": -2},
            {"max_tokens": 1, "top_p": 0.0},
            {"max_tokens": 1, "top_p": 1.5},
        ],
    )
    def test_invalid_params_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            SamplingParams(**kwargs)

    def test_valid_edge_params_accepted(self) -> None:
        # 边界合法值不应误伤：greedy、关闭 top-k/top-p、top_p=1.0
        SamplingParams(max_tokens=1, temperature=0.0)
        SamplingParams(max_tokens=1, top_k=-1, top_p=1.0)


class TestInputGuard:
    def test_non_1d_logits_rejected(self, sampler: Sampler) -> None:
        params = SamplingParams(max_tokens=1, temperature=1.0)
        with pytest.raises(ValueError, match="1D"):
            sampler.sample(torch.zeros(2, 4), params)
