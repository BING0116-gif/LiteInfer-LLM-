"""Task 02 端到端测试：手写生成循环 vs HF baseline 对齐（需要真实模型）。

核心验收（docs/07 Task 02）：greedy 输出与 HF baseline 基本一致。
两个实现对齐测试共享同一份加载好的权重，保证差异只能来自生成逻辑本身。
"""

from __future__ import annotations

import pytest
import torch

from liteinfer import EngineConfig
from liteinfer.model.baseline import HFBaseline
from liteinfer.model.generator import ManualGenerator
from liteinfer.model.loader import load_model_and_tokenizer
from liteinfer.sampling.params import SamplingParams

pytestmark = pytest.mark.model

PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def pair() -> tuple[HFBaseline, ManualGenerator]:
    """模型只加载一次，baseline 与手写循环共享同一份权重与 tokenizer。"""
    cfg = EngineConfig(device="cpu", dtype=torch.float32, max_new_tokens=16)
    loaded = load_model_and_tokenizer(cfg)
    baseline = HFBaseline(loaded.model, loaded.tokenizer, cfg)
    manual = ManualGenerator(loaded.model, loaded.tokenizer, cfg)
    return baseline, manual


class TestGreedyAlignment:
    def test_greedy_matches_hf(self, pair: tuple[HFBaseline, ManualGenerator]) -> None:
        baseline, manual = pair
        hf_out = baseline.generate(PROMPT, max_new_tokens=8)
        my_out = manual.generate(PROMPT, SamplingParams(max_tokens=8, temperature=0.0))
        assert my_out.text.strip(), "手写循环 greedy 输出为空"
        assert my_out.text == hf_out.text, (
            f"greedy 对齐失败:\n  HF    = {hf_out.text!r}\n  manual= {my_out.text!r}"
        )

    def test_greedy_is_deterministic(
        self, pair: tuple[HFBaseline, ManualGenerator]
    ) -> None:
        _, manual = pair
        o1 = manual.generate(PROMPT, SamplingParams(max_tokens=8, temperature=0.0))
        o2 = manual.generate(PROMPT, SamplingParams(max_tokens=8, temperature=0.0))
        assert o1.text == o2.text
        assert o1.output_tokens == o2.output_tokens

    def test_top_k_1_equals_greedy(
        self, pair: tuple[HFBaseline, ManualGenerator]
    ) -> None:
        _, manual = pair
        greedy = manual.generate(PROMPT, SamplingParams(max_tokens=8, temperature=0.0))
        topk1 = manual.generate(
            PROMPT, SamplingParams(max_tokens=8, temperature=1.0, top_k=1)
        )
        assert topk1.text == greedy.text


class TestSampling:
    def test_seed_reproducible(self, pair: tuple[HFBaseline, ManualGenerator]) -> None:
        _, manual = pair
        p1 = SamplingParams(max_tokens=16, temperature=0.8, top_p=0.9, seed=7)
        p2 = SamplingParams(max_tokens=16, temperature=0.8, top_p=0.9, seed=7)
        assert manual.generate(PROMPT, p1).text == manual.generate(PROMPT, p2).text

    def test_output_respects_max_tokens(
        self, pair: tuple[HFBaseline, ManualGenerator]
    ) -> None:
        _, manual = pair
        out = manual.generate(PROMPT, SamplingParams(max_tokens=8, temperature=0.7, seed=3))
        assert 1 <= out.output_tokens <= 8
        assert out.prompt_tokens > 0
        assert out.latency_s > 0

    def test_finish_reason_semantics(
        self, pair: tuple[HFBaseline, ManualGenerator]
    ) -> None:
        _, manual = pair
        out = manual.generate(PROMPT, SamplingParams(max_tokens=8, temperature=0.0))
        # 8 个 token 的短生成大概率撞不到 EOS；两种终止原因都合法，
        # 但必须是这两个值之一（Task 05 状态机依赖这个语义）
        assert out.finish_reason in {"eos", "length"}

    def test_output_excludes_prompt(
        self, pair: tuple[HFBaseline, ManualGenerator]
    ) -> None:
        _, manual = pair
        out = manual.generate(PROMPT, SamplingParams(max_tokens=8, temperature=0.0))
        assert not out.text.startswith(PROMPT)
