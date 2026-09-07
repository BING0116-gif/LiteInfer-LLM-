"""HF Baseline 端到端测试（需要真实模型，标记 model，走缓存不重复下载）。"""

from __future__ import annotations

import pytest
import torch

from liteinfer import EngineConfig
from liteinfer.model.baseline import HFBaseline

pytestmark = pytest.mark.model

PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def baseline() -> HFBaseline:
    # 模型只加载一次，供本模块全部用例复用；CPU FP32 是补充条款 A2 的本机约定
    cfg = EngineConfig(device="cpu", dtype=torch.float32, max_new_tokens=16)
    return HFBaseline.from_config(cfg)


class TestHFBaseline:
    def test_dtype_guard_applied(self, baseline: HFBaseline) -> None:
        # 即使配置误传 fp16，加载守卫也必须把它落到 float32
        assert baseline.dtype is torch.float32
        assert baseline.device.type == "cpu"

    def test_greedy_generation(self, baseline: HFBaseline) -> None:
        out = baseline.generate(PROMPT, max_new_tokens=8)
        assert out.text.strip(), "生成文本为空"
        assert 1 <= out.output_tokens <= 8  # 可能提前命中 EOS，允许少于上限
        assert out.prompt_tokens > 0
        assert out.latency_s > 0
        assert out.tokens_per_s > 0

    def test_greedy_is_deterministic(self, baseline: HFBaseline) -> None:
        # greedy 不采样，同输入必须同输出——这是 Task 02 手写循环的对齐前提
        o1 = baseline.generate(PROMPT, max_new_tokens=8)
        o2 = baseline.generate(PROMPT, max_new_tokens=8)
        assert o1.text == o2.text
        assert o1.output_tokens == o2.output_tokens

    def test_output_excludes_prompt(self, baseline: HFBaseline) -> None:
        # 只解码新增 token：prompt 原文不得混入生成文本
        out = baseline.generate(PROMPT, max_new_tokens=8)
        assert not out.text.startswith(PROMPT)
