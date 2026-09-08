"""Task 04 对齐测试：KV Cache 版生成 vs 无缓存 / vs HF（需要真实模型）。

三层判据，从细到粗：
1. prefill 的 logits 与 HF 全序列前向对齐；
2. 逐步 decode 的 logits 与"HF 前向到同一位置"对齐（这一步错了，生成
   到第 3、4 个 token 才开始分叉，最难排查）；
3. 端到端文本与 Task 02 的 ManualGenerator（HF 模型、无缓存）逐字一致，
   并与同模型的 no-cache 路径一致。

容差来自 liteinfer.model.alignment（CPU FP32: 1e-4），补充条款 A4。
"""

from __future__ import annotations

import pytest
import torch

from liteinfer import EngineConfig
from liteinfer.model.alignment import alignment_tolerances, top1_agreement
from liteinfer.model.cached_generator import CachedGenerationOutput, CachedGenerator
from liteinfer.model.eos import resolve_eos_ids
from liteinfer.model.generator import ManualGenerator
from liteinfer.model.minimal.weights import MinimalLoaded, load_minimal_from_hf
from liteinfer.sampling.params import SamplingParams

pytestmark = pytest.mark.model

PROMPT = "The capital of France is"
N_TOKENS = 24
N_DECODE_STEPS = 3


@pytest.fixture(scope="module")
def cfg() -> EngineConfig:
    return EngineConfig(device="cpu", dtype=torch.float32, max_new_tokens=N_TOKENS)


@pytest.fixture(scope="module")
def loaded(cfg: EngineConfig) -> MinimalLoaded:
    return load_minimal_from_hf(cfg)


@pytest.fixture(scope="module")
def gen(cfg: EngineConfig, loaded: MinimalLoaded) -> CachedGenerator:
    """复用模块级权重构造生成器；EOS 从 HF 模型解析（见 eos.py 说明）。"""
    return CachedGenerator(
        loaded.minimal,
        loaded.tokenizer,
        cfg,
        eos_ids=resolve_eos_ids(loaded.hf_model, loaded.tokenizer),
    )


@pytest.fixture(scope="module")
def runs(gen: CachedGenerator, loaded: MinimalLoaded, cfg: EngineConfig):
    """一次跑完三条链，供多个断言复用（CPU 上每条约 5~15 秒）。"""
    # 预热：第一次 forward 要付线程池初始化/内存分配/算子选择的成本（实测
    # 冷启动比热态慢 3~5 倍），不预热的话性能断言测的是"谁先跑"而不是
    # "谁更快"
    warmup = SamplingParams(max_tokens=2, temperature=0.0)
    gen.generate(PROMPT, warmup, use_cache=True)
    gen.generate(PROMPT, warmup, use_cache=False)

    params = SamplingParams(max_tokens=N_TOKENS, temperature=0.0)
    return {
        "kv": gen.generate(PROMPT, params, use_cache=True),
        "no_cache": gen.generate(PROMPT, params, use_cache=False),
        "manual": ManualGenerator(loaded.hf_model, loaded.tokenizer, cfg).generate(
            PROMPT, params
        ),
    }


class TestPrefillAndDecodeAlignment:
    atol: float
    rtol: float

    @pytest.fixture(autouse=True)
    def _tolerances(self) -> None:
        self.atol, self.rtol = alignment_tolerances("cpu")

    def test_prefill_logits_match_hf(self, loaded: MinimalLoaded, gen: CachedGenerator) -> None:
        input_ids = loaded.tokenizer(PROMPT, return_tensors="pt").input_ids
        cache = gen.new_cache(input_ids.shape[1] + N_TOKENS)
        with torch.inference_mode():
            mine = loaded.minimal(
                input_ids, kv_caches=cache.layer_caches, write_pos=0
            )
            hf = loaded.hf_model(input_ids).logits
        torch.testing.assert_close(mine, hf, atol=self.atol, rtol=self.rtol)
        assert top1_agreement(mine, hf) == 1.0

    def test_decode_logits_match_hf_step_by_step(
        self, loaded: MinimalLoaded, gen: CachedGenerator
    ) -> None:
        """每 decode 一步，都与 HF 对同一条扩展序列的前向结果对照。"""
        tokenizer = loaded.tokenizer
        input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
        cache = gen.new_cache(input_ids.shape[1] + N_DECODE_STEPS + 1)
        seq = input_ids.clone()
        with torch.inference_mode():
            logits = loaded.minimal(seq, kv_caches=cache.layer_caches, write_pos=0)
            for step in range(N_DECODE_STEPS):
                next_id = int(logits[0, -1].argmax())
                seq = torch.cat([seq, torch.tensor([[next_id]])], dim=1)
                cached = seq.shape[1] - 1
                mine = loaded.minimal(
                    torch.tensor([[next_id]]),
                    position_ids=torch.tensor([[cached]]),
                    kv_caches=cache.layer_caches,
                    write_pos=cached,
                )
                hf = loaded.hf_model(seq).logits
                torch.testing.assert_close(
                    mine, hf[:, -1:, :], atol=self.atol, rtol=self.rtol,
                    msg=lambda m, s=step: f"第 {s} 步 decode 与 HF 不一致: {m}",
                )
                logits = mine


class TestEndToEndEquivalence:
    def test_kv_matches_hf_manual_generator(self, runs) -> None:
        # 与 Task 02 的 HF 无缓存链逐字一致：端到端最强的判据
        assert runs["kv"].text == runs["manual"].text

    def test_kv_matches_same_model_no_cache(self, runs) -> None:
        # 与同模型的 no-cache 路径一致：排除了"HF vs 自建实现"的算子差异
        assert runs["kv"].text == runs["no_cache"].text
        assert runs["kv"].output_tokens == runs["no_cache"].output_tokens

    def test_finish_reason_is_consistent(self, runs) -> None:
        assert runs["kv"].finish_reason == runs["manual"].finish_reason


class TestCacheAccounting:
    def test_cached_tokens_accounting(self, runs) -> None:
        out: CachedGenerationOutput = runs["kv"]
        # prefill 写入 prompt；之后每个 decode 步写入 1 个 token。
        # 因长度上限结束时，最后一个 token 只被采样、不再被 forward，
        # 所以比"prompt + 输出"少 1；EOS 结束时刚好相等
        expected = out.prompt_tokens + out.output_tokens - (
            1 if out.finish_reason == "length" else 0
        )
        assert out.cached_tokens == expected

    def test_cache_bytes_formula(self, runs, gen: CachedGenerator) -> None:
        out: CachedGenerationOutput = runs["kv"]
        num_layers, num_kv_heads, head_dim = gen._kv_dims
        capacity = out.prompt_tokens + N_TOKENS
        expected = 2 * num_layers * num_kv_heads * head_dim * 4 * capacity
        assert out.cache_bytes == expected
        assert out.cache_bytes > 0
        assert runs["no_cache"].cache_bytes == 0


class TestSpeedup:
    def test_kv_faster_than_no_cache(self, runs) -> None:
        """KV 版必须比无缓存快（CPU 上的加速比只作逻辑验证，不进简历）。"""
        kv, no_cache = runs["kv"], runs["no_cache"]
        assert kv.text == no_cache.text, "输出不一致时比较延迟没有意义"
        # 留 10% 余量吸收计时抖动；实测（预热后，24 token）约 1.8x
        assert kv.latency_s * 1.1 < no_cache.latency_s, (
            f"KV 版未取得加速: kv={kv.latency_s:.3f}s no_cache={no_cache.latency_s:.3f}s"
        )
        # prefill 只占一小段，decode 才是 KV 复用的主战场
        assert kv.prefill_latency_s < kv.latency_s
        assert kv.decode_latency_s > 0
