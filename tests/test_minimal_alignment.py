"""Task 03 对齐测试：MinimalQwen vs HF Qwen2ForCausalLM（需要真实模型）。

对齐分四个粒度，从细到粗：
1. embedding 输出；
2. 每层的 attention / MLP 子模块输出（HF hook 抓取中间量，喂同样的输入）；
3. 每层的整层输出（HF output_hidden_states）；
4. 最终 logits 的 allclose + top-1 agreement，以及 greedy 生成逐 token 一致。

容差来自 liteinfer.model.alignment（CPU FP32: 1e-4），补充条款 A4。
"""

from __future__ import annotations

import pytest
import torch

from liteinfer import EngineConfig
from liteinfer.model.alignment import alignment_tolerances, top1_agreement
from liteinfer.model.minimal.model import build_causal_mask
from liteinfer.model.minimal.weights import MinimalLoaded, load_minimal_from_hf

pytestmark = pytest.mark.model

PROMPT = "The capital of France is"
N_GREEDY_STEPS = 4


@pytest.fixture(scope="module")
def loaded() -> MinimalLoaded:
    """模型只加载一次：HF 与 MinimalQwen 共享同一份 checkpoint 权重。"""
    cfg = EngineConfig(device="cpu", dtype=torch.float32)
    return load_minimal_from_hf(cfg)


@pytest.fixture(scope="module")
def ctx(loaded: MinimalLoaded):
    """编码 prompt，并准备好两个模型 forward 都需要的位置/掩码上下文。"""
    from transformers import AutoTokenizer

    cfg = EngineConfig(device="cpu", dtype=torch.float32)
    cache = str(cfg.resolved_hf_cache_dir() / "hub")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id, cache_dir=cache, local_files_only=cfg.local_files_only
    )
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    seq_len = input_ids.shape[1]
    pos = torch.arange(seq_len).unsqueeze(0)
    mask = build_causal_mask(seq_len, input_ids.device, torch.float32)
    return loaded, input_ids, pos, mask


class TestWeightLoading:
    def test_dtype_and_device_follow_config(self, loaded: MinimalLoaded) -> None:
        # 补充条款 A1/A2：dtype/device 只能来自 EngineConfig（CPU 上必须 FP32）
        for param in loaded.minimal.parameters():
            assert param.dtype == loaded.dtype == torch.float32
            assert param.device.type == loaded.device.type == "cpu"

    def test_all_hf_weights_consumed(self, loaded: MinimalLoaded) -> None:
        # strict 加载的旁证：state_dict 存储量与 HF 一致（用 state_dict 而
        # 非 parameters()：tied embedding 下 HF 的 parameters() 去重只算
        # 一次，而我们补的 lm_head 键指向同一权重，state_dict 口径才对齐；
        # RoPE 的 inv_freq 是非持久 buffer，两边都不进 state_dict）
        n_mine = sum(v.numel() for v in loaded.minimal.state_dict().values())
        n_hf = sum(v.numel() for v in loaded.hf_model.state_dict().values())
        assert n_mine == n_hf


class TestLayerWiseAlignment:
    atol: float
    rtol: float

    @pytest.fixture(autouse=True)
    def _tolerances(self) -> None:
        self.atol, self.rtol = alignment_tolerances("cpu")

    def test_embedding_output(self, ctx) -> None:
        loaded, input_ids, _, _ = ctx
        mine = loaded.minimal.model.embed_tokens(input_ids)
        hf = loaded.hf_model.model.embed_tokens(input_ids)
        torch.testing.assert_close(mine, hf, atol=self.atol, rtol=self.rtol)

    def test_per_layer_attention_and_mlp(self, ctx) -> None:
        """子模块级对齐：用 hook 抓 HF 每层 self_attn / mlp 的 (输入, 输出)，
        把相同输入喂给我们的模块，逐一比较输出。"""
        loaded, input_ids, pos, mask = ctx

        hf_records: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        def make_hook():
            def hook(module, args, kwargs, output):
                hidden_in = args[0] if args else kwargs["hidden_states"]
                # transformers 5.x 的 self_attn 返回 (output, attn_weights)
                # 二元组，mlp 返回裸 tensor；统一取第一个张量
                tensor_out = output[0] if isinstance(output, tuple) else output
                hf_records.append((hidden_in, tensor_out))

            return hook

        handles = []
        for layer in loaded.hf_model.model.layers:
            handles.append(layer.self_attn.register_forward_hook(
                make_hook(), with_kwargs=True))
            handles.append(layer.mlp.register_forward_hook(
                make_hook(), with_kwargs=True))

        with torch.inference_mode():
            loaded.hf_model(input_ids)
        for handle in handles:
            handle.remove()

        # 每层两条记录：先是 self_attn，再是 mlp
        assert len(hf_records) == 2 * len(loaded.minimal.model.layers)
        with torch.inference_mode():
            for i, (hidden_in, hf_out) in enumerate(hf_records):
                layer_idx = i // 2
                if i % 2 == 0:
                    mine_out = loaded.minimal.model.layers[layer_idx].self_attn(
                        hidden_in, pos, mask
                    )
                else:
                    mine_out = loaded.minimal.model.layers[layer_idx].mlp(hidden_in)
                torch.testing.assert_close(
                    mine_out, hf_out, atol=self.atol, rtol=self.rtol,
                    msg=lambda m, idx=i: f"第 {idx // 2} 层"
                    f"{'self_attn' if idx % 2 == 0 else 'mlp'} 不对齐: {m}",
                )

    def test_per_layer_hidden_states(self, ctx) -> None:
        """整层级对齐：逐层推进我们的主干，与 HF 的 hidden_states 序列对照。

        HF 的 hidden_states[i] 是第 i 层的输入（即第 i-1 层的输出），
        最后一项是 final norm 之后的值。
        """
        loaded, input_ids, pos, mask = ctx
        with torch.inference_mode():
            hf_hidden = loaded.hf_model.model(
                input_ids, output_hidden_states=True
            ).hidden_states
            h = loaded.minimal.model.embed_tokens(input_ids)
            for i, layer in enumerate(loaded.minimal.model.layers):
                torch.testing.assert_close(
                    h, hf_hidden[i], atol=self.atol, rtol=self.rtol,
                    msg=lambda m, idx=i: f"第 {idx} 层输入不对齐: {m}",
                )
                h = layer(h, pos, mask)
                if i + 1 < len(loaded.minimal.model.layers):
                    torch.testing.assert_close(
                        h, hf_hidden[i + 1], atol=self.atol, rtol=self.rtol,
                        msg=lambda m, idx=i: f"第 {idx} 层输出不对齐: {m}",
                    )
            final = loaded.minimal.model.norm(h)
        torch.testing.assert_close(final, hf_hidden[-1], atol=self.atol, rtol=self.rtol)


class TestModelAlignment:
    atol: float
    rtol: float

    @pytest.fixture(autouse=True)
    def _tolerances(self) -> None:
        self.atol, self.rtol = alignment_tolerances("cpu")

    def test_logits_allclose_and_top1(self, ctx) -> None:
        loaded, input_ids, _, _ = ctx
        with torch.inference_mode():
            mine = loaded.minimal(input_ids)
            hf = loaded.hf_model(input_ids).logits
        torch.testing.assert_close(mine, hf, atol=self.atol, rtol=self.rtol)
        agreement = top1_agreement(mine, hf)
        assert agreement == 1.0, f"top-1 agreement = {agreement:.4f}，应为 100%"

    def test_greedy_generation_token_by_token(self, ctx) -> None:
        """端到端冒烟：两边各自 argmax 生成若干步，逐 token 一致。

        只对齐"选 token"这一消费语义，不追求 logits 逐位相同——
        生成分叉才是真正不可接受的失败模式。
        """
        loaded, input_ids, _, _ = ctx
        ids = input_ids.clone()
        with torch.inference_mode():
            for _ in range(N_GREEDY_STEPS):
                mine_next = loaded.minimal.greedy_next_token(ids)
                hf_next = int(loaded.hf_model(ids).logits[0, -1].argmax().item())
                assert mine_next == hf_next
                ids = torch.cat([ids, torch.tensor([[mine_next]])], dim=1)
