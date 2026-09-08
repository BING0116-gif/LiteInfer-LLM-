"""Task 04 快测：不依赖真实模型，用小随机 MinimalQwen + 纯张量验证。

分两层：
1. 缓存本体（形状、字节数、读写语义、容量守卫）——纯张量，毫秒级；
2. 缓存与计算图的一致性——小随机模型上"整段 forward"必须等于
   "prefill + 逐步 decode"，这是 KV Cache 正确性的核心判据。

第 2 层用小模型而不是 0.5B：算子路径完全相同，但能在秒级跑完，
且不受权重下载/数值量级影响，失败时定位到的是逻辑而不是精度。
"""

from __future__ import annotations

import pytest
import torch

from liteinfer.cache.contiguous import ContiguousKVCache, KVCacheConfig, LayerKVCache
from liteinfer.model.minimal.model import MinimalQwenForCausalLM, build_causal_mask
from liteinfer.model.minimal.rotary import apply_rotary_pos_emb

TINY = dict(
    vocab_size=32,
    hidden_size=16,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,  # GQA：4 个 Q 头共享 2 个 KV 头
    head_dim=4,
    intermediate_size=24,
    rope_theta=10000.0,
    rms_norm_eps=1e-6,
    attention_bias=True,
    use_qk_norm=False,
    mlp_bias=False,
)


def _tiny_model(seed: int = 0) -> MinimalQwenForCausalLM:
    torch.manual_seed(seed)
    model = MinimalQwenForCausalLM(**TINY)  # type: ignore[arg-type]
    model.eval()
    return model


def _cache_for(model: MinimalQwenForCausalLM, max_seq_len: int) -> ContiguousKVCache:
    attn = model.model.layers[0].self_attn
    cfg = KVCacheConfig(
        num_layers=len(model.model.layers),
        num_kv_heads=attn.num_kv_heads,
        head_dim=attn.head_dim,
        max_seq_len=max_seq_len,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    return ContiguousKVCache(cfg)


def _greedy(
    model: MinimalQwenForCausalLM, prompt_ids: torch.Tensor, max_new_tokens: int, use_cache: bool
) -> list[int]:
    """greedy 生成：返回新生成的 token id 列表。"""
    cache = _cache_for(model, prompt_ids.shape[1] + max_new_tokens) if use_cache else None
    caches = cache.layer_caches if cache else None
    running = prompt_ids.clone()
    cached = prompt_ids.shape[1]
    out: list[int] = []
    with torch.inference_mode():
        logits = model(running, kv_caches=caches, write_pos=0)
        next_id = int(logits[0, -1].argmax())
        while len(out) < max_new_tokens:
            out.append(next_id)
            if len(out) >= max_new_tokens:
                break
            step = torch.tensor([[next_id]])
            if use_cache:
                logits = model(
                    step,
                    position_ids=torch.tensor([[cached]]),
                    kv_caches=caches,
                    write_pos=cached,
                )
                cached += 1
            else:
                running = torch.cat([running, step], dim=1)
                logits = model(running)
            next_id = int(logits[0, -1].argmax())
    return out


class TestKVCacheConfig:
    def test_bytes_per_token_formula(self) -> None:
        # docs/03 模块 5：2（K+V）× L × KVH × D × element_size
        cfg = KVCacheConfig(
            num_layers=2, num_kv_heads=2, head_dim=4, max_seq_len=10,
            dtype=torch.float32, device=torch.device("cpu"),
        )
        assert cfg.bytes_per_token() == 2 * 2 * 2 * 4 * 4
        assert cfg.total_bytes() == cfg.bytes_per_token() * 10

    def test_shape_params_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="正整数"):
            KVCacheConfig(
                num_layers=0, num_kv_heads=2, head_dim=4, max_seq_len=8,
                dtype=torch.float32, device=torch.device("cpu"),
            )

    def test_from_hf_config_derives_head_dim(self) -> None:
        class _Cfg:
            num_hidden_layers = 24
            num_key_value_heads = 2
            num_attention_heads = 14
            hidden_size = 896

        cfg = KVCacheConfig.from_hf_config(_Cfg(), 128, torch.float32, "cpu")
        # 0.5B 没有 head_dim 字段，按 hidden // num_heads = 64 推导
        assert (cfg.num_layers, cfg.num_kv_heads, cfg.head_dim) == (24, 2, 64)
        assert cfg.bytes_per_token() == 2 * 24 * 2 * 64 * 4


class TestLayerKVCache:
    def _layer(self, max_seq_len: int = 8) -> LayerKVCache:
        cache = ContiguousKVCache(
            KVCacheConfig(
                num_layers=1, num_kv_heads=2, head_dim=3, max_seq_len=max_seq_len,
                dtype=torch.float32, device=torch.device("cpu"),
            )
        )
        return cache.layer_caches[0]

    def test_append_twice_equals_concatenation(self) -> None:
        layer = self._layer(max_seq_len=8)
        first = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(1, 2, 2, 3)
        second = torch.arange(2 * 2 * 3, dtype=torch.float32).reshape(1, 2, 2, 3) + 100
        k1, v1 = layer.append(first, first, 0)
        k2, v2 = layer.append(second, second, 2)
        expected = torch.cat([first, second], dim=1)
        assert k2.shape == (1, 4, 2, 3)
        torch.testing.assert_close(k2, expected)
        torch.testing.assert_close(v2, expected)
        # 第一次返回的视图只覆盖已写入的部分，不能被后续写入"穿越"污染
        assert k1.shape == (1, 2, 2, 3)
        torch.testing.assert_close(k1, first)

    def test_read_returns_view_without_copy(self) -> None:
        layer = self._layer(max_seq_len=8)
        k, _ = layer.read(4)
        assert k.shape == (1, 4, 2, 3)
        # 视图而非副本：历史不产生额外拷贝，这是"零拷贝读取"的直接证据
        # （torch 没有 shares_memory，用 storage 指针判断共享内存）
        assert k.untyped_storage().data_ptr() == layer.k_buf.untyped_storage().data_ptr()

    def test_append_beyond_capacity_raises(self) -> None:
        layer = self._layer(max_seq_len=4)
        chunk = torch.zeros(1, 3, 2, 3)
        layer.append(chunk, chunk, 0)
        with pytest.raises(ValueError, match="越界"):
            layer.append(chunk, chunk, 2)  # 2 + 3 > 4

    def test_read_out_of_range_raises(self) -> None:
        layer = self._layer(max_seq_len=4)
        with pytest.raises(ValueError, match="超出"):
            layer.read(5)

    def test_batch_greater_than_one_rejected(self) -> None:
        layer = self._layer()
        chunk = torch.zeros(2, 1, 2, 3)
        with pytest.raises(ValueError, match="batch"):
            layer.append(chunk, chunk, 0)

    def test_kv_shape_mismatch_rejected(self) -> None:
        layer = self._layer()
        with pytest.raises(ValueError, match="形状不一致"):
            layer.append(torch.zeros(1, 1, 2, 3), torch.zeros(1, 2, 2, 3), 0)


class TestContiguousKVCache:
    def test_dtype_and_device_follow_config(self) -> None:
        # 补充条款 A1/A2：dtype/device 只能来自配置，且 CPU 上必须 FP32
        cache = _cache_for(_tiny_model(), 16)
        assert cache.k_cache.dtype == torch.float32
        assert cache.v_cache.dtype == torch.float32
        assert cache.k_cache.device.type == "cpu"

    def test_nbytes_matches_config(self) -> None:
        cache = _cache_for(_tiny_model(), 16)
        assert cache.nbytes == cache.cfg.total_bytes()
        # 2(K/V) × 2 层 × 2 KV 头 × 4 head_dim × 4 字节 × 16 token
        assert cache.nbytes == 2 * 2 * 2 * 4 * 4 * 16

    def test_reset_clears_content(self) -> None:
        cache = _cache_for(_tiny_model(), 8)
        chunk = torch.ones(1, 2, 2, 4)
        for layer in cache.layer_caches:
            layer.append(chunk, chunk, 0)
        assert cache.k_cache.abs().sum() > 0
        cache.reset()
        assert cache.k_cache.abs().sum() == 0


class TestCausalMaskWithPast:
    def test_past_len_zero_keeps_triu_semantics(self) -> None:
        mask = build_causal_mask(4, torch.device("cpu"), torch.float32)
        assert mask.shape == (1, 1, 4, 4)
        lower = torch.tril(torch.ones(4, 4)).bool()
        assert torch.all(mask[0, 0][lower] == 0)
        assert torch.all(mask[0, 0][~lower] == torch.finfo(torch.float32).min)

    def test_past_len_shifts_visibility(self) -> None:
        # query 的绝对位置是 3、4；key 是 0..4
        mask = build_causal_mask(2, torch.device("cpu"), torch.float32, past_len=3)
        assert mask.shape == (1, 1, 2, 5)
        row0 = mask[0, 0, 0]
        row1 = mask[0, 0, 1]
        minimum = torch.finfo(torch.float32).min
        # 第 0 个 query（位置 3）只能看到 key 0..3
        assert row0[:4].tolist() == [0.0] * 4 and row0[4].item() == minimum
        # 第 1 个 query（位置 4）能看到全部 key
        assert row1.tolist() == [0.0] * 5


class TestKVForwardEquivalence:
    """小随机模型上：带缓存的增量前向必须等价于无缓存的整段前向。"""

    def test_prefill_logits_equal_full_forward(self) -> None:
        model = _tiny_model()
        ids = torch.randint(0, TINY["vocab_size"], (1, 7))
        cache = _cache_for(model, 16)
        with torch.inference_mode():
            full = model(ids)
            prefill = model(ids, kv_caches=cache.layer_caches, write_pos=0)
        torch.testing.assert_close(prefill, full, atol=1e-6, rtol=1e-6)

    def test_incremental_decode_logits_equal_full_forward(self) -> None:
        model = _tiny_model()
        ids = torch.randint(0, TINY["vocab_size"], (1, 5))
        cache = _cache_for(model, 32)
        seq = ids.clone()
        with torch.inference_mode():
            logits = model(seq, kv_caches=cache.layer_caches, write_pos=0)
            for i in range(6):
                next_id = int(logits[0, -1].argmax())
                step = torch.tensor([[next_id]])
                cached = seq.shape[1]
                kv_logits = model(
                    step,
                    position_ids=torch.tensor([[cached]]),
                    kv_caches=cache.layer_caches,
                    write_pos=cached,
                )
                seq = torch.cat([seq, step], dim=1)
                reference = model(seq)  # 无缓存：整段重算，作为黄金参照
                torch.testing.assert_close(
                    kv_logits, reference[:, -1:, :], atol=1e-5, rtol=1e-5,
                    msg=lambda m, idx=i: f"第 {idx} 步 decode 与整段前向不一致: {m}",
                )
                logits = kv_logits

    def test_decode_needs_absolute_position_ids(self) -> None:
        # RoPE 是绝对位置编码：位置写错，续写内容就会整体错位
        model = _tiny_model()
        ids = torch.randint(0, TINY["vocab_size"], (1, 5))
        good, bad = _cache_for(model, 16), _cache_for(model, 16)
        with torch.inference_mode():
            model(ids, kv_caches=good.layer_caches, write_pos=0)
            model(ids, kv_caches=bad.layer_caches, write_pos=0)
            step = torch.tensor([[7]])
            pos = torch.tensor([[5]])
            correct = model(step, position_ids=pos, kv_caches=good.layer_caches, write_pos=5)
            shifted = model(
                step, position_ids=torch.tensor([[0]]), kv_caches=bad.layer_caches, write_pos=5
            )
        assert not torch.allclose(correct, shifted, atol=1e-6)

    def test_greedy_generation_matches_no_cache(self) -> None:
        model = _tiny_model()
        ids = torch.randint(0, TINY["vocab_size"], (1, 6))
        with torch.inference_mode():
            with_cache = _greedy(model, ids, max_new_tokens=8, use_cache=True)
            without_cache = _greedy(model, ids, max_new_tokens=8, use_cache=False)
        assert with_cache == without_cache

    def test_cache_stores_rotated_keys(self) -> None:
        """缓存里躺的必须是 **RoPE 之后** 的 K（写入时机检查）。

        写成旋转前的 K 也能"跑出合理的文本"，但每步 decode 都得把历史
        重算一遍旋转——是最难从结果反推的一类错误，所以用数值直接钉住。
        """
        model = _tiny_model(seed=1)
        seq_len = 6
        ids = torch.randint(0, TINY["vocab_size"], (1, seq_len))
        cache = _cache_for(model, 16)
        with torch.inference_mode():
            model(ids, kv_caches=cache.layer_caches, write_pos=0)

            # 手工重放第 0 层的 K/V 计算（第一层的输入就是 embedding，无需推层）
            attn = model.model.layers[0].self_attn
            normed = model.model.layers[0].input_layernorm(model.model.embed_tokens(ids))
            k_pre = attn.k_proj(normed).view(1, seq_len, attn.num_kv_heads, attn.head_dim)
            v_ref = attn.v_proj(normed).view(1, seq_len, attn.num_kv_heads, attn.head_dim)
            cos, sin = attn.rotary_emb(torch.arange(seq_len).unsqueeze(0))
            # apply_rotary_pos_emb 返回 (q, k)，这里只需要 k 分支
            k_rotated = apply_rotary_pos_emb(
                k_pre.transpose(1, 2), k_pre.transpose(1, 2), cos, sin
            )[1]

        k_cached, v_cached = cache.layer_caches[0].read(seq_len)
        torch.testing.assert_close(k_cached, k_rotated.transpose(1, 2), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(v_cached, v_ref, atol=1e-6, rtol=1e-6)
        # 反向证据：缓存内容确实不是旋转前的 K
        assert not torch.allclose(k_cached, k_pre, atol=1e-6)
