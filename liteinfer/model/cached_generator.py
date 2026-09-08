"""Task 04：KV Cache 版生成主链——prefill 一次，decode 逐步复用历史 KV。

与 Task 02 ``ManualGenerator`` 的关系：
- 循环骨架、Sampler、EOS 语义完全一致（EOS 解析共用 ``model.eos``）；
- 区别只在 forward 的形态：无缓存每步重算整条序列 O(n²)，本模块 prefill
  一次后每步只 forward 1 个新 token，历史 K/V 从缓存里读；
- ``generate(..., use_cache=False)`` 保留无缓存路径，用于**同模型**的消融
  对照：拿自建实现和 HF 比会把"算子实现差异"混进"KV 复用收益"里。

生产路径说明：本模块跑在自建的 MinimalQwen 上（Task 03 已与 HF 逐层对齐），
``model.generate()`` 仍然只在 Task 01 的 baseline 里出现过一次。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch

from liteinfer.cache.contiguous import ContiguousKVCache, KVCacheConfig
from liteinfer.config import EngineConfig
from liteinfer.model.eos import resolve_eos_ids
from liteinfer.model.generator import GenerationOutput
from liteinfer.model.minimal.weights import load_minimal_from_hf
from liteinfer.sampling.params import SamplingParams
from liteinfer.sampling.sampler import Sampler

logger = logging.getLogger("liteinfer.model.cached_generator")


@dataclass
class CachedGenerationOutput(GenerationOutput):
    """带缓存生成的结果：在 GenerationOutput 上补三段 KV 相关观测。

    - ``prefill_latency_s``：第一次 forward（处理完整 prompt）的耗时；
    - ``decode_latency_s``：后续所有 forward 的累计耗时；
    - ``cached_tokens`` / ``cache_bytes``：结束时缓存里躺了多少 token、
      占多少字节（无缓存路径为 0）。

    拆开 prefill/decode 是有意为之：这两段的瓶颈完全不同（prefill 偏
    compute-bound、decode 偏 memory-bound），合在一起的 tokens/s 会掩盖
    TTFT 与 TPOT 的真实差异，Task 10 的指标也要按这两段统计。
    """

    prefill_latency_s: float = 0.0
    decode_latency_s: float = 0.0
    cached_tokens: int = 0
    cache_bytes: int = 0


def _model_kv_dims(model: Any) -> tuple:
    """从 MinimalQwen 读出缓存形状三元组 ``(num_layers, num_kv_heads, head_dim)``。

    按 KV 头数而不是 Q 头数：GQA 下二者不等（0.5B 是 14 vs 2），
    存成 Q 头数会直接把缓存放大 7 倍。
    """
    try:
        layers = model.model.layers
        attn = layers[0].self_attn
    except AttributeError as exc:  # 不是 MinimalQwen 结构，别猜
        raise TypeError(
            "CachedGenerator 需要 MinimalQwenForCausalLM（有 model.layers[i].self_attn），"
            f"收到 {type(model).__name__}"
        ) from exc
    return len(layers), attn.num_kv_heads, attn.head_dim


class CachedGenerator:
    """带 KV Cache 的自回归生成。

    缓存的生命周期由本类持有（一次 generate 一份），模型只负责往里写。
    未来 Engine/Scheduler 接管缓存管理后，这里换成"借一块 block"即可，
    生成循环本身不用动。
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        cfg: EngineConfig,
        eos_ids: Optional[Iterable[int]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.sampler = Sampler()
        self._eos_ids = (
            frozenset(int(x) for x in eos_ids)
            if eos_ids is not None
            else resolve_eos_ids(model, tokenizer)
        )
        self._kv_dims = _model_kv_dims(model)

    @classmethod
    def from_config(cls, cfg: EngineConfig) -> "CachedGenerator":
        """加载 MinimalQwen 并构造生成器。

        EOS 刻意从 **HF 模型**解析：MinimalQwen 没有 ``generation_config``，
        若退回 tokenizer 会拿到 ``<|endoftext|>``(151643)，而 Qwen2.5 实际
        用 ``<|im_end|>``(151645) 收尾——终止符错了，生成就会一路跑到
        max_tokens，与无缓存链的对照也会失去意义。
        """
        loaded = load_minimal_from_hf(cfg)
        if loaded.tokenizer is None:  # 防御：load_minimal_from_hf 应始终带回
            raise RuntimeError("load_minimal_from_hf 未返回 tokenizer，无法构造生成器")
        eos_ids = resolve_eos_ids(loaded.hf_model, loaded.tokenizer)
        return cls(loaded.minimal, loaded.tokenizer, cfg, eos_ids)

    @property
    def device(self) -> torch.device:
        # 取自模型参数的实际所在设备（源头是 EngineConfig），见补充条款 A1
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def new_cache(self, max_seq_len: int) -> ContiguousKVCache:
        """按模型形状分配一份连续缓存。"""
        num_layers, num_kv_heads, head_dim = self._kv_dims
        cfg = KVCacheConfig(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            dtype=self.dtype,
            device=self.device,
        )
        return ContiguousKVCache(cfg)

    def generate(
        self,
        prompt: str,
        params: Optional[SamplingParams] = None,
        use_cache: bool = True,
        max_seq_len: Optional[int] = None,
    ) -> CachedGenerationOutput:
        """生成一次补全。

        Args:
            prompt: 用户输入文本。
            params: 采样参数；None 时退化为 greedy + ``cfg.max_new_tokens``。
            use_cache: False 时每步重算整条序列（消融基线，语义与
                ``ManualGenerator`` 一致，但跑在同一个 MinimalQwen 上）。
            max_seq_len: 缓存容量；None 时按 ``prompt + max_tokens`` 分配
                （多留 0 个余量刚好够用：最后一个 token 只需要被 forward
                一次就结束，不需要再给它腾位置）。
        """
        if params is None:
            params = SamplingParams(
                max_tokens=self.cfg.max_new_tokens, temperature=0.0
            )

        generator: Optional[torch.Generator] = None
        if params.seed is not None and not params.is_greedy:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(params.seed)

        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        prompt_tokens = int(input_ids.shape[1])

        capacity = max_seq_len if max_seq_len is not None else prompt_tokens + params.max_tokens
        if capacity < prompt_tokens:
            raise ValueError(
                f"缓存容量 {capacity} 装不下 prompt（{prompt_tokens} tokens）"
            )

        # 缓存必须在进入 inference_mode 之前分配：模式内新建的张量是
        # inference tensor，出了上下文再原地写会直接抛 RuntimeError
        cache = self.new_cache(capacity) if use_cache else None
        caches = cache.layer_caches if cache is not None else None

        generated: list[int] = []
        finish_reason = "length"
        cached_tokens = prompt_tokens if use_cache else 0
        prefill_latency = 0.0
        decode_latency = 0.0

        start = time.perf_counter()
        with torch.inference_mode():  # 推理路径禁用 autograd，省内存也防误用
            # ---- prefill：一次性处理整个 prompt，同时把 K/V 落进缓存 ----
            t0 = time.perf_counter()
            logits = self.model(input_ids, kv_caches=caches, write_pos=0)
            prefill_latency = time.perf_counter() - t0
            next_id = self.sampler.sample(logits[0, -1], params, generator)

            # ---- decode：每步只 forward 最新一个 token ----
            running_ids = input_ids
            while len(generated) < params.max_tokens:
                if next_id in self._eos_ids:
                    finish_reason = "eos"
                    # EOS 不 append：与 ManualGenerator / HF 的文本语义一致
                    break
                generated.append(next_id)
                if len(generated) >= params.max_tokens:
                    break

                step_input = torch.tensor([[next_id]], device=self.device)
                t0 = time.perf_counter()
                if use_cache:
                    # 绝对位置 = 已缓存长度；RoPE 依赖它，写 0 会让续写错位
                    position_ids = torch.tensor([[cached_tokens]], device=self.device)
                    logits = self.model(
                        step_input,
                        position_ids=position_ids,
                        kv_caches=caches,
                        write_pos=cached_tokens,
                    )
                    cached_tokens += 1
                else:
                    running_ids = torch.cat([running_ids, step_input], dim=1)
                    logits = self.model(running_ids)
                decode_latency += time.perf_counter() - t0
                next_id = self.sampler.sample(logits[0, -1], params, generator)
        latency = time.perf_counter() - start

        output_tokens = len(generated)
        text = (
            self.tokenizer.decode(generated, skip_special_tokens=True)
            if generated
            else ""
        )
        logger.debug(
            "生成完成: prompt=%d output=%d finish=%s prefill=%.3fs decode=%.3fs",
            prompt_tokens, output_tokens, finish_reason, prefill_latency, decode_latency,
        )
        return CachedGenerationOutput(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            latency_s=latency,
            tokens_per_s=(output_tokens / latency) if latency > 0 else 0.0,
            device=str(self.device),
            dtype=str(self.dtype).replace("torch.", ""),
            prefill_latency_s=prefill_latency,
            decode_latency_s=decode_latency,
            cached_tokens=cached_tokens,
            cache_bytes=cache.nbytes if cache is not None else 0,
        )
