"""Task 02：手写自回归生成主链（生产路径，不经过 ``generate()``）。

核心循环：
    forward 全序列 -> 取最后一步 logits -> Sampler 选 token -> append -> 下一轮

本阶段刻意不做 KV Cache（docs/03 模块 2："先不考虑性能，只建立可控推理
主链"）：每步把整个前缀重新 forward 一遍，复杂度 O(n^2)。Task 04 引入
KV Cache 时只替换 forward 策略，Sampler 与本循环骨架保持不变。

HFBaseline（Task 01）自本模块起降级为对齐参照物，退出生产路径。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch

from liteinfer.config import EngineConfig
from liteinfer.model.loader import LoadedModel, load_model_and_tokenizer
from liteinfer.sampling.params import SamplingParams
from liteinfer.sampling.sampler import Sampler

logger = logging.getLogger("liteinfer.model.generator")


@dataclass
class GenerationOutput:
    """一次生成的完整结果与元信息。

    与 baseline.GenerationOutput 字段对齐，另加 ``finish_reason``
    （"eos" / "length"），这是 Task 05 请求状态机要用的语义信号。
    """

    text: str
    prompt_tokens: int
    output_tokens: int
    finish_reason: str
    latency_s: float
    tokens_per_s: float
    device: str
    dtype: str


class ManualGenerator:
    """手写自回归生成循环。

    支持依赖注入（model/tokenizer 从构造函数进入），对齐测试可以与
    HFBaseline 共享同一份加载好的权重，保证差异只来自生成逻辑本身。
    """

    def __init__(self, model: Any, tokenizer: Any, cfg: EngineConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.sampler = Sampler()
        self._eos_ids = self._resolve_eos_ids()

    @classmethod
    def from_config(cls, cfg: EngineConfig) -> "ManualGenerator":
        loaded: LoadedModel = load_model_and_tokenizer(cfg)
        return cls(loaded.model, loaded.tokenizer, cfg)

    def _resolve_eos_ids(self) -> frozenset:
        """对齐 HF 的 EOS 语义：generation_config 优先于 tokenizer。

        Qwen2.5 的生成终止符是 <|im_end|>（151645），记录在
        generation_config.eos_token_id 而非 tokenizer.eos_token_id；
        该字段可能是 int 也可能是 list，两种形态都要归一成集合。
        """
        raw = getattr(self.model.generation_config, "eos_token_id", None)
        if raw is None:
            raw = self.tokenizer.eos_token_id
        if raw is None:
            return frozenset()
        if isinstance(raw, int):
            return frozenset({raw})
        return frozenset(int(x) for x in raw)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def generate(
        self, prompt: str, params: Optional[SamplingParams] = None
    ) -> GenerationOutput:
        """生成一次补全。

        Args:
            prompt: 用户输入文本。
            params: 采样参数；None 时退化为 greedy + ``cfg.max_new_tokens``，
                与 baseline 的默认行为一致，方便对齐测试直接互比。
        """
        if params is None:
            params = SamplingParams(
                max_tokens=self.cfg.max_new_tokens, temperature=0.0
            )

        generator: Optional[torch.Generator] = None
        if params.seed is not None and not params.is_greedy:
            # greedy 不经过随机数路径，建了也用不上，干脆不建
            # device 一律取自模型实际所在设备（源自 EngineConfig），见补充条款 A1
            generator = torch.Generator(device=self.device)
            generator.manual_seed(params.seed)

        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        prompt_tokens = int(input_ids.shape[1])
        generated: list[int] = []
        finish_reason = "length"

        start = time.perf_counter()
        with torch.inference_mode():  # 推理路径禁用 autograd，省内存也防误用
            for _ in range(params.max_tokens):
                # 无 KV Cache：每步全序列 forward。输出 [1, seq, vocab]，
                # 只取最后一步的 logits 参与选 token
                logits = self.model(input_ids).logits
                next_id = self.sampler.sample(logits[0, -1], params, generator)
                if next_id in self._eos_ids:
                    finish_reason = "eos"
                    # EOS 不 append：HF 靠 decode 时 skip_special_tokens 达到
                    # 同样的文本效果，这里在源头就不让它进序列
                    break
                generated.append(next_id)
                input_ids = torch.cat(
                    [
                        input_ids,
                        torch.tensor([[next_id]], device=self.device),
                    ],
                    dim=1,
                )
        latency = time.perf_counter() - start

        output_tokens = len(generated)
        text = self.tokenizer.decode(
            generated, skip_special_tokens=True
        ) if generated else ""
        return GenerationOutput(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            latency_s=latency,
            tokens_per_s=(output_tokens / latency) if latency > 0 else 0.0,
            device=str(self.device),
            dtype=str(self.dtype).replace("torch.", ""),
        )
