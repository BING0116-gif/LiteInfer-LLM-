"""HF Baseline：Task 01 的对齐基准。

重要：这是全项目唯一允许调用 ``model.generate()`` 的地方（docs/07 禁止
事项："可以仅在 Baseline 阶段使用"）。Task 02 的手写生成循环完成后，
本类降级为对齐参照物，不再位于生产路径上 —— 这个身份转变会在
PROGRESS.md 与 Task 02 的设计文档中再次声明。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch

from liteinfer.config import EngineConfig
from liteinfer.model.loader import LoadedModel, load_model_and_tokenizer

logger = logging.getLogger("liteinfer.model.baseline")


@dataclass
class GenerationOutput:
    """一次生成的完整结果与元信息。

    latency/tokens_per_s 在 CPU 上只用于逻辑验证（docs/04 指标降级规则），
    不得进入 README 性能表或简历。
    """

    text: str
    prompt_tokens: int
    output_tokens: int
    latency_s: float
    tokens_per_s: float
    device: str
    dtype: str


class HFBaseline:
    """封装 HF ``generate()`` 的最小推理入口，Task 02 的对齐参照。"""

    def __init__(self, model: Any, tokenizer: Any, cfg: EngineConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg: EngineConfig) -> "HFBaseline":
        loaded: LoadedModel = load_model_and_tokenizer(cfg)
        return cls(loaded.model, loaded.tokenizer, cfg)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        greedy: bool = True,
    ) -> GenerationOutput:
        """生成一次补全。

        Task 01 只实现 greedy（确定性），do_sample/temperature/top-k/top-p
        属于 Task 02 的 Sampling 范畴，这里刻意不做。
        """
        n = max_new_tokens if max_new_tokens is not None else self.cfg.max_new_tokens
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = int(enc["input_ids"].shape[1])
        # Qwen 的 tokenizer 可能没有 pad_token；generate 需要 pad 时用它兜底，
        # 否则 batch>1 的场景会告警甚至错位
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        start = time.perf_counter()
        with torch.inference_mode():  # 推理路径禁用 autograd，省内存也防误用
            out = self.model.generate(
                **enc,
                max_new_tokens=n,
                do_sample=not greedy,
                pad_token_id=pad_token_id,
            )
        latency = time.perf_counter() - start

        # 只解码新增部分：prompt 原文不属于生成结果，混进去会污染对齐测试
        new_ids = out[0, prompt_tokens:]
        output_tokens = int(new_ids.shape[0])
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return GenerationOutput(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
            tokens_per_s=(output_tokens / latency) if latency > 0 else 0.0,
            device=str(self.device),
            dtype=str(self.dtype).replace("torch.", ""),
        )
