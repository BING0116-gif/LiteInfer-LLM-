"""Task 03：Minimal Qwen Decoder——从零实现的 Qwen2 前向计算图。

模块命名刻意与 HF ``Qwen2ForCausalLM`` 的 state_dict 键保持一致
（``model.layers.{i}.self_attn.q_proj`` 等），这样 ``weights.py`` 的权重
重映射只需剥离 ``model.`` 前缀，``load_state_dict(strict=True)`` 就能
保证 checkpoint 里的每一个权重都被消费，杜绝"加载了但没用上"的静默错误。

本阶段只做 forward（无 KV Cache、无采样），生产生成链仍是 Task 02 的
ManualGenerator；Task 04 的 KV Cache 将复用这里的 attention/layer 结构。
"""

from liteinfer.model.minimal.model import (
    MinimalQwenForCausalLM,
    MinimalQwenModel,
    build_causal_mask,
)
from liteinfer.model.minimal.weights import MinimalLoaded, load_minimal_from_hf

__all__ = [
    "MinimalQwenForCausalLM",
    "MinimalQwenModel",
    "build_causal_mask",
    "MinimalLoaded",
    "load_minimal_from_hf",
]
