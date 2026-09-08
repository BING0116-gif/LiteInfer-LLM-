"""HF checkpoint -> MinimalQwen 的权重映射加载器。

策略：不做任何"智能"容错，走 ``load_state_dict(strict=True)``。
重映射规则只有一条——剥离 ``model.`` 前缀（我们的顶层属性名刻意与
HF 一致）。strict 加载的意义：checkpoint 里任何一个权重没被消费、
或者 shape 对不上，都会当场报错，而不是留下"模型能跑但结果悄悄不对"
的最难排查的一类 bug。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from liteinfer.config import EngineConfig
from liteinfer.model.loader import load_model_and_tokenizer
from liteinfer.model.minimal.model import MinimalQwenForCausalLM

logger = logging.getLogger("liteinfer.model.minimal.weights")


@dataclass
class MinimalLoaded:
    """MinimalQwen 及其对齐参照物的打包产物。"""

    minimal: MinimalQwenForCausalLM
    hf_model: Any  # transformers 的 Qwen2ForCausalLM，作为对齐参照保留
    device: torch.device
    dtype: torch.dtype


def _minimal_from_hf_config(
    hf_cfg: Any, hf_state_dict: dict[str, torch.Tensor]
) -> MinimalQwenForCausalLM:
    """从 HFConfig + checkpoint 键集合建模型。

    所有超参 getattr 兜底读取，不假设 0.5B 的具体数字。QK-Norm 不看
    config 而看权重键是否存在（transformers 5.x 的 Qwen2 已移除该模块、
    Qwen2.5 权重里也没有这些键，Qwen3 才有）——结构跟着权重走，
    strict 加载才有明确的语义。
    """
    hidden_size = hf_cfg.hidden_size
    num_heads = hf_cfg.num_attention_heads
    num_kv_heads = hf_cfg.num_key_value_heads
    # head_dim 在 0.5B 上是 64 = 896/14；Qwen2 家族里该值老 config 可能缺省，
    # 缺省时按"hidden 均分给 Q 头"推导
    head_dim = getattr(hf_cfg, "head_dim", None) or hidden_size // num_heads

    rope_scaling = getattr(hf_cfg, "rope_scaling", None) or {}
    # transformers 5.x 对标准 RoPE 也会落盘 rope_scaling={'rope_type': 'default'}
    # （实测 5.14.1 的 Qwen2Config），'default'/缺失都代表无缩放；
    # 其余（linear/dynamic/yarn 等）inv_freq 计算方式不同，宁可显式失败
    rope_type = rope_scaling.get("rope_type", "default")
    if rope_type != "default":
        raise NotImplementedError(
            f"暂不支持 rope_scaling={rope_scaling}，MinimalQwen 只实现了标准 RoPE"
        )

    use_qk_norm = "model.layers.0.self_attn.q_norm.weight" in hf_state_dict

    return MinimalQwenForCausalLM(
        vocab_size=hf_cfg.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=hf_cfg.num_hidden_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=hf_cfg.intermediate_size,
        rope_theta=getattr(hf_cfg, "rope_theta", 1000000.0),
        rms_norm_eps=hf_cfg.rms_norm_eps,
        attention_bias=getattr(hf_cfg, "attention_bias", True),
        use_qk_norm=use_qk_norm,
        mlp_bias=getattr(hf_cfg, "mlp_bias", False),
    )


def _remap_state_dict(hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """HF 键名 -> MinimalQwen 键名。

    我们的顶层结构（``self.model`` + ``self.lm_head``）与 HF
    ``Qwen2ForCausalLM`` 完全镜像，子模块命名也一致，因此键名映射是
    恒等的——这里保留函数是给 tied embedding 补键，并作为未来键名
    漂移时的唯一收口点。tied checkpoint 没有 lm_head.weight 键，
    用 embed_tokens 的权重补上（tied 本来就是同一块内存）。
    """
    remapped: dict[str, torch.Tensor] = dict(hf_state_dict)
    if "lm_head.weight" not in remapped and "model.embed_tokens.weight" in remapped:
        logger.info("检测到 tied word embedding，lm_head 复用 embed_tokens 权重")
        remapped["lm_head.weight"] = remapped["model.embed_tokens.weight"]
    return remapped


def load_minimal_from_hf(cfg: EngineConfig) -> MinimalLoaded:
    """加载 HF 模型并把权重搬进 MinimalQwen。

    复用 Task 01 的 loader：HF 模型的加载/缓存/设备迁移规则只有一份实现，
    MinimalQwen 不另起炉灶，HF 模型同时作为对齐参照物返回。
    """
    loaded = load_model_and_tokenizer(cfg)
    hf_model = loaded.model
    device, dtype = loaded.device, loaded.dtype

    hf_sd = dict(hf_model.state_dict())
    minimal = _minimal_from_hf_config(hf_model.config, hf_sd)

    # 权重统一 cast 到目标 dtype/device：config 声明的精度在此生效，
    # 之后的 forward 路径不再出现任何 dtype/device 决策
    remapped = {
        k: v.to(device=device, dtype=dtype)
        for k, v in _remap_state_dict(hf_sd).items()
    }
    # strict=True：键多键少 shape 不符都报错，这是本模块的核心安全阀
    minimal.load_state_dict(remapped, strict=True)
    minimal.to(device=device)
    minimal.eval()
    logger.info(
        "MinimalQwen 权重加载完成：%d 个张量, device=%s, dtype=%s",
        len(remapped), device, dtype,
    )
    return MinimalLoaded(minimal=minimal, hf_model=hf_model, device=device, dtype=dtype)
