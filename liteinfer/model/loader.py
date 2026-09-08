"""模型加载器。

职责边界：只负责"把 tokenizer 和模型按 EngineConfig 的声明加载到指定
设备"，不做任何推理逻辑。推理入口在 baseline.py（Task 01）和未来的
ModelRunner（Task 08）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from liteinfer.config import EngineConfig
from liteinfer.device import get_device, resolve_dtype

logger = logging.getLogger("liteinfer.model.loader")


@dataclass
class LoadedModel:
    """一次加载的完整产物。"""

    model: Any  # transformers PreTrainedModel；不写死类型以免绑死 transformers 版本
    tokenizer: Any
    config: EngineConfig
    device: torch.device  # 实际生效的设备（解析后的 torch.device）
    dtype: torch.dtype  # 实际生效的 dtype（经过 CPU 守卫后的）


def _from_pretrained(auto_cls: Any, cfg: EngineConfig, dtype: torch.dtype, cache_dir: str) -> Any:
    """兼容 transformers 新旧版本的 dtype 参数名。

    transformers 4.56 起 ``torch_dtype`` 改名为 ``dtype``；不能写死任何一个，
    否则换一个 transformers 版本就直接 TypeError。用 try/except 探测而非
    inspect 签名：签名字段可能是 **kwargs 隐藏的，探测比反射可靠。
    """
    common = dict(
        cache_dir=cache_dir,
        local_files_only=cfg.local_files_only,
        trust_remote_code=cfg.trust_remote_code,
    )
    try:
        return auto_cls.from_pretrained(cfg.model_id, dtype=dtype, **common)
    except TypeError:
        return auto_cls.from_pretrained(cfg.model_id, torch_dtype=dtype, **common)


def load_model_and_tokenizer(cfg: EngineConfig) -> LoadedModel:
    """按配置加载模型与 tokenizer。

    顺序刻意安排为：先解析 device/dtype，再下载/加载 —— 这样路径不对、
    dtype 非法这类配置错误在任何网络请求之前就暴露。
    """
    device = get_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype, device)
    # 对齐 huggingface_hub 的规范布局：<HF_HOME>/hub。
    # 直接把 hf_cache 根目录传给 cache_dir 会在根下另建一份 models--*，
    # 同一个模型在磁盘上存两份（本机实测各约 1GB）
    cache_dir = str(cfg.resolved_hf_cache_dir() / "hub")

    logger.info("加载 %s -> device=%s dtype=%s cache=%s",
                cfg.model_id, device, dtype, cache_dir)

    tokenizer = _load_tokenizer(cfg, cache_dir)
    model = _from_pretrained(AutoModelForCausalLM, cfg, dtype, cache_dir)
    # 全仓库唯一的设备迁移入口：device 一定来自 EngineConfig，见补充条款 A1
    model.to(device)
    model.eval()  # 推理引擎永远不需要 dropout/BN 的训练态行为
    logger.info("模型加载完成，参数量 %.2fB", sum(p.numel() for p in model.parameters()) / 1e9)
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        config=cfg,
        device=device,
        dtype=dtype,
    )


def _load_tokenizer(cfg: EngineConfig, cache_dir: str) -> Any:
    """加载 tokenizer，并在 fast 后端构建失败时回退到 slow（tiktoken 后端）。

    原因：本机 transformers 5.14.1 + tokenizers 0.22.2 存在 Rust 后端构建 bug，
    ``AutoTokenizer.from_pretrained`` 默认走 fast 会抛
    ``Couldn't instantiate the backend tokenizer``。该 bug 会连带让仓库内**所有**
    model 标记的测试无法加载模型。CPU 上 fast/slow tokenizer 的编码结果一致，
    回退不影响正确性，只影响速度（tokenizer 不是推理热路径瓶颈）。
    先试 fast、失败再回退：在正常的开发环境里仍优先用 fast。
    """
    common = dict(
        cache_dir=cache_dir,
        local_files_only=cfg.local_files_only,
        trust_remote_code=cfg.trust_remote_code,
    )
    try:
        return AutoTokenizer.from_pretrained(cfg.model_id, **common)
    except Exception as exc:  # fast 后端构建失败（环境相关，非配置错误）
        logger.warning(
            "fast tokenizer 构建失败，回退 use_fast=False: %s", exc
        )
        return AutoTokenizer.from_pretrained(cfg.model_id, use_fast=False, **common)
