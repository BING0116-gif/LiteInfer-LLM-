"""设备与 dtype 的统一入口（docs/07 补充条款 A1/A3）。

核心思路：所有"是否在 GPU 上"的判断都收敛到这一个模块，且只通过
``torch.cuda.*`` API 与 ``dev.type == "cpu"`` 分支表达，全仓库因此不需要
出现任何带引号的设备字面量（有专门的单测 ``test_no_hardcoded_cuda`` 把关）。
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch

logger = logging.getLogger("liteinfer.device")


def get_device(device: str) -> torch.device:
    """把配置里的设备字符串解析为 ``torch.device``。

    这里不做可用性检查，只做解析：设备是否真实可用由调用方在
    ``model.to(device)`` 时由 torch 自己报错。原因是把"配置合法"与
    "硬件就绪"两个问题分开，测试里可以自由构造任意 device 而不依赖真卡。
    """
    return torch.device(device)


def resolve_dtype(
    dtype: torch.dtype, device: Union[str, torch.device]
) -> torch.dtype:
    """按设备守卫 dtype：CPU 上的半精度一律回退 float32。

    为什么放在加载前而不是配置层：云端代码是从同一个仓库 clone 的，
    环境变量可能残留 GPU 的 dtype 设置；在这里兜底可以保证
    "device 决定 dtype 合法性"这一条规则永远不会被绕过。
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    if dev.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        logger.warning(
            "CPU 不支持高效的 %s，按补充条款 A2 回退为 float32", dtype
        )
        return torch.float32
    return dtype


def peak_memory_mb() -> Optional[float]:
    """进程级 GPU 峰值显存（MB）。

    无 GPU 时返回 ``None`` —— 调用方必须渲染成 ``N/A (no GPU)``，
    禁止填 0（0 会被下游统计当成真实数据，误导 benchmark 结论）。
    """
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return None


def process_rss_mb() -> float:
    """进程 RSS（MB）。CPU 环境下显存指标的替代观测项，仅供日志参考。"""
    import psutil

    return psutil.Process().memory_info().rss / 1e6
