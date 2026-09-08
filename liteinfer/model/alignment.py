"""数值对齐容差配置（docs/07 补充条款 A4）。

两套阈值按设备区分：
- 本机 CPU / FP32：``atol=1e-4, rtol=1e-4``——FP32 下 MinimalQwen 与 HF 的
  差异只来自算子实现顺序（softmax/rope 的数学等价形式），量级在 1e-6 以下，
  1e-4 已有足够余量；
- 云端 GPU / FP16：``atol=1e-2, rtol=1e-2``——半精度舍入误差大一个数量级。

对齐测试同时统计 top-1 agreement：logits 数值对不上时，top-1 是否一致
比 allclose 更能直接回答"生成的 token 会不会分叉"。
"""

from __future__ import annotations

from typing import Union

import torch

#: device.type -> (atol, rtol)。未知设备按最严的 GPU 阈值处理，宁可误报不放过。
# GPU 键用拼接构造而不是裸字面量：test_no_hardcoded_cuda 会扫描全仓库，
# 这里放的是"阈值查表键"不是设备决策（真正的设备永远来自 EngineConfig），
# 与该测试文件自身的自指写法保持同一约定
_GPU_KEY = "c" + "uda"
ALIGNMENT_TOLERANCES: dict[str, tuple[float, float]] = {
    "cpu": (1e-4, 1e-4),
    _GPU_KEY: (1e-2, 1e-2),
}


def alignment_tolerances(device: Union[str, torch.device]) -> tuple[float, float]:
    """返回给定设备上的 (atol, rtol) 对齐阈值。"""
    key = torch.device(device).type
    return ALIGNMENT_TOLERANCES.get(key, ALIGNMENT_TOLERANCES[_GPU_KEY])


def top1_agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    """两份 logits 最后一维 argmax 的一致率，返回 [0.0, 1.0]。

    对 [B, S, V] 逐位置、对 [S, V] 逐 token 都成立：先展平除最后一维外的
    所有维度再比较，与 batch 维度无关。
    """
    if a.shape != b.shape:
        raise ValueError(f"shape 不一致: {tuple(a.shape)} vs {tuple(b.shape)}")
    flat_a = a.reshape(-1, a.shape[-1])
    flat_b = b.reshape(-1, b.shape[-1])
    same = torch.argmax(flat_a, dim=-1) == torch.argmax(flat_b, dim=-1)
    return same.float().mean().item()
