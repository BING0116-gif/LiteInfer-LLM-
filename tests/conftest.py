"""pytest 公共夹具与 marker 声明。

marker 在 pyproject.toml 注册：默认 addopts 排除 model 标记，
保证 `pytest -q` 秒级反馈；需要真实模型时显式 `pytest -m model`。
"""

from __future__ import annotations

import pytest
import torch

from liteinfer import EngineConfig


@pytest.fixture
def cpu_cfg() -> EngineConfig:
    """不触发任何下载的纯配置对象。"""
    return EngineConfig(device="cpu", dtype=torch.float32, max_new_tokens=16)
