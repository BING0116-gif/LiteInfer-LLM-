"""集中式配置。

设计约束（docs/07 补充条款 A1/A2）：
- 设备与 dtype 只允许从 ``EngineConfig`` 进入系统，禁止在业务代码中硬编码；
- CPU 必须用 float32（CPU 的 FP16 支持差且更慢），dtype 的守卫在
  ``liteinfer.device.resolve_dtype`` 统一执行，配置层只做声明。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch

from liteinfer.scheduler.config import SchedulerConfig

# Windows 无开发者模式时 symlink 创建会静默失败，留下 0 字节的 snapshot 空壳
# （blob 完整但 config.json 读不到）。设为 1 强制 huggingface_hub 用真实文件副本。
# setdefault：用户显式设置的值优先，不覆盖
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

#: dtype 字符串别名表。配置从环境变量读入时是字符串，在这里归一化。
_DTYPE_ALIASES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B"


def parse_dtype(value: Union[str, torch.dtype]) -> torch.dtype:
    """把字符串别名解析为 ``torch.dtype``；非法值直接抛错，fail fast。

    之所以不静默回退：dtype 写错（比如想写 bf16 写成 bf 16）如果被吞掉，
    后续数值对齐测试会以莫名其妙的方式失败，排查成本远高于启动时报错。
    """
    if isinstance(value, torch.dtype):
        return value
    key = str(value).strip().lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(
            f"未知 dtype: {value!r}，可选值: {sorted(set(_DTYPE_ALIASES))}"
        )
    return _DTYPE_ALIASES[key]


def _repo_root() -> Path:
    # config.py 位于 <root>/liteinfer/config.py
    return Path(__file__).resolve().parent.parent


def default_hf_cache_dir() -> Path:
    """HF 缓存目录的兜底链：环境变量 > D 盘开发机路径 > 仓库内 hf_cache。

    为什么不只依赖 HF_HOME：开发约定是 HF_HOME=D:/LiteInfer/hf_cache，但
    该变量可能没设成（新 shell / 云端 clone 后），不兜底时 huggingface_hub
    会默认写 C 盘用户目录，C 盘只剩约 10GB，会爆。
    """
    env = os.environ.get("HF_HOME")
    if env:
        return Path(env)
    d_drive = Path("D:/LiteInfer/hf_cache")
    if d_drive.parent.exists():
        return d_drive
    return _repo_root() / "hf_cache"


@dataclass(frozen=True)
class EngineConfig:
    """引擎全局配置。后续 Scheduler / BlockPool / Runner 的参数也集中到这里。

    注意 ``dtype`` 存的是 torch.dtype 而不是字符串：dataclass 字段直接携带
    类型信息，避免每个使用方各自 parse 一遍、出现多份真相。
    """

    model_id: str = DEFAULT_MODEL_ID
    # 本机固定 "cpu"；上云时只改这一处。代码中不允许出现其它设备字面量。
    device: str = "cpu"
    # CPU 必须 float32；GPU 用 float16（T4/P100 不支持 bf16，见补充条款 A2）
    dtype: torch.dtype = torch.float32
    hf_cache_dir: Optional[Path] = None  # None -> default_hf_cache_dir() 兜底链
    max_new_tokens: int = 64
    seed: int = 42
    trust_remote_code: bool = False
    local_files_only: bool = False
    # Task 06 调度预算（sequence / token budget），集中到 EngineConfig 统一管理
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def resolved_hf_cache_dir(self) -> Path:
        return self.hf_cache_dir if self.hf_cache_dir else default_hf_cache_dir()

    @classmethod
    def from_env(cls, **overrides: Union[str, int, bool, torch.dtype]) -> "EngineConfig":
        """从环境变量构造配置，显式传入的 ``overrides`` 优先级最高。

        这样云环境只需 export 环境变量，不用改代码；本地调试又可以用
        ``EngineConfig.from_env(device="cpu")`` 这类参数临时覆盖。
        """
        values: dict = {}
        if os.environ.get("LITEINFER_MODEL_ID"):
            values["model_id"] = os.environ["LITEINFER_MODEL_ID"]
        if os.environ.get("LITEINFER_DEVICE"):
            values["device"] = os.environ["LITEINFER_DEVICE"]
        if os.environ.get("LITEINFER_DTYPE"):
            values["dtype"] = parse_dtype(os.environ["LITEINFER_DTYPE"])
        if os.environ.get("LITEINFER_MAX_NEW_TOKENS"):
            values["max_new_tokens"] = int(os.environ["LITEINFER_MAX_NEW_TOKENS"])
        if os.environ.get("HF_HOME"):
            values["hf_cache_dir"] = Path(os.environ["HF_HOME"])
        values.update(overrides)
        return cls(**values)
