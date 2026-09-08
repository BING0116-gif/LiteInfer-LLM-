"""LiteInfer: 单卡高吞吐 LLM 推理与服务引擎。

公开接口采用惰性导入：`import liteinfer` 本身不加载 torch/transformers，
保证快速测试与工具链（ruff 等）不受模型栈导入时间拖累。
"""

__version__ = "0.1.0"

__all__ = [
    "EngineConfig",
    "parse_dtype",
    "default_hf_cache_dir",
    "get_device",
    "resolve_dtype",
    "peak_memory_mb",
    "process_rss_mb",
    "SamplingParams",
    "Sampler",
    "ManualGenerator",
]

_CONFIG_NAMES = {"EngineConfig", "parse_dtype", "default_hf_cache_dir"}
_DEVICE_NAMES = {"get_device", "resolve_dtype", "peak_memory_mb", "process_rss_mb"}
_SAMPLING_NAMES = {"SamplingParams", "Sampler"}
_GENERATOR_NAMES = {"ManualGenerator"}


def __getattr__(name: str):
    if name in _CONFIG_NAMES:
        from liteinfer import config as _config

        return getattr(_config, name)
    if name in _DEVICE_NAMES:
        from liteinfer import device as _device

        return getattr(_device, name)
    if name in _SAMPLING_NAMES:
        from liteinfer import sampling as _sampling

        return getattr(_sampling, name)
    if name in _GENERATOR_NAMES:
        from liteinfer.model.generator import ManualGenerator

        return ManualGenerator
    raise AttributeError(f"module 'liteinfer' has no attribute {name!r}")
