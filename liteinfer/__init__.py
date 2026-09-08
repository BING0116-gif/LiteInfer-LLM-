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
    "MinimalQwenForCausalLM",
    "load_minimal_from_hf",
    "alignment_tolerances",
    "KVCacheConfig",
    "ContiguousKVCache",
    "PagedKVCache",
    "CachedGenerator",
    "EngineCore",
    "Request",
    "RequestStatus",
    "RequestRegistry",
    "Scheduler",
    "SchedulerConfig",
    "ModelRunner",
    "PagedLayerCache",
]

_CONFIG_NAMES = {"EngineConfig", "parse_dtype", "default_hf_cache_dir"}
_DEVICE_NAMES = {"get_device", "resolve_dtype", "peak_memory_mb", "process_rss_mb"}
_SAMPLING_NAMES = {"SamplingParams", "Sampler"}
_GENERATOR_NAMES = {"ManualGenerator"}
_MINIMAL_NAMES = {"MinimalQwenForCausalLM", "load_minimal_from_hf"}
_ALIGNMENT_NAMES = {"alignment_tolerances"}
_CACHE_NAMES = {"KVCacheConfig", "ContiguousKVCache", "PagedKVCache"}
_CACHED_GENERATOR_NAMES = {"CachedGenerator"}
_ENGINE_NAMES = {"EngineCore", "Request", "RequestStatus", "RequestRegistry"}
_SCHEDULER_NAMES = {"Scheduler", "SchedulerConfig"}
_RUNNER_NAMES = {"ModelRunner", "PagedLayerCache"}


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
    if name in _MINIMAL_NAMES:
        from liteinfer.model import minimal as _minimal

        return getattr(_minimal, name)
    if name in _ALIGNMENT_NAMES:
        from liteinfer.model import alignment as _alignment

        return getattr(_alignment, name)
    if name in _CACHE_NAMES:
        from liteinfer import cache as _cache

        return getattr(_cache, name)
    if name in _CACHED_GENERATOR_NAMES:
        from liteinfer.model.cached_generator import CachedGenerator

        return CachedGenerator
    if name in _ENGINE_NAMES:
        from liteinfer.engine import (
            EngineCore,
            Request,
            RequestStatus,
            RequestRegistry,
        )

        return {
            "EngineCore": EngineCore,
            "Request": Request,
            "RequestStatus": RequestStatus,
            "RequestRegistry": RequestRegistry,
        }[name]
    if name in _SCHEDULER_NAMES:
        from liteinfer.scheduler import Scheduler, SchedulerConfig

        return {
            "Scheduler": Scheduler,
            "SchedulerConfig": SchedulerConfig,
        }[name]
    if name in _RUNNER_NAMES:
        from liteinfer.model.runner import ModelRunner, PagedLayerCache

        return {
            "ModelRunner": ModelRunner,
            "PagedLayerCache": PagedLayerCache,
        }[name]
    raise AttributeError(f"module 'liteinfer' has no attribute {name!r}")
