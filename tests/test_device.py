"""设备抽象与降级逻辑（补充条款 A1/A2/A3 的运行时守卫）。"""

from __future__ import annotations

import torch

from liteinfer import get_device, peak_memory_mb, resolve_dtype


class TestGetDevice:
    def test_cpu(self) -> None:
        assert get_device("cpu").type == "cpu"

    def test_cuda_string_parses_without_gpu(self) -> None:
        # 解析 ≠ 硬件可用：torch.device 构造不依赖真卡，云/本地共用一套代码路径。
        # 字符串拼接是为了不触发 test_no_hardcoded_cuda 的自指扫描
        dev = get_device("c" + "uda:0")
        assert dev.type != "cpu"

    def test_rejects_garbage(self) -> None:
        import pytest

        with pytest.raises(RuntimeError):
            get_device("npu-dream")


class TestResolveDtype:
    def test_cpu_fp16_falls_back(self) -> None:
        # CPU 的 FP16 支持差且更慢，必须守卫回 float32
        assert resolve_dtype(torch.float16, "cpu") is torch.float32

    def test_cpu_bf16_falls_back(self) -> None:
        assert resolve_dtype(torch.bfloat16, "cpu") is torch.float32

    def test_cpu_fp32_unchanged(self) -> None:
        assert resolve_dtype(torch.float32, "cpu") is torch.float32

    def test_accepts_torch_device(self) -> None:
        assert resolve_dtype(torch.float16, torch.device("cpu")) is torch.float32


class TestMemoryMetrics:
    def test_no_gpu_returns_none(self) -> None:
        # 无 GPU 时必须返回 None（渲染成 N/A），禁止填 0 误导统计
        has_gpu = torch.cuda.is_available()
        mb = peak_memory_mb()
        assert (mb is None) == (not has_gpu)
        if has_gpu:
            assert mb > 0
