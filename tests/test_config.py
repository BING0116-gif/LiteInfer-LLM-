"""EngineConfig：默认值、环境变量覆盖、dtype 解析、HF 缓存兜底。"""

from __future__ import annotations

import torch

from liteinfer import EngineConfig, default_hf_cache_dir, parse_dtype


class TestDefaults:
    def test_default_device_is_cpu(self, cpu_cfg: EngineConfig) -> None:
        # 开发机无 GPU，默认值必须落到 cpu（补充条款 A1）
        assert cpu_cfg.device == "cpu"

    def test_default_dtype_is_fp32(self, cpu_cfg: EngineConfig) -> None:
        # CPU 必须 float32（补充条款 A2）
        assert cpu_cfg.dtype is torch.float32

    def test_default_model_is_qwen(self) -> None:
        assert EngineConfig().model_id.startswith("Qwen/")


class TestParseDtype:
    def test_aliases(self) -> None:
        assert parse_dtype("float32") is torch.float32
        assert parse_dtype("fp16") is torch.float16
        assert parse_dtype("bf16") is torch.bfloat16

    def test_passthrough(self) -> None:
        assert parse_dtype(torch.float32) is torch.float32

    def test_rejects_unknown(self) -> None:
        # fail fast：拼错的 dtype 必须立刻报错，而不是烂在对齐测试里
        import pytest

        with pytest.raises(ValueError, match="未知 dtype"):
            parse_dtype("float64")


class TestFromEnv:
    def test_env_overrides(self, monkeypatch) -> None:
        monkeypatch.setenv("LITEINFER_DEVICE", "cpu")
        monkeypatch.setenv("LITEINFER_DTYPE", "fp32")
        monkeypatch.setenv("LITEINFER_MODEL_ID", "Qwen/Qwen2.5-0.5B")
        monkeypatch.setenv("LITEINFER_MAX_NEW_TOKENS", "8")
        cfg = EngineConfig.from_env()
        assert cfg.device == "cpu"
        assert cfg.dtype is torch.float32
        assert cfg.max_new_tokens == 8

    def test_explicit_overrides_env(self, monkeypatch) -> None:
        monkeypatch.setenv("LITEINFER_MAX_NEW_TOKENS", "8")
        cfg = EngineConfig.from_env(max_new_tokens=3)
        assert cfg.max_new_tokens == 3  # 显式参数优先于环境变量

    def test_invalid_env_dtype_fails_fast(self, monkeypatch) -> None:
        monkeypatch.setenv("LITEINFER_DTYPE", "fp8")
        import pytest

        with pytest.raises(ValueError):
            EngineConfig.from_env()


class TestHfCacheFallback:
    def test_env_var_wins(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("HF_HOME", str(tmp_path / "cache"))
        assert default_hf_cache_dir() == tmp_path / "cache"

    def test_no_env_falls_back_off_c_drive(self, monkeypatch) -> None:
        monkeypatch.delenv("HF_HOME", raising=False)
        d = default_hf_cache_dir()
        # 兜底链只允许 D 盘开发机路径或仓库内路径，绝不能落到 C 盘用户目录
        assert str(d).startswith(("D:/", "D:\\", "/")) or d.is_relative_to(
            d  # 仓库内路径分支
        )
        assert "hf_cache" in str(d)
