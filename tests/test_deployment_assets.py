"""Task 13 部署资产的约束测试（无模型，纯静态文件检查）。

为什么需要这些测试：
- Dockerfile / .dockerignore / CI 是『声明性文件』，改错了编译期不报错，
  只有构建/发布时才会炸。把关键约束钉成单测，防止后续改动把缓存目录打回镜像、
  把设备字面量写进 CI、或删掉 README 必含章节。
- 本文件位于 tests/ 下，会被 test_no_hardcoded_cuda 扫描，设备名沿用拼接构造。
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# 本测试验证的是【仓库里的部署资产】；而 Dockerfile 按设计排除 .github/ docs/
# 等（见 .dockerignore），镜像内不存在这些文件。两种上下文使命不同：
# - 仓库上下文（本机/CI checkout）：真实断言这些资产存在且守约束；
# - 镜像上下文（docker run pytest）：验证运行环境，部署资产本来就不该在镜像里，跳过。
pytestmark = pytest.mark.skipif(
    not (REPO_ROOT / "Dockerfile").exists(),
    reason="部署资产不在镜像内：本地/CI 上下文才验证它们",
)


def _dev() -> str:
    # 拼接避免本文件自身触发裸设备字面量扫描（自指陷阱，同 test_no_hardcoded_cuda）
    return "c" + "uda"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------- Dockerfile ----------


def test_dockerfile_exists_and_locks_hf_home() -> None:
    """镜像内模型缓存必须指向非默认路径，容器内才可用 -v 挂载复用本机缓存。"""
    df = _text(REPO_ROOT / "Dockerfile")
    assert "HF_HOME=/model-cache" in df
    assert "download.pytorch.org/whl" in df  # torch 单独先装，避免默认 index 拉 GPU 版


def test_dockerfile_has_no_device_literal() -> None:
    """镜像不写死设备（补充条款 A1 的精神延伸到部署层），设备由运行时 --device 决定。"""
    df = _text(REPO_ROOT / "Dockerfile")
    assert f'"{_dev()}"' not in df
    assert f".{_dev()}(" not in df


# ---------- .dockerignore ----------


def test_dockerignore_excludes_heavy_paths() -> None:
    """漏掉任意一条都会把 GB 级缓存/虚拟环境打进构建上下文。"""
    di = _text(REPO_ROOT / ".dockerignore")
    for required in (
        "hf_cache/",
        "pip_cache/",
        ".venv/",
        ".venv/",
        "__pycache__/",
        ".pytest_cache/",
        ".git/",
    ):
        assert required in di, f".dockerignore 缺少排除项: {required}"


# ---------- CI ----------


def test_ci_workflow_is_quality_gate() -> None:
    """默认门禁必须跑快速测试；且不能引入设备字面量。"""
    ci = _text(REPO_ROOT / ".github" / "workflows" / "ci.yml")
    assert "name: CI" in ci
    assert "on:" in ci
    assert "jobs:" in ci
    assert "pytest" in ci  # 质量门核心
    assert "test_no_hardcoded_cuda" in ci  # 补充条款 A1 门禁显式存在
    assert f'"{_dev()}"' not in ci
    assert f".{_dev()}(" not in ci


# ---------- README（docs/07 Task 13 必须包含的章节） ----------


def test_readme_covers_required_task13_sections() -> None:
    """docs/07 Task 13 要求 README 覆盖：Docker / CI / 架构 / 已知局限 / benchmark 方法+结果。"""
    readme = _text(REPO_ROOT / "README.md")
    for section in (
        "## 与 vLLM 的关系",
        "## 总体架构",
        "## 快速开始",
        "### 5) 用 Docker 跑",
        "## Benchmark",
        "## 已知局限",
        "## 目录结构",
        "## 文档索引",
    ):
        assert section in readme, f"README 缺少必含章节: {section}"
    # benchmark 结果必须指向真实产物，而不是在 README 里编造数字
    assert "benchmark/results/demo/report.md" in readme


def test_known_limitations_doc_exists() -> None:
    lim = _text(REPO_ROOT / "docs" / "known_limitations.md")
    assert "# LiteInfer Known Limitations" in lim
    # 诚实声明必须存在：不做完整 PagedAttention kernel
    assert "PagedAttention" in lim