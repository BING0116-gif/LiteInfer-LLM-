"""补充条款 A1 验收：仓库 Python 代码中不得出现裸的设备字面量。

扫描范围：liteinfer/ tests/ examples/ benchmark/ 下的所有 .py。
docs/*.md 不扫——规划文档里的示例代码只是说明文字，不是可执行路径。

本测试自身也要通过扫描，所以匹配 pattern 用拼接构造（自指陷阱）：
如果直接在注释或代码里写出"带引号的那个设备名"，这个文件就会成为唯一违规者。
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("liteinfer", "tests", "examples", "benchmark")
EXCLUDED_PARTS = {".venv", "venv", "__pycache__", ".pytest_cache", "hf_cache", "pip_cache"}


def _quoted_pattern() -> str:
    return '"' + "c" + "uda" + '"'


def _method_call_pattern() -> str:
    return "." + "c" + "uda" + "("


def _offending_lines() -> list[str]:
    offenders: list[str] = []
    quoted = _quoted_pattern()
    method_call = _method_call_pattern()
    for name in SCAN_DIRS:
        root = REPO_ROOT / name
        if not root.exists():
            continue
        for py in sorted(root.rglob("*.py")):
            if any(part in EXCLUDED_PARTS for part in py.parts):
                continue
            for lineno, line in enumerate(
                py.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if quoted in line or method_call in line:
                    offenders.append(
                        f"{py.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}"
                    )
    return offenders


def test_no_bare_cuda_literal() -> None:
    offenders = _offending_lines()
    assert not offenders, (
        "发现裸设备字面量（补充条款 A1 禁止），"
        "设备必须统一走 EngineConfig.device：\n" + "\n".join(offenders)
    )
