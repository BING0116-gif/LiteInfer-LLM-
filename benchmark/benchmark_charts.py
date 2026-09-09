"""Task 12 图表渲染（docs/05 §11 的图表清单，脚注附实验记录 §8）。

本脚本只消费 ``liteinfer_benchmark`` 落盘的结果文件，不重复测量：
    python benchmark/benchmark_charts.py --csv benchmark/results/<run>/results.csv \
        --env benchmark/results/<run>/env.json --outdir benchmark/results/<run>/charts

每张图底部脚注 = env.json 的关键字段（模型/设备/dtype/torch/线程/时间戳），
满足 docs/05 §8「每张图必须附实验记录」；记录缺少某列则该图直接跳过
（不画"看起来对"的空图）。

图表文字用英文：matplotlib 默认字体无 CJK 字形，中文会画成方框；
报告（benchmark_report）可以用中文，两者语言解耦。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

# Agg：无显示器的 CI / 远端同样能出图（且不会因为缺 GUI 后端挂掉）
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from benchmark.liteinfer_benchmark import CellRow, read_rows_csv  # noqa: E402

#: 需要出图的指标列：(列名, y 轴标签, 是否×1000 转 ms)
METRIC_FIGURES: tuple[tuple[str, str, bool], ...] = (
    ("out_tokens_per_s", "Output tokens/s", False),
    ("ttft_p95", "TTFT p95 (ms)", True),
    ("tpot_p95", "TPOT p95 (ms)", True),
    ("e2e_p50", "E2E p50 (ms)", True),
    ("kv_util_peak", "KV utilization peak", False),
)

#: 生成的图文件名（metrics）由列名推导，命中该表的列用这里的显示名
_WORKLOAD_ORDER: tuple[str, ...] = ("decode", "prefill", "typical", "sp")


def _env_footnote(env: dict[str, Any]) -> str:
    """把实验记录压成一行脚注。字段缺失用 n/a，绝不让缺字段炸掉出图。"""
    def get(*keys: str) -> str:
        for k in keys:
            if k in env and env[k] is not None:
                return str(env[k])
        return "n/a"

    return (
        f"model={get('model_id')} | {get('device')} | {get('dtype')} | "
        f"block_size={get('block_size')} | torch={get('torch_version')} | "
        f"threads={get('torch_threads')} | {get('platform')} | "
        f"ts={get('timestamp_utc')}"
    )


def _grid_figure(workloads: Sequence[str]):
    """2 列网格；workload 多于 4 个时向下增长。"""
    cols = 2
    rows_n = (len(workloads) + cols - 1) // cols
    fig, axes = plt.subplots(
        rows_n, cols, figsize=(4.4 * cols, 3.2 * max(rows_n, 1)), squeeze=False
    )
    for i, wl in enumerate(workloads):
        ax = axes[i // cols][i % cols]
        ax.set_title(f"workload: {wl}")
        ax.grid(True, alpha=0.3)
    for idx in range(len(workloads), rows_n * cols):  # 超出 workload 的空格子隐藏
        axes[idx // cols][idx % cols].axis("off")
    # 共享 x 轴（同一并发数列表），便于逐条对比
    for row in axes:
        for ax in row[1:]:
            ax.sharex(axes[0][0])
    return fig, axes


def _plot_metric(
    rows: Sequence[CellRow],
    env: dict[str, Any],
    metric: str,
    ylabel: str,
    to_ms: bool,
    outdir: Path,
) -> Optional[Path]:
    """渲染一个指标的全 workloads 网格图；无数据返回 None。"""
    present = [r for r in rows if getattr(r, metric) is not None]
    if not present:
        print(f"[skip] 指标 {metric} 全部为 None，跳过图表")
        return None
    workloads = [w for w in _WORKLOAD_ORDER if any(r.workload == w for r in present)]
    if not workloads:
        workloads = sorted({r.workload for r in present})
    engines = sorted({r.engine for r in present})
    # 顺序固定：引擎顺序与矩阵书写顺序一致（hf nokv kv batch prefix）
    engine_order = [e for e in ("hf", "nokv", "kv", "batch", "prefix") if e in engines]
    engine_order += [e for e in engines if e not in engine_order]

    fig, axes = _grid_figure(workloads)
    for wl in workloads:
        ax = axes[workloads.index(wl) // 2][workloads.index(wl) % 2]
        for eng in engine_order:
            pts = [
                (r.concurrency, getattr(r, metric))
                for r in present
                if r.workload == wl and r.engine == eng
            ]
            if not pts:
                continue
            xs, ys = zip(*sorted(pts))
            ys = [y * 1000 if to_ms else y for y in ys]
            ax.plot(xs, ys, marker="o", label=eng)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(ylabel, fontsize=13)
    fig.text(
        0.5, 0.002, _env_footnote(env), ha="center", va="bottom",
        fontsize=6, wrap=True,
    )
    for ax in axes.flat:
        ax.set_xlabel("concurrency")
        if metric == "kv_util_peak":
            ax.set_ylabel(ylabel + " (fraction)")
        else:
            ax.set_ylabel(ylabel)
    name = f"{metric}_vs_concurrency.png"
    out = outdir / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[chart] {out.name}")
    return out


def _plot_prefix_ttft(
    rows: Sequence[CellRow], env: dict[str, Any], outdir: Path
) -> Optional[Path]:
    """shared-prefix workload：batch vs prefix 的 TTFT p50（每组并发一对比柱）。"""
    sp = [r for r in rows if r.workload in ("sp",) and r.ttft_p50 is not None]
    engines = {r.engine for r in sp}
    if not {"batch", "prefix"} <= engines:
        print("[skip] prefix 对比图需要 batch 与 prefix 两驱动的 sp 数据")
        return None
    concs = sorted({r.concurrency for r in sp})
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.35
    xs = list(range(len(concs)))
    for pos, eng in enumerate(("batch", "prefix")):
        ys = []
        for c in concs:
            val = next(
                (r.ttft_p50 * 1000 for r in sp if r.engine == eng and r.concurrency == c),
                None,
            )
            ys.append(val if val is not None else 0.0)
        bars = ax.bar(
            [x + (pos - 0.5) * width for x in xs], ys, width,
            label=f"{eng} TTFT p50",
        )
        if eng == "prefix":  # 柱顶标注命中 token 数
            for bar, c in zip(bars, concs):
                hit = next(
                    (
                        r.prefix_hit_tokens_mean
                        for r in sp
                        if r.engine == eng and r.concurrency == c
                    ),
                    None,
                )
                if hit:
                    ax.annotate(
                        f"hit={hit:.0f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=7,
                    )
    ax.set_xticks(xs)
    ax.set_xticklabels([str(c) for c in concs])
    ax.set_xlabel("concurrency")
    ax.set_ylabel("TTFT p50 (ms)")
    ax.set_title("shared-prefix workload: prefix cache effect")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.text(
        0.5, 0.002, _env_footnote(env), ha="center", va="bottom", fontsize=6, wrap=True
    )
    out = outdir / "sp_prefix_ttft_vs_concurrency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[chart] {out.name}")
    return out


def render_charts(
    csv_path: Path, env_dict: Optional[dict[str, Any]], outdir: Path
) -> list[Path]:
    """渲染全部图表，返回生成的 PNG 路径列表。csv/env 缺失即抛错（fail fast）。"""
    if outdir.exists() is False:
        outdir.mkdir(parents=True)
    env = env_dict or _load_env(csv_path.parent / "env.json")
    rows = read_rows_csv(csv_path)
    made: list[Path] = []
    for metric, ylabel, to_ms in METRIC_FIGURES:
        out = _plot_metric(rows, env, metric, ylabel, to_ms, outdir)
        if out is not None:
            made.append(out)
    out = _plot_prefix_ttft(rows, env, outdir)
    if out is not None:
        made.append(out)
    return made


def _load_env(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="LiteInfer benchmark 图表")
    p.add_argument("--csv", required=True, help="results.csv 路径")
    p.add_argument("--env", default=None, help="env.json 路径（缺省取 csv 同目录）")
    p.add_argument("--outdir", required=True, help="图表输出目录")
    args = p.parse_args(argv)
    env = _load_env(Path(args.env)) if args.env else None
    render_charts(Path(args.csv), env, Path(args.outdir))
    return 0


if __name__ == "__main__":
    sys.exit(main())