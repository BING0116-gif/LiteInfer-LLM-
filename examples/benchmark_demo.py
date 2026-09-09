"""Task 12 最小运行示例：CPU 冒烟全链路（测量 -> 图表 -> 报告）。

    set "HF_HOME=D:/LiteInfer/hf_cache"
    set PYTHONPATH=d:/LiteInfer
    python examples/benchmark_demo.py --smoke

等价于手动分三步跑：liteinfer_benchmark（默认 smoke 口径）-> benchmark_charts
-> benchmark_report；这里是它们的编排壳，不复制任何测量逻辑。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from benchmark import benchmark_charts, benchmark_report, liteinfer_benchmark
from liteinfer.config import DEFAULT_MODEL_ID

DEMO_OUTDIR = "benchmark/results/demo"


def main() -> int:
    p = argparse.ArgumentParser(
        description="LiteInfer benchmark 最小示例（CPU smoke 全链路）"
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument(
        "--smoke", action="store_true", default=True,
        help="冒烟口径（默认开）；--no-smoke 切到 full 口径",
    )
    p.add_argument(
        "--outdir", default=DEMO_OUTDIR, help="结果目录（默认 benchmark/results/demo）"
    )
    p.add_argument(
        "--engines", nargs="+", default=None,
        help="驱动子集（默认 smoke 的 kv/batch/prefix）",
    )
    p.add_argument(
        "--concurrency", nargs="+", type=int, default=None,
        help="并发列表（默认 smoke 的 1 2 4）",
    )
    args = p.parse_args()

    outdir = Path(args.outdir)
    argv = [
        "--device", args.device,
        "--dtype", args.dtype,
        "--model-id", args.model_id,
        "--outdir", str(outdir),
    ]
    if args.smoke:
        argv.append("--smoke")
    if args.engines:
        argv += ["--engines", *args.engines]
    if args.concurrency:
        argv += ["--concurrency"] + [str(c) for c in args.concurrency]

    rc = liteinfer_benchmark.main(argv)
    if rc != 0:
        return rc

    csv_path = outdir / "results.csv"
    env_path = outdir / "env.json"
    charts_dir = outdir / "charts"
    report_path = outdir / "report.md"

    print("\n---- 图表 ----")
    benchmark_charts.render_charts(csv_path, None, charts_dir)
    print("\n---- 报告 ----")
    benchmark_report.build_report(
        csv_path, env_path, charts_dir, report_path
    )
    print(f"\n完成。产物：\n  {csv_path}\n  {env_path}\n  {charts_dir}/"
          f"\n  {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())