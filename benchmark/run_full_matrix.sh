#!/usr/bin/env bash
# Task 12 云端一键脚本：一轮跑完全矩阵（docs/07 Task 12 上云执行约定）。
#
# 用法（云端 Linux，仓库根目录）：
#   LITEINFER_DEVICE=cuda LITEINFER_DTYPE=float16 bash benchmark/run_full_matrix.sh
#   # T4/P100 等不支持 bf16 的卡务必用 float16（补充条款 A2）
#
# 约定：
#   - 设备/精度全部走环境变量，脚本不硬编码；video memory 指标无 GPU 时脚本
#     仍可跑（结果里为 N/A）。
#   - HF_HOME 默认仓库内 hf_cache，避免模型下到系统盘；云端先 export 可覆盖。
#   - 产物（results.csv / env.json / charts/*.png / report.md）写在
#     benchmark/results/<run>/ 下，跑完立即 git add + commit（云端随时被回收）。
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

DEVICE="${LITEINFER_DEVICE:?请设置 LITEINFER_DEVICE（如 cuda）}"
DTYPE="${LITEINFER_DTYPE:-float16}"
MODEL_ID="${LITEINFER_MODEL_ID:-Qwen/Qwen2.5-0.5B}"

TAG="${BENCH_TAG:-gpu-full}"
OUT="benchmark/results/$TAG"

echo "==> 全矩阵：device=$DEVICE dtype=$DTYPE model=$MODEL_ID out=$OUT"
python benchmark/liteinfer_benchmark.py \
    --device "$DEVICE" --dtype "$DTYPE" --model-id "$MODEL_ID" \
    --workloads decode prefill typical sp \
    --concurrency 1 2 4 8 16 32 \
    --engines hf nokv kv batch prefix \
    --outdir "$OUT"

echo "==> 图表"
python benchmark/benchmark_charts.py \
    --csv "$OUT/results.csv" --env "$OUT/env.json" --outdir "$OUT/charts"

echo "==> 报告"
python benchmark/benchmark_report.py \
    --csv "$OUT/results.csv" --env "$OUT/env.json" \
    --charts-dir "$OUT/charts" --out "$OUT/report.md"

echo "==> 完成。产物："
ls -la "$OUT"
git add "$OUT" benchmark 2>/dev/null || true
git commit -m "benchmark(Task 12): $TAG full matrix results (device=$DEVICE, dtype=$DTYPE)" 2>/dev/null || true
echo "==> 已提交（可用 git push 同步回本机）"