"""Task 04 延迟 benchmark：KV Cache vs no-cache（同一模型，同一采样）。

与生产路径分离：本脚本只做测量与输出，不参与 pytest，也不改任何推理逻辑。

用法（仓库根目录）：

    set PYTHONPATH=d:\\LiteInfer
    python benchmark/kv_cache_benchmark.py --device cpu --max-new-tokens 32 64
    python benchmark/kv_cache_benchmark.py --device cpu --max-new-tokens 16 32 64 --repeat 3 --warmup 1 --json result.json

设计要点：
- **两条链跑同一个 MinimalQwen**，只切换 ``use_cache``。拿 HF 模型当基线会把
  "算子实现差异"混进"KV 复用收益"，消融结论就不可信了。
- **warmup + repeat + 中位数**：CPU 上的计时抖动远大于 GPU（频率/调度/后台
  进程），单次测量不足以支撑任何结论。
- **每次都校验输出一致**：性能数字只有在"两条链算的是同一件事"时才有意义，
  不一致直接失败而不是打印一个漂亮的速度比。
- 无 GPU 时显存指标输出 ``N/A (no GPU)``（补充条款 A3：禁止填 0）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass

from liteinfer import CachedGenerator, EngineConfig
from liteinfer.config import DEFAULT_MODEL_ID, parse_dtype
from liteinfer.device import peak_memory_mb
from liteinfer.model.cached_generator import CachedGenerationOutput
from liteinfer.sampling.params import SamplingParams

DEFAULT_PROMPT = "The capital of France is"


@dataclass
class BenchRow:
    """一行结果。时间单位统一为秒，保留原始值不做四舍五入以免误读。"""

    mode: str
    max_new_tokens: int
    prompt_tokens: int
    output_tokens: int
    total_s: float
    min_s: float
    max_s: float
    tokens_per_s: float
    prefill_s: float
    decode_s: float
    cached_tokens: int
    cache_bytes: int
    text: str


def _run_once(
    gen: CachedGenerator, prompt: str, params: SamplingParams, use_cache: bool
) -> CachedGenerationOutput:
    return gen.generate(prompt, params, use_cache=use_cache)


def _measure(
    gen: CachedGenerator,
    prompt: str,
    max_new_tokens: int,
    use_cache: bool,
    warmup: int,
    repeat: int,
) -> tuple:
    """跑 warmup + repeat 次，返回 (总耗时中位数, 最后一次的输出对象)。"""
    params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    for _ in range(warmup):
        _run_once(gen, prompt, params, use_cache)
    latencies: list[float] = []
    out = None
    for _ in range(repeat):
        out = _run_once(gen, prompt, params, use_cache)
        latencies.append(out.latency_s)
    return statistics.median(latencies), out, min(latencies), max(latencies)


def main() -> int:
    parser = argparse.ArgumentParser(description="KV Cache vs no-cache 延迟对照")
    parser.add_argument(
        "--device",
        default="cpu",
        help="设备字符串（默认 cpu）；云端只需传入 GPU 设备名，dtype 另用 --dtype 指定",
    )
    parser.add_argument("--dtype", default="float32", help="dtype 别名，默认 float32")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--max-new-tokens", type=int, nargs="+", default=[16, 32],
        help="生成长度列表（docs/03 建议 32/64/128/256；本机 CPU 约 3 tok/s，"
             "默认取 16/32 以控制在分钟级，长序列留给 Task 12 的 GPU benchmark）",
    )
    parser.add_argument("--repeat", type=int, default=2, help="每种配置重复次数（取中位数）")
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="预热次数（不计入结果）。默认 1：冷启动比热态慢数倍，不预热结论不可信",
    )
    parser.add_argument("--json", default=None, help="结果写入 JSON 文件路径")
    args = parser.parse_args()

    cfg = EngineConfig(
        model_id=args.model_id,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        max_new_tokens=max(args.max_new_tokens),
    )
    gen = CachedGenerator.from_config(cfg)
    print(
        f"model={cfg.model_id} device={gen.device} dtype={gen.dtype} "
        f"repeat={args.repeat} warmup={args.warmup}"
    )
    print(f"prompt={args.prompt!r}\n")

    rows: list[BenchRow] = []
    for n_tokens in args.max_new_tokens:
        no_cache_s, no_cache_out, nc_min, nc_max = _measure(
            gen, args.prompt, n_tokens, use_cache=False,
            warmup=args.warmup, repeat=args.repeat,
        )
        kv_s, kv_out, kv_min, kv_max = _measure(
            gen, args.prompt, n_tokens, use_cache=True,
            warmup=args.warmup, repeat=args.repeat,
        )

        if kv_out.text != no_cache_out.text:
            print(
                f"[FAIL] {n_tokens} tokens：两条链输出不一致，性能对比无意义\n"
                f"  kv      : {kv_out.text!r}\n  no_cache: {no_cache_out.text!r}"
            )
            return 1

        for mode, total, out, lo, hi in (
            ("kv-cache", kv_s, kv_out, kv_min, kv_max),
            ("no-cache", no_cache_s, no_cache_out, nc_min, nc_max),
        ):
            rows.append(
                BenchRow(
                    mode=mode,
                    max_new_tokens=n_tokens,
                    prompt_tokens=out.prompt_tokens,
                    output_tokens=out.output_tokens,
                    total_s=total,
                    min_s=lo,
                    max_s=hi,
                    tokens_per_s=(out.output_tokens / total) if total > 0 else 0.0,
                    prefill_s=out.prefill_latency_s,
                    decode_s=out.decode_latency_s,
                    cached_tokens=out.cached_tokens,
                    cache_bytes=out.cache_bytes,
                    text=out.text,
                )
            )
        speedup = no_cache_s / kv_s if kv_s > 0 else 0.0
        print(
            f"[len={n_tokens:>3}] no-cache {no_cache_s:7.3f}s ({no_cache_out.tokens_per_s:5.2f} tok/s) | "
            f"kv-cache {kv_s:7.3f}s ({kv_out.tokens_per_s:5.2f} tok/s) | "
            f"speedup {speedup:5.2f}x | output identical: OK"
        )

    print("\n明细：")
    header = (
        f"{'mode':<10}{'len':>5}{'prompt':>8}{'out':>5}{'median_s':>10}{'min_s':>8}{'max_s':>8}"
        f"{'tok/s':>8}{'prefill_s':>11}{'decode_s':>10}{'cache_bytes':>13}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.mode:<10}{row.max_new_tokens:>5}{row.prompt_tokens:>8}{row.output_tokens:>5}"
            f"{row.total_s:>10.3f}{row.min_s:>8.3f}{row.max_s:>8.3f}"
            f"{row.tokens_per_s:>8.2f}{row.prefill_s:>11.3f}"
            f"{row.decode_s:>10.3f}{row.cache_bytes:>13}"
        )

    mem = peak_memory_mb()
    print(f"\npeak memory: {mem if mem is not None else 'N/A (no GPU)'}")
    print("[note] CPU 上的加速比仅用于验证 KV 复用逻辑，不能写进简历/汇报。")

    if args.json:
        payload = {
            "model_id": cfg.model_id,
            "device": str(gen.device),
            "dtype": str(gen.dtype),
            "prompt": args.prompt,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "peak_memory_mb": mem,
            "rows": [asdict(r) for r in rows],
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"结果已写入 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
