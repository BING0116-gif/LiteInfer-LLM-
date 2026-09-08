"""LiteInfer OpenAI 兼容服务的启动入口。

    python -m liteinfer.server.main --port 8000

设备与 dtype 一律来自命令行 / ``EngineConfig``（补充条款 A1/A2）：本文件里没有任何
设备字面量，上云只改 ``--device`` / ``--dtype``。
"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from liteinfer.config import EngineConfig, parse_dtype
from liteinfer.scheduler.config import SchedulerConfig
from liteinfer.server.app import create_app


def build_config(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="liteinfer-server", description="启动 LiteInfer OpenAI 兼容服务"
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--model-id", default=EngineConfig.model_id, help="HF 模型 id")
    parser.add_argument(
        "--device",
        default=EngineConfig.device,
        help="运行设备；无 NVIDIA GPU 的开发机保持默认 cpu",
    )
    parser.add_argument(
        "--dtype",
        default="fp32",
        help="计算 dtype：CPU 用 fp32；GPU 用 fp16（T4 / P100 不支持 bf16）",
    )
    parser.add_argument("--max-tokens", type=int, default=64, help="默认最大生成 token 数")
    parser.add_argument("--max-num-seqs", type=int, default=16, help="并发序列上限")
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=2048, help="单步 token 预算"
    )
    parser.add_argument("--block-size", type=int, default=16, help="Paged KV 物理块大小")
    parser.add_argument("--log-level", default="info", help="uvicorn 日志级别")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_config(argv)
    cfg = EngineConfig(
        model_id=args.model_id,
        device=args.device,
        dtype=parse_dtype(args.dtype),
        max_new_tokens=args.max_tokens,
        scheduler=SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
        ),
        block_size=args.block_size,
    )
    app = create_app(cfg=cfg)

    import uvicorn  # 延迟导入：只在本入口真正需要时才依赖 Web 栈

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
