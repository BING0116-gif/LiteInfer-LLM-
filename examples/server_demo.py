"""Task 09 端到端示例：用 **OpenAI Python Client** 调用 LiteInfer。

这是 docs/07 Task 09 的硬验收（"OpenAI Python Client 可直接调用" +
"Client disconnect 后 KV 正确回收"）的可执行版本：

1. 后台线程起一个真实的 uvicorn 服务（Qwen2.5-0.5B，CPU FP32）；
2. 用 ``openai.OpenAI(base_url=".../v1")`` 依次调用
   - 非流式 ``/v1/completions``
   - 流式 ``/v1/completions``（逐 token 打印）
   - 流式 ``/v1/chat/completions``
3. 流式读到第 2 个 chunk 就断开，再查 ``/health``，确认 ``kv_blocks_used`` 归零。

运行（仓库根目录）：

    set "HF_HOME=D:/LiteInfer/hf_cache"
    set PYTHONPATH=d:/LiteInfer
    python examples/server_demo.py --max-tokens 8
"""

from __future__ import annotations

import argparse
import socket
import threading
import time

import httpx
import torch

from liteinfer import EngineConfig
from liteinfer.scheduler.config import SchedulerConfig
from liteinfer.server import create_app

PROMPT = "The capital of France is"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _start_server(cfg: EngineConfig, port: int):
    """在后台线程起 uvicorn，返回 (server, thread)。"""
    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(
            create_app(cfg=cfg), host="127.0.0.1", port=port, log_level="warning"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(600):  # 加载 0.5B 模型需要一点时间
        if server.started:
            return server, thread
        time.sleep(0.1)
    raise RuntimeError("uvicorn 未能启动")


def _wait_health(base_url: str, timeout: float = 5.0) -> dict:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        response = httpx.get(f"{base_url}/health", timeout=5.0)
        if response.status_code == 200:
            return response.json()
        time.sleep(0.1)
    raise RuntimeError("服务未就绪")


def _run_client(port: int, max_tokens: int) -> bool:
    from openai import OpenAI

    base_url = f"http://127.0.0.1:{port}/v1"
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    model = EngineConfig.model_id
    ok = True

    print("[1] 非流式 /v1/completions")
    response = client.completions.create(
        model=model, prompt=PROMPT, max_tokens=max_tokens, temperature=0.0
    )
    print(f"    text={response.choices[0].text!r} "
          f"finish={response.choices[0].finish_reason} "
          f"usage={response.usage.prompt_tokens}+{response.usage.completion_tokens}")
    ok = ok and bool(response.choices[0].text)

    print("[2] 流式 /v1/completions（逐 token 打印）")
    streamed: list[str] = []
    for chunk in client.completions.create(
        model=model, prompt=PROMPT, max_tokens=max_tokens, temperature=0.0, stream=True
    ):
        delta = chunk.choices[0].text
        streamed.append(delta)
        print(f"    +{delta!r}", flush=True)
    stream_text = "".join(streamed)
    ok = ok and stream_text == response.choices[0].text
    print(f"    拼接结果与非流式一致: {stream_text == response.choices[0].text}")

    print("[3] 流式 /v1/chat/completions")
    chat_parts: list[str] = []
    for chunk in client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=max_tokens,
        temperature=0.0,
        stream=True,
    ):
        delta = chunk.choices[0].delta
        if delta.role:
            print(f"    role={delta.role}", flush=True)
        if delta.content:
            chat_parts.append(delta.content)
            print(f"    +{delta.content!r}", flush=True)
    print(f"    chat 输出: {''.join(chat_parts)!r}")

    print("[4] 客户端断连 -> 服务端回收 KV 块")
    # with_streaming_response 的 context manager 退出时会真正关闭 HTTP 连接，
    # 这是 openai SDK 里唯一能 deterministic 地"读到一半就断开"的写法
    with client.completions.with_streaming_response.create(
        model=model, prompt=PROMPT, max_tokens=64, temperature=0.0, stream=True
    ) as response:
        for index, chunk in enumerate(response.parse()):
            print(f"    +{chunk.choices[0].text!r}", flush=True)
            if index >= 1:
                break
    # 离开 with 即断开；等服务端把块还回来
    blocks_used = None
    for _ in range(100):
        health = httpx.get(f"http://127.0.0.1:{port}/health", timeout=5.0).json()
        blocks_used = health["kv_blocks_used"]
        if blocks_used == 0:
            break
        time.sleep(0.1)
    print(f"    断连后 kv_blocks_used={blocks_used} "
          f"(total={health['kv_blocks_total']}) -> {'OK' if blocks_used == 0 else 'FAIL'}")
    return ok and blocks_used == 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Task 09：用 OpenAI Python Client 调用 LiteInfer 服务"
    )
    parser.add_argument("--max-tokens", type=int, default=8, help="每个请求最多生成多少 token")
    parser.add_argument("--max-num-seqs", type=int, default=4, help="并发序列上限")
    args = parser.parse_args()

    cfg = EngineConfig(
        device="cpu",
        dtype=torch.float32,  # CPU 必须 FP32（补充条款 A2）
        max_new_tokens=args.max_tokens,
        scheduler=SchedulerConfig(
            max_num_seqs=args.max_num_seqs, max_num_batched_tokens=2048
        ),
    )
    port = _free_port()
    print(f"启动服务: 127.0.0.1:{port}  model={cfg.model_id}  device={cfg.device}")
    server, thread = _start_server(cfg, port)
    try:
        health = _wait_health(f"http://127.0.0.1:{port}", timeout=60.0)
        print(f"服务就绪: device={health['device']} dtype={health['dtype']} "
              f"kv_blocks_total={health['kv_blocks_total']}")
        ok = _run_client(port, args.max_tokens)
    finally:
        server.should_exit = True
        thread.join(timeout=15)

    print(f"\n[final] OpenAI Client 调用 + 断连回收: {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
