# Task 09 设计文档：Async Engine + Streaming API

**运行环境**：CPU（本机无 NVIDIA GPU，全程 `EngineConfig.device="cpu"`、`dtype=float32`）

---

## 1. 源代码

| 文件 | 职责 |
|---|---|
| `liteinfer/engine/async_engine.py` | **新增**。`StreamChunk`、`AsyncStream`、`AsyncEngine`（命令队列 + 后台引擎循环 + 每请求输出队列 + 取消） |
| `liteinfer/server/schemas.py` | **新增**。OpenAI 兼容的 pydantic v2 请求/响应模型 |
| `liteinfer/server/sse.py` | **新增**。SSE 帧编码 / 解析、`[DONE]`、禁用中间层缓冲的响应头 |
| `liteinfer/server/app.py` | **新增**。`create_app`：`/v1/completions`、`/v1/chat/completions`、`/v1/models`、`/health`、`/v1/requests/{id}/cancel` |
| `liteinfer/server/main.py` | **新增**。uvicorn 启动入口（`python -m liteinfer.server.main`） |
| `liteinfer/server/__init__.py` | **新增**。导出 `create_app`（刻意不进 `liteinfer/__init__`，避免 import 拉起 Web 栈） |
| `tests/_fakes.py` | **新增**。测试共用的会写 KV 的假模型/假 tokenizer |
| `tests/test_async_engine.py` | **新增**。12 条快测 + 1 条 model 标记 |
| `tests/test_server_api.py` | **新增**。16 条快测（含真实 uvicorn 上的断连回收） |
| `examples/async_engine_demo.py` | **新增**。asyncio 层并发流式 + 取消 + 与 CachedGenerator 逐字对照 |
| `examples/server_demo.py` | **新增**。起真实服务 + **OpenAI Python Client** 调非流式/流式/chat/断连 |
| `liteinfer/engine/__init__.py` | 修改。导出 `AsyncEngine` / `StreamChunk` |
| `liteinfer/__init__.py` | 修改。惰性导出 `AsyncEngine` / `StreamChunk` |
| `pyproject.toml` | 修改。dev/serving 依赖补 FastAPI 栈；`asyncio_mode = "auto"` |

**`EngineCore` 一行未改** —— 这是本阶段划定的边界。

---

## 2. Unit Tests

### `tests/test_async_engine.py`（12 条快测 + 1 条 model）

| 测试 | 钉住的行为 |
|---|---|
| `test_stream_yields_tokens_then_terminal_chunk` | 顺序、index 递增、仅最后一个 chunk 带 `finish_reason` |
| `test_stream_matches_sync_engine_run` | 流式拼接 == 同步 `EngineCore.run()` 文本 |
| `test_eos_terminal_chunk_carries_no_token` | EOS 那一步 `token_id is None`，文本为空（不吞字、不多吐字） |
| `test_concurrent_streams_stay_independent` | 3 条流 `asyncio.gather` 并发，互不串台 |
| `test_early_close_cancels_and_frees_blocks` | 提前关闭流 → CANCELLED + `num_blocks_used == 0` |
| `test_explicit_abort_while_running` | `abort()` 返回时取消已生效、块已归还 |
| `test_blocks_reclaimed_after_normal_completion` | 正常结束也零泄漏 |
| `test_shutdown_terminates_in_flight_stream` | 关停不会把消费方永久挂起（必收终态 chunk） |
| `test_event_loop_not_blocked_by_forward` | **前向必须卸载到线程**：生成期间 ticker 协程能持续被调度 |
| `test_submit_propagates_validation_error` | prompt 超 token budget 时 fail fast |
| `test_stream_of_unknown_request_raises` | 未知 rid 抛 KeyError |
| `test_start_is_idempotent` | 重复 `start()` 不会起第二个引擎循环 |
| `test_real_model_stream_matches_cached_generator` | **model 标记**：Qwen2.5-0.5B 流式拼接与 CachedGenerator 逐字一致 |

### `tests/test_server_api.py`（16 条快测）

| 测试 | 钉住的行为 |
|---|---|
| `test_health_reports_device_and_stats` | `/health` 如实上报 device/dtype/块池 |
| `test_models_endpoint` | `/v1/models` 的 OpenAI 形状 |
| `test_completion_non_stream` | 文本、finish_reason、usage 三项正确 |
| `test_completion_accepts_prompt_list` | prompt 数组 → 多个 choice，usage 累加 |
| `test_completion_stream_sse` | SSE 帧序列 = 4 个 token 帧 + 1 个收尾帧 + `[DONE]` |
| `test_stream_rejects_prompt_list` | `stream=True` + 数组 → 400 |
| `test_oversized_prompt_returns_400` | 超长 prompt 在响应头发出前就被拒 |
| `test_chat_completion_non_stream` | `message.role == "assistant"` |
| `test_chat_completion_stream_first_chunk_carries_role` | 首个 chunk 必须声明 `role` |
| `test_chat_completion_rejects_empty_messages` | 空 messages → 400 |
| `test_chat_prompt_uses_tokenizer_template` | 有 chat template 时走模板 |
| `test_chat_prompt_fallback_without_template` | 无模板时降级为角色拼接，不让请求失败 |
| `test_cancel_endpoint_cancels_and_frees_blocks` | 取消端点返回时块已归还 |
| `test_cancel_unknown_request_returns_404` | 未知 rid → 404 |
| `test_closing_stream_generator_cancels_request` | 确定性版本：直接 `aclose()` 响应体生成器 |
| `test_client_disconnect_frees_blocks` | **真实 uvicorn + 真 socket**：读到第 2 个 chunk 就断开 → 块归零 + CANCELLED |

---

## 3. Design Document

本文件。

---

## 4. 最小运行示例

```bash
set "HF_HOME=D:/LiteInfer/hf_cache"
set PYTHONPATH=d:/LiteInfer
python examples/async_engine_demo.py --max-tokens 8
python examples/server_demo.py --max-tokens 8
```

`async_engine_demo` 实测（Qwen2.5-0.5B, CPU FP32）：

```text
[并发流式] 3 个请求，每请求的 token 到达时立即打印
  [p0] +' Paris'    [p1] +' __'     [p0] +'.'
  [p1] +'__\n'      [p0] +' It'     [p1] +'A'
  ...
[结束] used_blocks=0 free_blocks=18

[取消] 生成 3 个 token 后断开连接
  status=cancelled 已生成=4 used_blocks=0

[parity] The capital of France is
  async stream : ' Paris. It is the largest city in'
  CachedGen    : ' Paris. It is the largest city in'
  identical    : True
[final] 与 CachedGenerator 全部一致 + 取消后零泄漏: True -> OK
```

`server_demo` 实测（OpenAI Python Client 2.52 打本地 uvicorn）：

```text
服务就绪: device=cpu dtype=float32 kv_blocks_total=36
[1] 非流式 /v1/completions
    text=' Paris. It is the largest city in' finish=length usage=5+8
[2] 流式 /v1/completions（逐 token 打印）
    +' Paris' +'.' +' It' +' is' +' the' +' largest' +' city' +' in' +''
    拼接结果与非流式一致: True
[3] 流式 /v1/chat/completions
    role=assistant  +' salute' ...
[4] 客户端断连 -> 服务端回收 KV 块
    断连后 kv_blocks_used=0 (total=36) -> OK
```

---

## 5. 关键数据结构

### `StreamChunk`

```python
StreamChunk(request_id, token_id, text, finished, finish_reason, index=0)
# text 是"本次增量"，不是累积全文；terminate chunk 的 text 为空串
```

### `AsyncEngine`

```
AsyncEngine
├── _cmd : asyncio.Queue[_Command]        # 外部意图 -> 引擎循环（串行化对 core 的访问）
├── _out : dict[rid, asyncio.Queue]       # 每请求一条输出流（docs/02 §9）
├── _counts : dict[rid, int]              # 已投递 chunk 数 -> StreamChunk.index
├── submit(prompt, params) -> rid         # 投命令 + await future
├── stream(rid) -> AsyncStream
├── generate(prompt, params) -> AsyncStream    # submit + stream
├── abort(rid)        # 等取消生效（HTTP 取消端点）
├── abort_nowait(rid) # 不等（生成器清理路径，见 6.3）
└── _run_loop()       # drain 命令 -> to_thread(core.step) -> 投递 token
```

### 一次流式请求的数据流

```text
POST /v1/completions (stream=True)
  → handler: await engine.submit(prompt, params)   # 校验失败在这里就 400，响应头还没发出
  → StreamingResponse(_completion_sse(...))
       └─ it = engine.stream(rid).__aiter__()
          while True:
              chunk = await it.__anext__()
              yield b"data: {...}\n\n"
          finally:
              await it.aclose()          # 断连时确定性触发清理

引擎侧（另一个协程）：
  _run_loop: drain _cmd → await to_thread(core.step) → _dispatch(results)
                                                        └─ _out[rid].put(chunk)
```

### 线程模型

```text
事件循环线程   : HTTP handler / SSE 生成 / 引擎循环的协程部分
worker 线程    : core.step()（一次前向），同一时刻最多一个
约束           : core 只被引擎循环协程访问（单写者），因此无需加锁
```

---

## 6. 为什么这样设计

### 6.1 单写者纪律：所有对 `EngineCore` 的访问都走命令队列

`EngineCore` 不是线程安全的（registry / scheduler / 块池都是裸 dict）。如果 HTTP handler 直接
`core.submit(...)`，而 worker 线程正在 `core.step()`，就是数据竞争。

`_run_loop` 是唯一访问 core 的协程；命令只在"当前没有 step 在飞"时被 drain（单协程单线程，
天然不会交错）。代价是 `submit` 最多等一个 step，换来的是零竞态、零锁。

### 6.2 阻塞前向必须 `asyncio.to_thread`

`step()` 是一次前向（CPU 上 0.5B 约 0.3s/token）。直接在协程里调会把事件循环焊死，
SSE 一帧都发不出去。`to_thread` 卸载后，每产出一个 token 事件循环就能空出来刷一次响应。
`test_event_loop_not_blocked_by_forward` 用一个 ticker 协程钉住这条性质。

同一时刻只有一个 step 在飞（循环里是 `await` 而非并发派发），所以不需要给 core 加锁。

### 6.3 清理路径用 `abort_nowait`，不用 `abort`

`AsyncStream._iterate` 的 `finally` 里如果 `await` 一个"由引擎循环 resolve 的 future"，
在生成器 finalizer / 事件循环关闭的场景下有可能永远等不到（abort 命令没人消费）。
因此分成两个 API：

- `abort()`：`await` 到生效——HTTP 取消端点和示例用，保证返回时块已归还；
- `abort_nowait()`：只投递命令——生成器清理路径用，"循环下一轮必执行"已足够保证取消发生。

### 6.4 SSE 生成器手写 `__anext__` + `finally: await it.aclose()`

如果写成 `async for chunk in engine.stream(rid)`，外层（SSE）被关闭时，内层流只靠 GC 兜底，
其 `finally` 的执行时机不确定（靠 `loop.call_soon_threadsafe` 调度），"断连是否回收 KV"
就不可验证。显式拿迭代器并 `aclose()`，清理时机确定。

### 6.5 最后一帧：先送 token，再补一个只带 `finish_reason` 的空帧

`EngineCore` 在达到 `max_tokens` 时，把"最后一个 token"和"finished=True"合并在**同一个**
`RequestStepResult` 里。最初的实现对 `finished` 的 chunk 一律 `text=""`，结果最后一个 token
被吞掉（实测 `'345'` 而不是 `'3456'`）。现在按 OpenAI 协议拆成两帧：token 帧 + 只带
`finish_reason` 的空帧；EOS 那一步本来就没有 token（`token_id is None`），单独处理成只发终态帧。

### 6.6 校验必须在响应头发出之前完成

流式端点先 `await engine.submit(...)` 再构造 `StreamingResponse`：prompt 超长等校验错误
要在 200 + `text/event-stream` 发出**之前**变成 400，否则客户端只会看到一个断掉的流。

### 6.7 服务层不出现在 `liteinfer/__init__.py`

导入 `liteinfer.server` 会拉起 FastAPI。快速测试（不依赖模型、也不依赖 Web 栈）不该为此
付出代价，所以服务层用显式 `from liteinfer.server import create_app`。
`AsyncEngine` 本身只依赖标准库 asyncio，因此可以进惰性导出。

---

## 7. Alternative 方案

| 方案 | 为什么没选 / 何时选 |
|---|---|
| **A. 独立进程 + ZMQ（vLLM V1 的做法）** | 真正把 GIL 与引擎隔离、可多卡/多进程扩展。代价是 IPC 协议、序列化、进程生命周期管理，对单机单卡的本项目是明显的过度设计；调试与单测也变复杂。等需要 TP/PP 或多机时再上。 |
| **B. `loop.run_in_executor` + 给 core 加 `threading.Lock`** | 能跑，但锁把"串行"这件事从"结构上不可能并发"降级为"靠人不忘加锁"，一旦新增入口就会漏。单写者队列把正确性变成结构性保证。 |
| **C. 每请求一个 `asyncio.Task` 直接调 core** | 等价于 B，且会让请求间的调度顺序不确定（破坏 FCFS 与 Scheduler 的权威性）。 |
| **C2. 让 `AsyncEngine.generate` 直接是 async generator** | 简洁，但会形成"SSE 生成器套流生成器"的嵌套，内层清理时机不确定（见 6.4）。返回 `AsyncStream` 对象可显式 `aclose()`。 |
| **D. 用 `sse-starlette` / 现成 SSE 库** | 少写 20 行，但帧格式是线上协议的一部分，藏在第三方库里不利于排查"最后一个 token 被吞"这类问题；也多一个依赖。 |
| **E. 支持 `stop` 序列** | 需要在服务端缓冲已生成文本并按 stop 截断，还会牵扯"截断时已产出 token 的 usage 统计"。本阶段范围外，列入 Known Limitations。 |

---

## 8. 当前 Known Limitations

1. **没有 `stop` / `n>1` / `logprobs` / function calling**。OpenAI 请求里带这些字段会被
   pydantic 静默忽略（不报错，但也**不生效**）——比"声明了却不支持"安全，但仍需注意。
2. **流式响应不带 `usage`**。需要 `stream_options: {"include_usage": true}` 才符合 OpenAI 语义，
   Task 10（Metrics）接入后再补。
3. **一个进程只有一个引擎循环**：`step()` 仍是"逐个请求跑前向"，不是一次 batched matmul。
   因此并发请求是**时间片交错 + 逐请求前向**，吞吐提升有限（同 Task 08 的限制）。
4. **块池容量静态**：并发准入没有"按剩余可用块数"做闸门，`BlockPool` 耗尽时会在生成中途抛
   `ValueError`（由 `_run_loop` 捕获 → 取消全部在飞请求 → 循环回到空闲态，不会静默死掉）。
5. **取消不保证"立刻停止当前 step"**：已在飞的 step 会跑完，取消在下一次 drain 生效。
   CPU 单 step 约 0.3s，可感知但可接受；GPU 上同理。
6. **`abort_nowait` 是 fire-and-forget**：引擎循环若已停（或被外部杀死），命令不会被消费。
   `_run_loop` 退出时与 `shutdown()` 都会兜底 `_finish_all_streams`，正常路径不会漏。
7. **单进程单服务**：`uvicorn --workers N` 会各自加载一份模型与块池，互不共享。
8. **chat 只支持纯文本 `content`**，多模态数组会在 pydantic 校验阶段失败。
9. **Windows 本机内存偏紧**：35 条 model 测试跑在同一进程会撑爆内存（见 §9 备注）。

---

## 9. 进入下一阶段前的验收命令

```bash
set "HF_HOME=D:\LiteInfer\hf_cache"
set PYTHONPATH=d:\LiteInfer

python -m pytest tests/test_async_engine.py -q   # 12 passed
python -m pytest tests/test_server_api.py -q     # 16 passed（含真实 uvicorn 断连回收）
python -m pytest tests/test_no_hardcoded_cuda.py -q   # 1 passed
python -m pytest -q                             # 161 passed（Task 01-08 不回归）
python -m pytest -m model -q tests/test_async_engine.py   # 1 passed（真模型流式对齐）
python -m liteinfer.server.main --port 8000      # 起服务
python examples/async_engine_demo.py --max-tokens 8
python examples/server_demo.py --max-tokens 8    # OpenAI Client + 断连回收
```

**备注（本机环境问题，非代码问题）**：`pytest -m model -q`（35 条一次性跑）会在第 1~16 个
测试后触发 `Windows fatal exception: access violation`，崩溃点在
`transformers/core_model_loading.py::_materialize_copy`。实测**逐文件跑 8 个文件全部通过**
（1+4+7+7+8+2+2+4 = 35 passed）。原因是每个测试模块都有 module 级 `loaded` fixture，
FP32 的 0.5B HF 模型 + MinimalQwen 副本各约 2GB，本机 16.9GB 内存仅剩约 5GB 可用时，
连续多次加载会撑爆。建议：要么逐文件跑，要么把 model fixture 改成 session 级只加载一次
（涉及改动 Task 01-08 的既有测试文件，本次未动）。
