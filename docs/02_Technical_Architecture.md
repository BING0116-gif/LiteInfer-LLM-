# LiteInfer 技术架构设计

## 1. 总体架构

```text
Client
  │
  │ HTTP / OpenAI SDK
  ▼
┌────────────────────────────┐
│ API Server                 │
│ FastAPI                    │
│ /v1/chat/completions       │
│ /v1/completions            │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│ Input Processor            │
│ Tokenizer                  │
│ Params Validation          │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│ Async Engine               │
│ Request Queue              │
│ Streaming Queue            │
│ Cancellation               │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│ Engine Core                │
│ Request Registry           │
│ Scheduler                  │
│ KV Cache Manager           │
│ Model Runner               │
│ Sampler                    │
│ Output Processor           │
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────┐
│ GPU Worker                 │
│ Qwen Decoder               │
│ KV Cache                   │
│ LM Head                    │
└────────────────────────────┘

旁路：
Metrics / Tracing / Benchmark / Logging
```

## 2. 请求生命周期

```text
HTTP Request
   ↓
Tokenize
   ↓
Request(status=WAITING)
   ↓
waiting_queue
   ↓
Scheduler admission
   ↓
PREFILL
   ↓
first token
   ↓
DECODE
   ↓
stream token
   ↓
EOS / max_tokens / cancelled
   ↓
FINISHED / CANCELLED
   ↓
free KV blocks
```

## 3. Engine Core 主循环

```python
while True:
    receive_new_requests()
    scheduled_batch = scheduler.schedule()
    outputs = model_runner.execute(scheduled_batch)
    update_request_state(outputs)
    emit_streaming_outputs()
    release_finished_requests()
```

Engine Core 是项目的“总控层”。

## 4. Prefill 与 Decode

### Prefill
输入完整 Prompt，一次处理多个 Token，并为这些 Token 建立 KV Cache。

特点：
- 计算量大；
- 更偏 compute-bound；
- 对 TTFT 影响明显。

### Decode
每一步通常只新增一个 Token，但持续读取历史 KV Cache。

特点：
- 高频迭代；
- 内存读写占比高；
- 对 TPOT/ITL 影响明显。

## 5. Continuous Batching

传统静态 Batch：
```text
[A B C]
B 完成后：
[A _ C]
C 完成后：
[A _ _]
```

Continuous Batching：
```text
[A B C]
B 完成：
[A D C]
C 完成：
[A D E]
```

每个 Decode Step 后都允许重新调度。

## 6. Paged KV Cache

### 逻辑
```text
Request A logical blocks:
[0][1][2]
```

### 物理
```text
A0 → physical 7
A1 → physical 21
A2 → physical 4
```

### Block Table
```python
block_table["A"] = [7, 21, 4]
```

位置映射：
```text
position = 37
block_size = 16

logical_block = 37 // 16 = 2
offset        = 37 % 16  = 5

physical_block = block_table[2]
```

最终访问：
```text
K[physical_block][offset]
V[physical_block][offset]
```

## 7. 推荐 KV Cache 物理布局

```python
K[layer]:
[num_blocks, block_size, num_kv_heads, head_dim]

V[layer]:
[num_blocks, block_size, num_kv_heads, head_dim]
```

## 8. Scheduler 输入与输出

输入：
- waiting requests
- running requests
- available KV blocks
- token budget
- sequence budget

输出：
- 本轮执行哪些 request
- 每个 request 执行多少 token

核心权衡：
```text
更大 Batch
→ Throughput ↑
→ Queue Time / TTFT 可能 ↑
```

## 9. Async Serving

```text
HTTP Request
   ↓
asyncio queue
   ↓
Engine
   ↓
per-request output queue
   ↓
SSE Streaming
```

每个请求维护自己的输出队列。

## 10. 请求取消

Client Disconnect：
```text
Request → CANCELLED
↓
Scheduler 移除
↓
停止继续 decode
↓
释放 KV Blocks
```

必须防止：
- GPU 继续白算；
- KV Cache 泄漏。

## 11. Prefix Cache

相同前缀：
```text
System Prompt
+ shared instructions
```

第一次：
```text
Prefill
↓
cache full blocks
```

后续：
```text
hash prefix block
↓
cache hit
↓
reuse KV block
↓
ref_count + 1
```

只缓存完整 Block，降低一致性复杂度。

## 12. 推荐代码目录

```text
liteinfer/
│
├── liteinfer/
│   ├── config.py
│   ├── engine/
│   ├── scheduler/
│   ├── cache/
│   ├── model/
│   ├── sampling/
│   ├── server/
│   └── observability/
│
├── benchmark/
├── tests/
├── docs/
├── Dockerfile
├── pyproject.toml
└── README.md
```
