# LiteInfer 模块开发方案 + 知识讲解

# 模块 1：Model Loader

## 目标
负责模型配置、Tokenizer、权重、dtype 和 GPU 初始化。

## 建议接口
```python
EngineConfig(
    model_path,
    dtype,
    device,
    max_model_len,
    block_size,
    max_num_seqs,
    max_num_batched_tokens,
    gpu_memory_utilization,
)
```

## 第一阶段
允许：
```python
AutoModelForCausalLM.from_pretrained(...)
```

但最终禁止使用：
```python
model.generate(...)
```

## 面试知识
- FP32：4 Byte
- FP16：2 Byte
- BF16：2 Byte
- BF16 指数范围更大，通常比 FP16 更稳定，但需要硬件支持。

## 验收
- 能加载模型与 tokenizer；
- 显示 GPU、dtype、权重显存；
- 最终核心推理不依赖 `generate()`。

---

# 模块 2：Manual Autoregressive Generation

## 原理
Decoder-only LLM 本质是：
```text
输入 token
↓
forward
↓
logits
↓
选 next token
↓
append
↓
继续 forward
```

## 伪代码
```python
tokens = tokenizer(prompt)

for _ in range(max_new_tokens):
    logits = model(tokens)
    next_token = sampler(logits[:, -1, :])
    tokens.append(next_token)
    if next_token == eos:
        break
```

## 目标
先不考虑性能，只建立可控推理主链。

## 面试知识
Autoregressive generation 是串行的，第 N 个 Token 依赖前面的 Token，因此无法一次并行生成未来所有 Token。

## 验收
- 不调用 `generate()`；
- Greedy 模式输出与 HF 基本一致。

---

# 模块 3：Sampler

## 必须支持
- Greedy
- Temperature
- Top-K
- Top-P
- EOS
- max_tokens
- seed

## Temperature
```text
logits' = logits / T
```
T 越低越确定。

## Top-K
只保留最高的 K 个候选。

## Top-P
保留累计概率达到 P 的最小 Token 集合。

## 工程设计
Model Runner 负责 logits；
Sampler 只负责 next token。

## 验收
固定 seed 时结果可重复。

---

# 模块 4：Minimal Qwen Decoder

## 建议自己实现
- Embedding
- RMSNorm
- RoPE
- GQA Attention
- SwiGLU MLP
- Residual
- LM Head

## RMSNorm
```text
RMS(x) = sqrt(mean(x^2) + eps)
y = x / RMS(x) * weight
```

## RoPE
不直接加 position embedding，而是根据 position 对 Q/K 做旋转，使 Attention 获得相对位置信息。

## GQA
例如：
```text
Q heads = 32
KV heads = 8
```
多个 Query Head 共享较少的 KV Head。

核心收益：
- 降低 KV Cache 显存；
- 降低 Decode 阶段 KV 访问量。

## SwiGLU
```text
gate = silu(W_gate x)
up   = W_up x
hidden = gate * up
output = W_down hidden
```

## 验收
与 HuggingFace 模型逐层比：
- embedding
- attention
- decoder layer
- final logits

使用 `torch.allclose()` 或统计 top-1 agreement。

---

# 模块 5：KV Cache

## 为什么需要
历史 Token 的 K/V 在后续 Decode 中不会变化。

没有 KV Cache：
```text
A
A B
A B C
A B C D
```

有 KV Cache：
```text
保存 K_A/V_A ...
新 token 只计算自己的 Q/K/V
```

## 单 Token KV 显存近似
```text
2 × num_layers × num_kv_heads × head_dim × bytes_per_element
```

2 代表 K + V。

## 第一阶段
先实现 contiguous KV Cache。

## 验收
对比：
- no cache
- cache

生成长度：
32 / 64 / 128 / 256

记录 latency。

---

# 模块 6：Request Model

## 推荐字段
```python
class Request:
    request_id
    prompt_token_ids
    output_token_ids
    sampling_params

    status
    num_prompt_tokens
    num_computed_tokens

    block_table

    arrival_time
    scheduled_time
    first_token_time
    finish_time
```

## 状态
学习版：
```text
WAITING → PREFILL → DECODING → FINISHED
```

进阶版：
通过 `num_computed_tokens` 判断进度。

## 为什么要状态机
推理服务不是函数调用，而是跨多个 Scheduler Step 的长生命周期任务。

---

# 模块 7：Engine Core

## 核心职责
- 接收新请求；
- 调 Scheduler；
- 调 Model Runner；
- 更新 Request；
- 输出 Token；
- 回收资源。

## 核心循环
```python
while True:
    receive_new_requests()
    batch = scheduler.schedule()
    outputs = model_runner.execute(batch)
    update_requests(outputs)
    emit_outputs()
    release_finished_requests()
```

## 验收
单进程下可以同时维护多个 Request 状态。

---

# 模块 8：Scheduler

## V0 策略
FCFS：First Come First Served。

## 约束
```text
max_num_seqs
max_num_batched_tokens
available_kv_blocks
```

## 调度过程
```text
running:
A B C

waiting:
D E F

检查：
- token budget
- sequence budget
- kv blocks

若有资源：
D 被 admit
```

## 核心知识
Scheduler 本质是：
**Latency vs Throughput 的资源调度问题。**

Batch 大：
- throughput ↑
- GPU utilization ↑
- queue time 可能 ↑
- TTFT 可能 ↑

## 进阶
Chunked Prefill：
将超长 Prompt 拆成多个 token chunk，与 Decode 请求交错执行。

---

# 模块 9：Continuous Batching

## Static Batch
```text
[A B C]
B 完成：
[A _ C]
```

## Continuous Batching
```text
[A B C]
B 完成：
[A D C]
```

## 关键点
每次 Decode Step 后重新调度。

## 验收
8~16 个不同输出长度请求，可以动态进入/退出 batch。

---

# 模块 10：Block Pool / Paged KV

## 数据结构
```python
class KVBlock:
    block_id
    ref_count
    block_hash
    is_free
```

## Block Pool
```text
Block0
Block1
...
BlockN
```

## Free Queue
```text
[1, 5, 7, 9, 12, ...]
```

## Request Block Table
```python
request_A.block_table = [12, 7, 98]
```

## 核心接口
```python
allocate(request_id, num_tokens)
append_slots(request_id, num_tokens)
get_block_table(request_id)
free(request_id)
get_usage()
```

## Block Size Trade-off
Block 小：
- 内部碎片少；
- metadata 多；
- 索引/管理开销高。

Block 大：
- 管理简单；
- 最后一个 block 浪费更大。

建议消融：
8 / 16 / 32 / 64。

## 重要表述
Paged KV Cache Manager ≠ 完整 PagedAttention Kernel。

第一版允许：
```text
Paged KV blocks
↓
gather
↓
连续 K/V
↓
PyTorch Attention
```

这证明的是内存管理与调度，不应声称实现高性能 CUDA PagedAttention。

---

# 模块 11：Model Runner

## Scheduler
决定“谁跑”。

## Model Runner
决定“怎么跑”。

## 第一版接口
```python
run_prefill(...)
run_decode(...)
```

## Decode Batch
```text
A_last
B_last
C_last
↓
一次 GPU forward
```

这正是 Continuous Batching 的核心收益之一。

## 负责
- input ids
- positions
- attention metadata
- block table
- KV read/write
- forward
- logits

---

# 模块 12：Async Engine

## 为什么
HTTP 线程不能等模型整段输出完成。

## 架构
```text
HTTP
↓
asyncio input queue
↓
engine
↓
per-request output queue
↓
SSE stream
```

## 请求取消
Client disconnect：
```text
CANCELLED
↓
remove from scheduler
↓
free KV blocks
```

## 验收
用户中途取消后，GPU 不继续生成，KV 不泄漏。

---

# 模块 13：OpenAI-Compatible API

## 接口
- POST `/v1/completions`
- POST `/v1/chat/completions`

## 参数
- model
- messages/prompt
- max_tokens
- temperature
- top_p
- stream

## 价值
现有 OpenAI SDK、Agent 框架只需修改 `base_url` 即可调用。

---

# 模块 14：Streaming

使用 SSE。

目标：
```text
Token 1
Token 2
Token 3
...
```

而不是等待整段生成完成。

TTFT 因此成为可测指标。

---

# 模块 15：Observability

## Request-level
- Queue Time
- TTFT
- TPOT
- ITL
- E2E Latency
- Prompt Tokens
- Output Tokens

## Server-level
- requests_running
- requests_waiting
- kv_cache_usage
- prompt_tokens_total
- generated_tokens_total
- tokens_per_second

## Tracing
```text
RECEIVED
TOKENIZED
WAITING
SCHEDULED
PREFILL_START
FIRST_TOKEN
DECODE
FINISHED
```

---

# 模块 16：Prefix Cache

## 场景
多个请求共享相同长 System Prompt。

第一次：
```text
Prefill → cache blocks
```

后续：
```text
hash → hit → reuse block
```

## 推荐 Hash
```text
hash(
    previous_block_hash,
    current_block_tokens,
    model_namespace
)
```

## ref_count
共享 Block 不能在一个请求结束时直接释放。

```text
A/B/C 共用 Block 7
ref_count = 3
```

直到 `ref_count == 0` 才返回 free queue。

---

# 模块 17：Metrics / Benchmark

必须独立成模块，不能最后临时补。

核心指标：
- TTFT
- TPOT
- ITL
- Request Throughput
- Output Tokens/s
- Peak GPU Memory
- KV Cache Utilization
- P50/P95 latency
