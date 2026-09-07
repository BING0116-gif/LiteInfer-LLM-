# LiteInfer 测试与 Benchmark 方案

# 1. 测试原则
所有“优化”必须先有 baseline，再有 benchmark。

禁止：
- 先写“提升 3 倍”再凑数据；
- 只跑一个 prompt；
- 只看平均延迟；
- 不记录硬件、模型和参数。

---

# 2. Unit Tests

## Scheduler
必须验证：
- FCFS 顺序；
- max_num_seqs 不突破；
- max_num_batched_tokens 不突破；
- KV 不足时拒绝 admission；
- finished request 会被移除。

## Block Manager
验证：
- allocate
- append
- free
- reuse
- ref_count
- OOM handling

无 Prefix Cache 时：
```text
free_blocks + allocated_blocks = total_blocks
```

## Sampler
验证：
- greedy deterministic
- temperature 合法性
- top-k
- top-p
- seed reproducibility
- EOS

## Request
验证：
- 生命周期合法；
- timestamp 正确；
- cancel 后不可继续 schedule。

---

# 3. KV Cache Stress Test

随机循环 10000 次：
```text
create request
allocate blocks
append tokens
finish/cancel request
free blocks
```

最终：
```text
allocated = 0
free = total
```

检测 memory leak。

---

# 4. 模型正确性

固定：
- same model
- same weights
- same dtype
- same prompt
- greedy decoding
- same seed

比较：
```text
HuggingFace
vs
LiteInfer
```

指标：
- logits allclose
- top-1 agreement
- final token sequence

---

# 5. Integration Tests

完整链路：
```text
HTTP
↓
API
↓
Tokenizer
↓
Engine
↓
Scheduler
↓
Model
↓
Streaming
```

验证：
- HTTP 200
- stream/non-stream
- EOS
- max_tokens
- invalid params
- client cancellation
- concurrent requests

---

# 6. Cancellation Test

创建 20 个请求。
随机取消 10 个。

验证：
- Scheduler 中移除；
- KV Block 释放；
- GPU 不继续生成；
- 其他请求不受影响。

---

# 7. Benchmark 版本

至少比较：

## Baseline A
HuggingFace Sequential Generate

## Baseline B
Manual Generation without KV Cache

## V1
+ KV Cache

## V2
+ Continuous Batching

## V3
+ Paged KV Cache

## V4
+ Prefix Cache

---

# 8. Workloads

## Workload A：Decode-heavy
```text
64 prompt tokens
256 output tokens
```

测试 Decode 性能。

## Workload B：Prefill-heavy
```text
2048 prompt tokens
32 output tokens
```

测试 Prefill。

## Workload C：Typical Chat
```text
512 prompt tokens
128 output tokens
```

测试常规请求。

## Workload D：Shared Prefix
```text
2048 shared prefix
+
different user queries
```

测试 Prefix Cache。

---

# 9. 并发
至少：
```text
1
2
4
8
16
32
```

硬件允许再测 64。

---

# 10. 指标

## TTFT
从请求到达，到第一个 Token 输出。

## TPOT
可近似：
```text
(E2E - TTFT) / (output_tokens - 1)
```

## ITL
相邻两个输出 Token 的时间间隔。

## Throughput
- requests/s
- output tokens/s

## Latency
- P50
- P95

## Memory
- model memory
- peak GPU memory
- KV cache memory
- KV utilization

---

# 11. 必须画的图

1. Concurrency → Throughput
2. Concurrency → P95 TTFT
3. Concurrency → P95 TPOT
4. Concurrency → GPU Memory
5. Block Size → Throughput / KV Utilization
6. Shared Prefix Length → TTFT
7. Queue Length → TTFT
8. Cache Hit Rate → Prefill Saved Tokens

---

# 12. Block Size 消融
测试：
```text
8
16
32
64
```

比较：
- throughput
- internal fragmentation
- metadata overhead
- KV utilization

---

# 13. Scheduler 消融
可以比较：
- FCFS
- Decode-priority
- token-budget variation

观察：
- TTFT
- TPOT
- throughput
- starvation 风险

---

# 14. 实验记录模板

```text
GPU:
CUDA:
PyTorch:
Model:
dtype:
block_size:
max_num_seqs:
max_num_batched_tokens:
prompt_len:
output_len:
concurrency:
num_requests:
```

结果：
```text
requests/s:
output tokens/s:
TTFT P50:
TTFT P95:
TPOT P50:
TPOT P95:
E2E P95:
peak GPU memory:
KV utilization:
```

---

# 15. 简历数字规则
所有简历数字必须来自 Benchmark 结果。

不能提前写：
```text
吞吐提升 3.8x
```

必须：
```text
先测 → 分析 → 再写
```
