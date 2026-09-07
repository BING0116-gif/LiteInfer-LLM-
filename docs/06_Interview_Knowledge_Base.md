# LiteInfer 面试知识库

# 1. 这个项目是干什么的？
LiteInfer 是一个单卡 LLM 推理服务引擎。

它位于：
```text
Agent / RAG / App
       ↓
LiteInfer
       ↓
Qwen/Llama
       ↓
GPU
```

负责请求排队、调度、Batch、KV Cache、流式输出、显存管理和性能监控。

---

# 2. Mini-vLLM / LiteInfer 算 Agent 吗？
不算。

它属于：
**LLM Inference Infrastructure / Model Serving**

不是：
- Agent
- RAG
- 基座模型训练

它不训练 Qwen 参数，而是让已经训练好的模型更高效地运行。

---

# 3. 为什么 LLM 生成是慢的？
Decoder-only LLM 是自回归生成。

第 N 个 Token 依赖前 N-1 个 Token，因此生成阶段存在天然串行依赖。

---

# 4. 为什么需要 KV Cache？
历史 Token 的 K/V 在后续 Decode 中不会变化。

缓存历史 KV 后，新一步只需要计算新增 Token 的 Q/K/V，避免重复计算整个历史序列。

---

# 5. KV Cache 为什么吃显存？
近似：
```text
2 × layers × kv_heads × head_dim × tokens × bytes
```

2 代表 K 和 V。

高并发 = 多条 sequence 同时持有 KV。

---

# 6. Prefill 与 Decode 区别
Prefill：
- 一次处理整个 Prompt；
- token 数多；
- 计算量大；
- 决定第一个 token 何时出现。

Decode：
- 每步生成少量 token；
- 持续读取历史 KV；
- 高频循环；
- 影响 TPOT / ITL。

---

# 7. 为什么 Continuous Batching 有用？
Static Batch 中短请求完成后 slot 空闲。

Continuous Batching 每个 step 都可以：
- 移除完成请求；
- 加入新请求；
- 动态重组 Batch。

因此提高 GPU 利用率和总体吞吐。

---

# 8. Scheduler 本质是什么？
不是“排队算法”。

本质是：
**Latency vs Throughput 的资源调度问题。**

更大 Batch：
- throughput 更高；
- TTFT 可能更差。

更激进的低延迟调度：
- TTFT 更好；
- GPU 利用率可能降低。

---

# 9. 什么是 Paged KV Cache？
把 KV Cache 切成固定大小 Block。

每个请求通过 Block Table 将逻辑位置映射到物理 GPU Block。

类似：
```text
虚拟页 → 页表 → 物理页
```

---

# 10. Paged KV 和 PagedAttention 是一回事吗？
不是。

Paged KV：
- block allocation
- block table
- memory reuse
- fragmentation control

PagedAttention：
还要求 Attention Kernel 能直接高效读取非连续 block。

如果第一版只是 gather 后再做 PyTorch Attention，应该称为 Paged KV 管理，不应声称实现完整高性能 PagedAttention Kernel。

---

# 11. 为什么 Block Size 不能越小越好？
Block 小：
- 内部碎片少；
- metadata 和索引开销高。

Block 大：
- 管理简单；
- 尾部浪费大。

所以必须 Benchmark。

---

# 12. 为什么选 FCFS？
第一版目标是验证系统正确性。

FCFS：
- 简单；
- 行为可解释；
- 易测试；
- 可以作为后续复杂调度策略 baseline。

---

# 13. KV Cache 不够怎么办？
第一版做 admission control：

```text
新请求继续等待
```

不要立即引入 CPU Swap。

进阶可以做：
- preemption
- recompute
- CPU swap

---

# 14. 为什么不直接写 CUDA？
因为需要先分离：
- 系统正确性问题；
- Kernel 性能问题。

先用 PyTorch 建立正确 baseline，再根据 profiling 决定 Triton/CUDA 优化。

---

# 15. 为什么不用 vLLM？
项目目标是学习和自主实现 LLM Serving 核心执行链。

vLLM 可以作为参考和 benchmark，但不能作为内部 runtime。

---

# 16. 为什么实现 Minimal Qwen？
如果只调用 `AutoModelForCausalLM`，对 Transformer 内部理解仍然浅。

自主实现：
- RMSNorm
- RoPE
- GQA
- SwiGLU
- Decoder

可以证明理解模型结构，而不是只会调用 API。

---

# 17. GQA 为什么能省 KV Cache？
Q Head 多，KV Head 少。

多个 Query Head 共享较少的 K/V，因此历史 KV Cache 规模显著下降。

---

# 18. Prefix Cache 是什么？
多个请求共享相同前缀时，复用前缀对应 KV。

典型：
- system prompt
- few-shot examples
- shared policy prompt

收益主要体现在减少重复 Prefill。

---

# 19. 为什么 Prefix Cache 用 ref_count？
共享 Block 可能同时被多个请求引用。

只有 `ref_count == 0` 时才能真正回收到 Free Pool。

---

# 20. 为什么只缓存完整 Block？
Partial Block 可能继续追加 token。

把可变尾部共享会增加一致性和 hash 语义复杂度。

完整 Block 更适合作为不可变共享单元。

---

# 21. TTFT 是什么？
Time To First Token。

从请求到达到第一个 token 输出的时间。

包括：
- queue
- schedule
- prefill
- output pipeline

---

# 22. TPOT 是什么？
Time Per Output Token。

衡量首 token 后续生成速度。

---

# 23. ITL 是什么？
Inter-Token Latency。

相邻 token 输出间隔。

---

# 24. 为什么吞吐高不代表用户体验好？
为了吞吐可以把 Batch 做很大，但 Queue Time 和 TTFT 可能变差。

所以 Serving 必须同时观察：
- throughput
- TTFT
- TPOT
- P95 latency

---

# 25. 为什么需要 Cancellation？
用户断开后如果 GPU 仍继续生成：
- 浪费算力；
- 占用 KV；
- 影响其他请求。

所以取消是资源管理问题，不只是 API 功能。

---

# 26. 如何证明项目不是 Demo？
至少展示：
- correctness test
- unit test
- stress test
- architecture
- metrics
- benchmark
- ablation
- known limitations

---

# 27. 如何回答“你最大的技术难点是什么？”
推荐回答主线：

1. 从单请求推理切到多请求状态机；
2. Scheduler 与 KV Cache 必须同时考虑；
3. Paged KV 的 Block 生命周期复杂；
4. Streaming/Cancel 会引入跨层状态同步；
5. 通过 stress test 和 invariant 保证资源不会泄漏。

---

# 28. 如何回答“为什么这个项目有价值？”
因为它把能力从：
```text
LLM 应用开发
```
扩展到：
```text
LLM Runtime / Serving / Systems
```

与 Agent/RAG 项目形成明显互补。
