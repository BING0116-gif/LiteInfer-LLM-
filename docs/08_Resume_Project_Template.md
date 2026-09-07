# LiteInfer 简历项目模板

注意：以下性能数字均必须等待真实 Benchmark 后填写。

## 项目名称
**LiteInfer — 单卡高吞吐 LLM 推理与服务引擎｜独立开发**

## 技术栈
Python / PyTorch / CUDA / Transformer / KV Cache / Continuous Batching / Paged KV Cache / FastAPI / Docker

## 简历描述模板

### 项目背景
面向开源 Decoder-only 大模型私有化部署与高并发 Serving 场景，从零实现单卡 LLM 推理服务引擎，覆盖模型执行、请求调度、KV Cache 管理、流式输出和性能评测完整链路。

### 核心架构
自主实现 Autoregressive Decode、Request State Machine、Continuous Batching Scheduler、Block-based Paged KV Cache Manager 与 Async Serving Pipeline，并通过 OpenAI-Compatible API 对外提供模型服务。

### 模型与推理
基于 Qwen 架构实现 RMSNorm、RoPE、GQA Attention 与 SwiGLU Decoder，并加载开源权重完成 HuggingFace Forward 对齐；实现 Prefill/Decode 分离和增量 KV Cache，避免 Decode 阶段重复计算历史 Token。

### 调度与显存
实现 FCFS + Token Budget 调度机制，支持多请求动态加入/退出 Batch；基于固定 Block Pool、Free Queue 和 Request Block Table 管理 KV Cache，实现动态分配、复用和请求结束自动回收。

### 服务与可观测
构建 FastAPI + asyncio 异步服务，提供 OpenAI-Compatible Chat/Completion API、SSE Streaming 与 Client Cancellation；记录 TTFT、TPOT、ITL、P95、Throughput 与 KV Cache Utilization。

### Benchmark
在【GPU 型号】、【模型】、【并发】场景下，相比【Baseline】：
- Output Throughput 提升【待实测】；
- P95 TTFT 为【待实测】；
- Peak GPU Memory 为【待实测】；
- Prefix Cache 在共享前缀场景下减少【待实测】Prefill 开销。

---

# 面试自我介绍版

“LiteInfer 是我为了补足大模型底层推理系统能力做的一个项目。我没有直接调用 vLLM，而是从 HuggingFace 单请求生成开始，逐步去掉 `model.generate()`，自己实现自回归 Decode、KV Cache、Request 状态机、Continuous Batching Scheduler 和 Block-based Paged KV Cache。之后再通过 FastAPI 和 asyncio 做成 OpenAI-Compatible 的推理服务，并针对 TTFT、TPOT、吞吐和显存利用率做 Benchmark。这个项目的重点不是替代 vLLM，而是把 LLM Serving 的核心执行链真正实现并量化验证。”

---

# 与知微形成的能力故事

```text
知微
= 上层 Agent / RAG / 产品能力

LiteInfer
= 底层 LLM Runtime / Serving / Systems 能力
```

最终可以展示：
```text
知微 Agent
↓
LiteInfer
↓
Qwen
↓
GPU
```

这比两个彼此无关项目更容易形成完整技术叙事。
