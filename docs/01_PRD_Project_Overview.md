# LiteInfer PRD / Project Overview

## 1. 产品定位
LiteInfer 是一个面向开源大模型的单卡推理服务引擎。

它解决的问题不是“模型会不会回答问题”，而是：
- 多个用户同时请求时怎么排队；
- 哪些请求应该一起送入 GPU；
- 如何避免每个请求单独 `model.generate()`；
- KV Cache 如何分配、复用与回收；
- 请求结束后 GPU 显存如何释放；
- 如何平衡吞吐和延迟；
- 如何让 Agent/RAG 系统通过标准 API 直接调用本地模型；
- 如何量化 TTFT、TPOT、P95、吞吐和 KV 利用率。

## 2. 目标用户
- 自部署 Qwen/Llama 的 AI 开发者；
- Agent/RAG 系统开发者；
- 企业私有化模型部署团队；
- 学校/实验室 AI 平台；
- 想学习 LLM Inference Systems 的开发者。

## 3. 典型应用场景

### 场景 A：企业内部私有化模型服务
```text
企业业务系统
   ↓
LiteInfer
   ↓
本地 Qwen
   ↓
企业 GPU
```

### 场景 B：Agent/RAG 底层模型服务
```text
Agent
 ↓
Tool / RAG
 ↓
LiteInfer
 ↓
本地模型
```

### 场景 C：高并发聊天/客服
几十个用户同时请求，由 Scheduler 动态组成 Batch，而不是每人单独跑一次模型。

### 场景 D：代码助手/知识库
多个请求共享相同系统提示词时，可以通过 Prefix Cache 复用前缀 KV。

### 场景 E：与“知微”集成
```text
知微 Agent
   ↓
LLM Provider
   ↓
LiteInfer OpenAI-Compatible API
   ↓
Qwen
   ↓
GPU
```

## 4. 产品核心痛点
直接使用 HuggingFace `generate()`：
- 对单请求方便；
- 对高并发服务不够灵活；
- 请求生命周期、队列、状态、取消、回收不透明；
- 很难做自定义 Scheduler；
- 很难自主研究 KV Cache 管理；
- 很难做系统级可观测和消融实验。

## 5. 产品成功标准
项目完成后，必须可以：
1. 使用 OpenAI Python Client 直接请求本地 LiteInfer；
2. 支持多个请求动态加入/退出 Batch；
3. 支持流式 Token 输出；
4. 支持中途取消请求并释放 KV Block；
5. 输出 TTFT / TPOT / Throughput / KV Utilization；
6. 在固定模型和固定 GPU 上完成真实 Benchmark；
7. 可以解释每一处核心设计，而不是依赖第三方 Runtime。

## 6. 产品边界
LiteInfer 是学习型、工程型推理引擎，不宣称替代生产级 vLLM/SGLang/TensorRT-LLM。

第一版重点：
- 正确性；
- 核心推理链路；
- Scheduler；
- KV Cache；
- Paged KV；
- Serving；
- Benchmark。

高级功能以后再做。

## 7. 功能优先级

### P0 必须完成
- Manual Autoregressive Decode
- KV Cache
- Request State
- Engine Loop
- FCFS Scheduler
- Continuous Batching
- Block Pool
- Paged KV Cache Manager
- FastAPI
- SSE Streaming
- Cancellation
- Metrics
- Benchmark

### P1 推荐完成
- Minimal Qwen Decoder
- Prefix Cache
- Prometheus Metrics
- Docker
- CI
- Stress Test

### P2 进阶
- Chunked Prefill
- Triton Paged Attention
- CUDA Graph
- Speculative Decoding
- Tensor Parallel
