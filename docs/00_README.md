# LiteInfer 项目总览

## 项目名称
**LiteInfer — 单卡高吞吐 LLM 推理与服务引擎**

## 一句话定位
面向开源 Decoder-only 大模型，自主实现从模型加载、自回归推理、KV Cache、连续批处理、Paged KV Cache、请求调度到 OpenAI-Compatible API 的完整 LLM Serving Engine，并通过系统 Benchmark 定量分析吞吐、延迟与显存利用率。

## 项目目标
LiteInfer 不是“做一个聊天机器人”，而是实现大模型应用下面的推理基础设施层。

典型链路：

```text
Agent / RAG / 应用
        ↓
OpenAI-Compatible API
        ↓
LiteInfer
        ↓
Qwen / Llama
        ↓
GPU
```

## 第一版范围
只支持：
- 单机
- 单 GPU
- Decoder-only Transformer
- Qwen2/Qwen2.5 架构
- FP16/BF16
- 文本生成
- OpenAI-Compatible API
- SSE Streaming

暂不支持：
- Multi-GPU
- Tensor Parallel
- MoE
- AWQ/GPTQ/FP8
- 多模态
- LoRA Serving
- Speculative Decoding
- CPU KV Swap
- 分布式推理
- 自定义 CUDA PagedAttention Kernel

## 最终必须完成的 10 项
1. 不依赖 `model.generate()` 完成核心推理。
2. 自主实现 Autoregressive Decode。
3. 自主管理 KV Cache。
4. 自主实现 Scheduler。
5. 支持 Continuous Batching。
6. 自主实现 Block Pool + Paged KV Cache。
7. 支持 OpenAI-Compatible Streaming API。
8. 有完整 TTFT/TPOT/Throughput 指标。
9. 有真实 Benchmark 与 Ablation。
10. 核心模块有 Unit Test、Stress Test 与 Design Document。

## 推荐开发顺序
M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7

- M0：HuggingFace Baseline
- M1：Manual Generation + Sampling
- M2：Minimal Qwen + KV Cache
- M3：Request + Engine Core
- M4：Scheduler + Continuous Batching
- M5：Paged KV Cache
- M6：API + Async + Streaming + Metrics
- M7：Prefix Cache + Benchmark + 文档打磨

## 目录说明
- `01_PRD_Project_Overview.md`：产品定位、用户、场景、边界。
- `02_Technical_Architecture.md`：整体技术架构与数据流。
- `03_Module_Development_Guide.md`：每个模块的开发方案与知识讲解。
- `04_Roadmap_and_Milestones.md`：里程碑、周计划和验收标准。
- `05_Test_and_Benchmark_Plan.md`：测试、压测、消融实验设计。
- `06_Interview_Knowledge_Base.md`：面试原理与高频追问。
- `07_Agent_Development_Instructions.md`：交给其他 Agent 的开发合同与任务拆分。
- `08_Resume_Project_Template.md`：最终简历写法模板。
- `09_Project_Checklist.md`：最终交付检查清单。
