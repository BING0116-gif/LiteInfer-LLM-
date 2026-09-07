# LiteInfer 开发 Roadmap 与 Milestones

## 总原则
必须按顺序开发。
上一阶段测试不过，不进入下一阶段。

---

# 双环境开发策略（前置约束）

## 硬件实况（2026-09-07 实测）
- 本机（华为笔记本）：i5-13420H / 15.7GB RAM / **无 NVIDIA GPU**，仅 Intel 核显，无 CUDA
- 老师提供的阿里云 GPU：配额已耗尽，暂不可用
- 云端 GPU 备选：魔搭 ModelScope（A10 24GB，有免费额度）、AutoDL（1.5–2 元/时，可 SSH）、Kaggle（30h/周 T4×2）

## 阶段分工
| 阶段 | 运行环境 | 能否在本机完成 |
|---|---|---|
| M0–M5 | 本机 CPU | 完全可以，含单测与 stress test |
| M6 | 本机 CPU 为主 | 服务链路、SSE、取消逻辑与 GPU 无关 |
| M7 | 云端 GPU | 吞吐 / 显存 / 利用率类指标必须有真实 GPU |

**结论：不要卡在"没 GPU"上。M0–M5 现在就能开工。**

## 由此产生的 5 条硬性要求
1. **禁止硬编码 `cuda`**。所有 device 必须来自 `EngineConfig.device`，代码里禁止出现字面量 `"cuda"` / `.cuda()` / `.to("cuda")`。
2. **dtype 可配置**。本机 CPU 用 **FP32**（CPU 上 FP16 支持差且更慢）；云端 GPU 用 FP16（T4/P100 不支持 BF16，仅 A10/A100 可用 BF16）。
3. **显存指标必须可降级**。`torch.cuda.*` 系列在无 GPU 时要有 fallback，返回 `None` 或改用 RAM 占用，**不能让 benchmark 直接崩**。
4. **数值对齐容差按设备区分**。CPU FP32 与 GPU FP16 的 `allclose` 阈值不同，验收时需写明 atol/rtol。
5. **模型缓存目录指向非系统盘**。本机 C 盘仅剩约 10GB，必须设置 `HF_HOME`。

## 本机环境准备
- venv 建在 D 盘，避免 C 盘空间不足
- PyTorch 装 **CPU 版**（约 200MB），不要装 CUDA 版（约 2.5GB）
- 模型下载走 ModelScope 或设置 `HF_ENDPOINT=https://hf-mirror.com`

---

# M0：HuggingFace Baseline

## 目标
建立最简单可运行基准。

## 功能
- 加载 Qwen2.5-0.5B
- Tokenizer
- `model.generate()` 仅作为 baseline
- 记录单请求 latency

## 验收
输入 Prompt 可以正常输出。

---

# M1：Manual Generation + Sampling

## 目标
去掉 `generate()`。

## 功能
- forward
- logits
- greedy
- temperature
- top-k
- top-p
- EOS

## 验收
Greedy 模式下与 HF 结果基本一致。

---

# M2：Minimal Qwen + KV Cache

## 目标
理解并控制模型内部推理。

## 功能
- RMSNorm
- RoPE
- GQA
- SwiGLU
- Decoder Layer
- contiguous KV Cache
- Prefill / Decode 分离

## 验收
- logits 与 HF 对齐；
- KV Cache 版本正确生成；
- no-cache vs cache 有性能对比。

## CPU 环境补充说明
- 本机用 **FP32** 做对齐（CPU FP16 精度差且更慢），建议 `atol=1e-4, rtol=1e-4`；
- 上云后改用 FP16，容差需放宽到 `atol=1e-2`，**两套阈值都要写进测试**；
- 逐层对比建议同时统计 top-1 agreement，比纯 allclose 更能反映实际生成是否一致；
- 0.5B 模型在 8 核 CPU 上单次 forward 约几十毫秒，对齐测试可接受。

---

# M3：Request + Engine Core

## 目标
从“函数调用”升级为“服务引擎”。

## 功能
- Request
- Request Registry
- waiting/running
- lifecycle
- Engine Loop

## 验收
同时维护多个 Request 状态。

---

# M4：Scheduler + Continuous Batching

## 目标
支持动态并发。

## 功能
- FCFS
- max_num_seqs
- max_num_batched_tokens
- admission control
- dynamic join/leave batch

## 验收
8~16 个不同长度请求动态进入/退出 batch。

---

# M5：Paged KV Cache

## 目标
自主实现 Block-based KV 管理。

## 功能
- KVBlock
- BlockPool
- FreeQueue
- BlockTable
- allocate
- append
- free
- usage
- gather-based attention

## 验收
- Stress test 10000 次随机 allocate/free；
- 无内存泄漏；
- `free + allocated = total` 不变量成立。

## CPU 环境补充说明
- **本阶段完全不依赖 GPU**。BlockPool / FreeQueue / BlockTable 是纯数据结构，
  在 CPU 上跑 10000 次 stress test 与在 GPU 上等价，且更快更省；
- KV 张量先用 CPU tensor 占位即可，上云时只改 `EngineConfig.device`；
- `get_usage()` 这类指标只统计 block 数量（与显存无关），无需降级处理；
- 只有 "peak GPU memory" 这类真显存指标才需要 fallback，留到 M7 处理。

---

# M6：Serving

## 目标
做成可以真正被应用调用的推理服务。

## 功能
- FastAPI
- `/v1/completions`
- `/v1/chat/completions`
- OpenAI SDK compatibility
- asyncio
- SSE Streaming
- cancellation

## 验收
OpenAI Python Client 能直接调用。
中途取消请求后资源正确回收。

---

# M7：Benchmark + Prefix Cache + 打磨

## 功能
- TTFT
- TPOT
- ITL
- P50/P95
- Throughput
- GPU memory
- KV usage
- Prefix Cache
- Benchmark scripts
- Docker
- README
- Design docs

## 运行环境：**云端 GPU**（本机无 GPU，本阶段指标无法在本机产出）

## 验收
完成至少 4 组 workload、5 组 concurrency 和完整消融实验。

## 上云执行说明
- 代码通过 Git 同步（本机 push → 云端 `git clone` → 跑完 push → 本机 pull），
  **不需要手工传文件**；
- 优先选 **魔搭 ModelScope**：A10 24GB、有免费额度、国内直连、Qwen 下载最快；
  备选 AutoDL（1.5–2 元/时，可 SSH）；Kaggle 需注意 T4 不支持 BF16，只能用 FP16；
- 时间预算：0.5B 模型单次 benchmark 约 1–3 分钟，
  4 workload × 6 并发 × 多版本粗估 **8–15 GPU 小时**，加调试约 20 小时内；
- **上云前必须保证 M0–M6 已在本机全部跑通**，云端只做 benchmark，不用来 debug 业务逻辑；
- 跑完立即把原始 CSV/JSON 与图表 commit 回仓库，云端环境随时可能被回收。

## 指标降级规则
| 指标 | 本机 CPU | 云端 GPU |
|---|---|---|
| TTFT / TPOT / ITL / E2E | 可测，但不代表 GPU 性能 | 正式数据 |
| Throughput | 可测，仅作逻辑验证 | 正式数据 |
| Peak GPU Memory | 不可用，返回 `None` | 正式数据 |
| KV Cache Utilization | 用 block 数占比代替 | 正式数据 |

---

# 建议 8 周计划

## 排期前提（已按实际硬件调整）
- **Week 1–5 全部在本机 CPU 完成**，不依赖任何 GPU，现在即可开工；
- **Week 6–7 需要云端 GPU**（魔搭 / AutoDL / Kaggle），若届时仍无 GPU 资源，
  可只完成 Metrics 与 Tracing 的代码与逻辑验证，把 Benchmark 数字留空；
- 云端时间宝贵，**上云前务必确认 M0–M6 已本机跑通**；
- 本机 CPU 上 decode 较慢（0.5B 约 10–25 tokens/s），写测试用例时应
  减小 `max_tokens`（如 32–64）以缩短反馈周期。

## Week 1
- Transformer 推理基础
- HF baseline
- Manual generation
- sampler
- KV Cache 基础

## Week 2
- Minimal Qwen
- RMSNorm
- RoPE
- GQA
- SwiGLU
- correctness test

## Week 3
- Request
- Engine Core
- Scheduler
- Continuous Batching

## Week 4
- BlockPool
- Paged KV
- Stress Test

## Week 5
- FastAPI
- Async Engine
- Streaming
- Cancellation

## Week 6　【需要云端 GPU】
- Metrics
- Tracing
- Benchmark
- 若 GPU 未就绪：先完成指标采集代码，用 CPU 小流量验证逻辑正确性

## Week 7　【需要云端 GPU】
- Prefix Cache（逻辑可先在 CPU 写完并测通）
- Shared-prefix benchmark
- 跑完立即 commit 原始数据与图表

## Week 8
- Docker
- CI
- README
- Architecture
- Resume wording
- Interview notes

---

# 后续进阶优先级

1. Prefix Cache
2. Chunked Prefill
3. Triton Paged Attention
4. CUDA Graph
5. Speculative Decoding
6. Tensor Parallel
