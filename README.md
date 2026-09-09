# LiteInfer — 单卡高吞吐 LLM 推理与服务引擎

![CI](https://github.com/BING0116-gif/LiteInfer-LLM-/actions/workflows/ci.yml/badge.svg)

从零手写的 LLM Serving Engine（mini-vLLM 教学级实现）：不调用 `model.generate()`，
自主实现自回归推理、KV Cache、连续批处理、Paged KV、调度器、OpenAI-Compatible API
与系统 Benchmark。

> 本仓库只放**代码**。项目规划文档在 `docs/`，开发规范以
> `docs/07_Agent_Development_Instructions.md` 为准（含补充条款 A：设备抽象）。
> 开发进度与每 Task 的坑见 [PROGRESS.md](PROGRESS.md)。

## 与 vLLM 的关系（诚实声明）

LiteInfer **不依赖** vLLM / SGLang / TensorRT-LLM 作为内部 Runtime，也不宣称"完整复刻 vLLM"：

- 推理执行链（Decode Loop / KV Cache / Scheduler / Block Pool）全部从零实现；
- Paged KV 是 **PyTorch gather 实现**，**不是** CUDA PagedAttention Kernel
  （完整 Kernel 不在本仓库第一版范围）；
- HF `generate()` 仅作为 Baseline 阶段的正确性参照物，生产路径不调用。

## 特性

- **自主推理链**：Manual Generation Loop + Sampling（greedy / temperature / top-k / top-p），greedy 与 HF Baseline 逐字一致
- **手写 Minimal Qwen Decoder**：RMSNorm / RoPE / GQA / SwiGLU / Decoder Layer / LM Head，数值与 HF 对齐（CPU FP32 atol/rtol 1e-4，top-1 100%）
- **KV Cache**：连续缓存（零拷贝历史视图）→ Paged KV（BlockPool / FreeQueue / BlockTable，10k 次 stress 测试零泄漏）
- **Continuous Batching**：FCFS + waiting/running 双队列 + sequence/token 双预算，动态准入
- **Prefix Cache**：blake2b 块哈希 + 引用计数 + LRU 驱逐，共享前缀 workload 可观察命中（命中 token / TTFT 下降）
- **OpenAI-Compatible API**：FastAPI + SSE 流式 + 请求取消（断连即回收 KV 块）
- **Metrics + Tracing**：TTFT / TPOT / ITL / E2E / KV Utilization / Request Trace
- **Benchmark**：5 引擎驱动 × 4 workload × 并发矩阵，CSV / 图表 / Markdown 报告全自动

## 总体架构

```text
Client (OpenAI SDK / curl)
        │ HTTP / SSE
        ▼
┌───────────────────────────────┐
│ API Server (FastAPI)          │
│  /v1/completions /chat        │
│  /v1/models /health /metrics  │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Async Engine                 │
│  command queue + 后台循环     │
│  per-request streaming queue  │
│  取消 / 断连回收              │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ Engine Core + Scheduler       │
│  waiting/running 队列 FCFS    │
│  token/sequence 预算准入      │
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ ModelRunner + 共享块池        │
│  prefill / decode(每步1 token)│
│  paged KV(block_size=16)      │
│  prefix cache(blake2b 链式哈希)│
└───────────────┬───────────────┘
                ▼
┌───────────────────────────────┐
│ MinimalQwenForCausalLM        │
│  (24 层 decoder + LM Head)    │
└───────────────────────────────┘
   旁路：Metrics / Tracing / Benchmark
```

详细设计文档见 [docs/02_Technical_Architecture.md](docs/02_Technical_Architecture.md)
与 `docs/design/`（每个模块一份设计文档）。

## 快速开始

### 0) 环境要求

| 项 | 说明 |
|---|---|
| 开发机 | 无 NVIDIA GPU 同样可跑：**一切代码在 CPU 上可运行** |
| 设备配置 | 统一走 `EngineConfig.device`，禁止硬编码设备字面量（有单测把关） |
| dtype | CPU 用 `torch.float32`；云端 GPU 用 `torch.float16`（T4/P100 不支持 bf16） |
| 模型缓存 | `HF_HOME=D:\LiteInfer\hf_cache`（模型缓存必须放非系统盘） |
| Python | 3.12（路径示例用 `D:/LiteInfer/.venv`，请按实际修改） |

### 1) 安装

```bash
# Windows（Git Bash / CMD，路径用 D:/ 风格避免转义问题）
python -m venv D:/LiteInfer/.venv
D:/LiteInfer/.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
D:/LiteInfer/.venv/Scripts/python.exe -m pip install -e ".[dev]"
```

### 2) 跑测试

```bash
# 快速测试（秒级，不需要模型，默认跳过 -m model）
D:/LiteInfer/.venv/Scripts/python.exe -m pytest -q

# 需要真实模型的测试与示例（首次会下载 Qwen2.5-0.5B 约 1GB 到 hf_cache）
D:/LiteInfer/.venv/Scripts/python.exe -m pytest -q -m model
```

### 3) 跑示例

```bash
# 流式并发示例：3 请求 + 取消 + 与 CachedGenerator 对照
D:/LiteInfer/.venv/Scripts/python.exe examples/async_engine_demo.py --max-tokens 8

# 完整 benchmark 冒烟（测量 → 图表 → 报告，产物落 benchmark/results/demo/）
D:/LiteInfer/.venv/Scripts/python.exe examples/benchmark_demo.py --smoke
```

### 4) 启动 OpenAI 兼容服务

```bash
set "HF_HOME=D:\LiteInfer\hf_cache"
D:/LiteInfer/.venv/Scripts/python.exe -m liteinfer.server.main --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B","messages":[{"role":"user","content":"2+2=?"}],"stream":true}'
```

API 端点一览：`/v1/completions`、`/v1/chat/completions`、`/v1/models`、
`/v1/requests/{id}/cancel`、`/v1/requests/{id}/trace`、`/metrics`、`/health`。

### 5) 用 Docker 跑

```bash
# 构建（默认 CPU 版，GPU 机器见 Dockerfile 头部注释）
docker build -t liteinfer:cpu .

# 运行：挂载本机模型缓存避免容器内重复下载
docker run --rm -p 8000:8000 \
  -v D:/LiteInfer/hf_cache:/model-cache \
  liteinfer:cpu

# 容器内自检（镜像已内置测试与示例）
docker run --rm liteinfer:cpu python -m pytest -q
docker run --rm liteinfer:cpu python -m liteinfer.server.main --help
```

> 镜像内不写死设备：容器内跑 CPU 推理用默认 `--device cpu`；
> GPU 容器启动时用 `--device cuda` 等真实设备名从配置读取，镜像无需区分 GPU/CPU。

## Benchmark

### 方法

- 引擎（真实驱动）：`hf`（HF Baseline）、`nokv`（无缓存）、`kv`（连续 KV）、
  `batch`（EngineCore：连续批处理 + Paged KV）、`prefix`（+Prefix Cache）；
  6 个命名消融 → 5 个真实驱动（continuous batching 与 paged KV 在引擎内是两轴测量），
  详见 [docs/design/benchmark.md](docs/design/benchmark.md)。
- workload：decode-heavy（16→48）、prefill-heavy（160→16）、typical（48→24）、
  shared-prefix（160/16）；并发 1/2/4/8/16/32；block_size=16；greedy。
- 指标：TTFT / TPOT / ITL / E2E（P50/P95）、requests/s、output tokens/s、
  KV utilization、prefix hit tokens；顺带全引擎逐字一致性校验（parity）。
- 一键全矩阵（云端）：`benchmark/run_full_matrix.sh`（设 LITEINFER_DEVICE/LITEINFER_DTYPE）。

### 结果（诚实的数字）

**规则（docs/05 §15）：所有数字来自真实 Benchmark 输出，不允许提前写死。**
本机 CPU 数字只验证流水线正确性，**不用于简历/汇报**；正式性能数字按
docs/07 Task 12 的约定在云端 GPU（FP16）执行。

本机 CPU smoke 实测（Qwen2.5-0.5B，CPU FP32，`examples/benchmark_demo.py --smoke`，
36 cell 全数通过，完整表格见 [benchmark/results/demo/report.md](benchmark/results/demo/report.md)）：

| 观测 | 结果 | 备注 |
|---|---|---|
| 输出一致性 | 36 cell 全部 `parity=OK` | 引擎驱动输出与顺序 KV 逐字一致 |
| sp@conc4 TTFT p50 | prefix **4642.9ms** vs batch **11812.5ms**（↓60.7%） | 同 cell 共享前缀命中 |
| prefix 命中 | 随并发增长：decode 8/12，prefill 80/120，sp 96/144（token） | 只在同一引擎实例内命中 |
| kv 稳态吞吐 | ~2.6–3.2 tok/s（CPU） | 仅供管线验证 |
| 显存指标 | `N/A (no GPU)` | 禁止填 0 |

上云后把 `benchmark/results/` 下的 CSV/JSON/图表 commit 回仓库后，本表即替换为 GPU 数字。

## 已知局限

- 单机单设备、Decode-only、Qwen2/Qwen2.5 架构；不支援 Multi-GPU、Tensor Parallel、MoE、量化、多模态、LoRA、Speculative Decoding、分布式推理；
- Paged KV 为 gather 实现，非 CUDA PagedAttention Kernel；
- Prefix Cache 只缓存完整 Block；
- 详见 [docs/known_limitations.md](docs/known_limitations.md)。

## 目录结构

```text
LiteInfer/
├── docs/                 # 9 份规划文档 + design/（每模块设计文档）
├── liteinfer/            # 引擎源码
│   ├── config.py         # EngineConfig 配置中心
│   ├── engine/           # 请求模型 / EngineCore / AsyncEngine
│   ├── scheduler/        # 连续批处理调度器
│   ├── cache/            # 连续 KV / Paged KV / Prefix Cache
│   ├── model/            # Minimal Qwen / ModelRunner / generator
│   ├── sampling/         # SamplingParams / Sampler
│   ├── server/           # FastAPI + SSE + OpenAI 兼容 schema
│   └── observability/    # Metrics / Tracing
├── examples/             # 最小运行示例
├── tests/                # 单元测试（默认跳过 model 标记）
├── benchmark/            # 基准脚本、一键上云脚本与结果
├── .github/workflows/    # CI（快速测试质量门）
├── Dockerfile            # CPU 优先镜像
└── PROGRESS.md           # 开发进度 + 每 Task 踩坑记录
```

## 文档索引

| 文档 | 用途 |
|---|---|
| [docs/00_README.md](docs/00_README.md) | 项目总览（定位 / 目标 / 范围） |
| [docs/02_Technical_Architecture.md](docs/02_Technical_Architecture.md) | 技术架构与数据流 |
| [docs/07_Agent_Development_Instructions.md](docs/07_Agent_Development_Instructions.md) | 开发规范（含补充条款 A / 任务拆分） |
| [docs/design/](docs/design/) | 每个模块的设计文档（9 项交付） |
| [PROGRESS.md](PROGRESS.md) | 开发进度（Task 01–14 状态表 + 踩坑记录） |