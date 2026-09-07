# LiteInfer — 单卡高吞吐 LLM 推理与服务引擎

从零手写的 LLM Serving Engine（mini-vLLM），覆盖 Autoregressive Decode、KV Cache、
Continuous Batching、Paged KV Cache、Scheduler、OpenAI-Compatible API 与 Benchmark。

> 本仓库只放**代码**。项目规划文档在 `docs/`，开发规范以 `docs/07_Agent_Development_Instructions.md` 为准。

## 当前状态

- [x] 仓库初始化
- [ ] Task 01：项目骨架 + HF Baseline
- [ ] Task 02–14

## 开发环境（重要）

| 项 | 说明 |
|---|---|
| 开发机 | 无 NVIDIA GPU，**一切代码必须在 CPU 上可运行** |
| 设备配置 | 统一走 `EngineConfig.device`，**禁止硬编码 `"cuda"`** |
| dtype | CPU 用 `torch.float32`；云端 GPU 用 `torch.float16` |
| 模型缓存 | `HF_HOME=D:\LiteInfer\hf_cache`（已设为用户环境变量） |
| Python | 3.13 |

## 快速开始（Task 01 完成后补充）

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## 目录结构

```text
LiteInfer/
├── docs/            # 9 份规划文档（勿改动 07 号的规范语义）
├── liteinfer/       # 引擎源码（Task 01 起创建）
├── tests/           # 单元测试
├── benchmark/       # 基准脚本与结果
└── README.md
```
