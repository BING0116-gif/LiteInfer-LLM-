# LiteInfer — 单卡高吞吐 LLM 推理与服务引擎

从零手写的 LLM Serving Engine（mini-vLLM），覆盖 Autoregressive Decode、KV Cache、
Continuous Batching、Paged KV Cache、Scheduler、OpenAI-Compatible API 与 Benchmark。

> 本仓库只放**代码**。项目规划文档在 `docs/`，开发规范以 `docs/07_Agent_Development_Instructions.md` 为准。

## 当前状态

- [x] 仓库初始化
- [ ] Task 01：项目骨架 + HF Baseline（代码完成，验收中）
- [ ] Task 02–14

## 开发环境（重要）

| 项 | 说明 |
|---|---|
| 开发机 | 无 NVIDIA GPU，**一切代码必须在 CPU 上可运行** |
| 设备配置 | 统一走 `EngineConfig.device`，**禁止硬编码 `"cuda"`**（有单测把关） |
| dtype | CPU 用 `torch.float32`；云端 GPU 用 `torch.float16`（T4/P100 不支持 BF16） |
| 模型缓存 | `HF_HOME=D:\LiteInfer\hf_cache`（未设环境变量时代码内兜底到 D 盘/仓库内） |
| Python | 3.12（venv 在 `D:\LiteInfer\.venv`） |

## 快速开始

```bash
# 1) venv（已建好则跳过；路径必须写 D:/ 风格，Git Bash 下 /d/ 不会转换）
C:\Users\HUAWEI\AppData\Local\Programs\Python\Python312\python.exe -m venv D:/LiteInfer/.venv

# 2) 依赖（torch 装 CPU 版，省 2.3GB）
D:\LiteInfer\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
D:\LiteInfer\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 3) 快速测试（秒级，不需要模型）
D:\LiteInfer\.venv\Scripts\python.exe -m pytest -q

# 4) 需要真实模型的测试与示例（首次会下载 Qwen2.5-0.5B 约 1GB 到 hf_cache）
D:\LiteInfer\.venv\Scripts\python.exe -m pytest -q -m model
D:\LiteInfer\.venv\Scripts\python.exe examples\baseline_demo.py --prompt "你好" --max-new-tokens 32
```

## 目录结构

```text
LiteInfer/
├── docs/
│   ├── ...          # 9 份规划文档（勿改动 07 号的规范语义）
│   └── design/      # 每个 Task 的设计文档
├── liteinfer/       # 引擎源码
│   ├── config.py    # EngineConfig 配置中心
│   ├── device.py    # 设备/dtype 抽象与降级
│   └── model/       # loader + HF baseline
├── examples/        # 最小运行示例
├── tests/           # 单元测试（默认跳过 model 标记）
├── benchmark/       # 基准脚本与结果
└── README.md
```
