# LiteInfer 开发进度

> **每个新开发窗口的 AI 先读这个文件**，它记录了当前进度、已有文件与关键接口。
> 每完成一个 Task，把 AI 给出的记录追加到下方「Task 记录」章节。

## 总体进度

| Task | 内容 | 运行环境 | 状态 | Commit |
|---|---|---|---|---|
| 01 | 项目骨架 + HF Baseline | CPU | ⬜ 未开始 | |
| 02 | Manual Generation Loop + Sampling | CPU | ⬜ 未开始 | |
| 03 | Minimal Qwen Decoder | CPU（FP32 对齐） | ⬜ 未开始 | |
| 04 | Contiguous KV Cache | CPU | ⬜ 未开始 | |
| 05 | Request + Engine Core | CPU | ⬜ 未开始 | |
| 06 | Continuous Batching Scheduler | CPU | ⬜ 未开始 | |
| 07 | BlockPool + Paged KV | CPU | ⬜ 未开始 | |
| 08 | ModelRunner 接入 Paged KV | CPU | ⬜ 未开始 | |
| 09 | Async Engine + Streaming API | CPU | ⬜ 未开始 | |
| 10 | Metrics + Tracing | CPU | ⬜ 未开始 | |
| 11 | Prefix Cache | CPU | ⬜ 未开始 | |
| 12 | Benchmark + Ablation | **云端 GPU** | ⬜ 未开始 | |
| 13 | Docker + CI + README | CPU | ⬜ 未开始 | |
| 14 | 接入知微 | CPU | ⬜ 未开始 | |

## 环境备忘

- 开发机：**无 NVIDIA GPU**，所有 Task 01–11 必须在 CPU 上跑通
- Python 3.13；PyTorch 装 CPU 版
- `HF_HOME=D:\LiteInfer\hf_cache`
- 设备统一走 `EngineConfig.device`，禁止硬编码 `"cuda"`

---

# Task 记录

## Task 00：仓库初始化（2026-09-07）

**新增文件**

```text
.gitignore        # 拦截模型文件与缓存
README.md         # 项目说明与环境约束
PROMPT.md         # 开发窗口标准 prompt 模板
PROGRESS.md       # 本文件
docs/             # 9 份规划文档
```

**关键约定**

- 规划文档（图纸）在 `C:\Users\HUAWEI\Desktop\LiteInfer_Project_Plan\`
- 代码仓库在 `D:\LiteInfer`，两者分离
- 每个 Task 开一个新对话窗口，prompt 用 `PROMPT.md`，只改 Task 编号

**踩过的坑**

- 沙箱环境无法完成 GitHub 浏览器授权，push 需在用户自己的 CMD 执行；
  CMD 切盘要用 `cd /d D:\LiteInfer`（反斜杠），不能写 `cd /d/LiteInfer`

---

<!-- 后续每个 Task 完成后，把 AI 生成的记录追加到这里 -->
