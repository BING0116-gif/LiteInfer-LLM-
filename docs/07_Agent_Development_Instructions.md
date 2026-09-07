# 交给其他 Agent 的开发说明 / 开发合同

请严格按照本文件执行。

# 一、项目目标
实现 LiteInfer：一个单卡高吞吐 LLM 推理与服务引擎。

重点不是复刻完整 vLLM，而是自主实现 LLM Serving 核心执行链。

---

# 二、禁止事项

禁止使用以下系统作为内部 Runtime：
- vLLM
- SGLang
- TensorRT-LLM

可以：
- 阅读架构；
- 作为 benchmark/reference。

禁止最终依赖：
```python
model.generate()
```

可以仅在 Baseline 阶段使用。

禁止：
- 预先编造性能数字；
- 一次性生成整个项目；
- 跳过测试；
- 跳过 Design Document；
- 把 gather-based Paged KV 声称为完整 PagedAttention CUDA Kernel。

---

# 补充条款 A：运行环境与设备抽象（强制）

## 背景
开发环境**没有 NVIDIA GPU**（本机仅 Intel 核显，无 CUDA）。
Task 01–11 必须能在 **CPU 上完整运行并通过测试**；只有 Task 12 的 GPU benchmark 需要上云。

## A1. 禁止硬编码设备（最重要）
代码中禁止出现以下任何形式：
```python
device = "cuda"
x = x.cuda()
model.to("cuda")
torch.zeros(..., device="cuda")
```
所有设备必须统一从配置读取：
```python
EngineConfig.device   # 本机 "cpu"，云端 "cuda"
```
建议提供统一入口，例如 `get_device(config)`，并写一条测试断言：
**全仓库 grep 不到裸 `"cuda"` 字面量**。

## A2. dtype 由配置决定，不写死
```python
EngineConfig.dtype    # 本机 torch.float32，云端 torch.float16
```
- CPU 上用 **FP32**：CPU 的 FP16 支持差且反而更慢；
- GPU 上用 **FP16**；只有确认是 A10/A100（Ampere+）时才可用 `bfloat16`；
- **T4 与 P100 不支持 BF16**，若云端拿到这两种卡，必须降级为 FP16。

## A3. GPU 相关 API 必须可降级
任何 `torch.cuda.*` 调用都要有 CPU fallback，不允许在无 GPU 时抛异常：
```python
def peak_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return None          # 或退回 psutil 读取进程 RSS
```
Metrics 模块中显存类指标在 CPU 下返回 `None`，报告里标注 `N/A (no GPU)`，**不要填 0**（会误导）。

## A4. 数值对齐容差按设备区分
| 环境 | dtype | 建议容差 |
|---|---|---|
| 本机 CPU | FP32 | `atol=1e-4, rtol=1e-4` |
| 云端 GPU | FP16 | `atol=1e-2, rtol=1e-2` |
两套阈值都要写进测试配置，并用 top-1 agreement 作为补充判据。

## A5. 磁盘与缓存
- 本机 C 盘仅剩约 10GB：**venv 与模型缓存必须放在 D 盘**；
- 设置 `HF_HOME` 指向非系统盘；
- 国内环境优先用 ModelScope 或 `HF_ENDPOINT=https://hf-mirror.com` 下载 Qwen；
- PyTorch 装 **CPU 版**，不要装 CUDA 版（省约 2.3GB）。

## A6. 每个 Task 必须声明运行环境
交付时在 Design Document 中写明：
```text
运行环境：CPU / GPU / 两者皆可
本机验证命令：...
```
Task 01–11 的验收命令**必须能在本机 CPU 上直接跑通**。

---

# 三、每个 Task 的交付格式

每完成一个 Task，必须同时输出：

1. 源代码；
2. Unit Tests；
3. Design Document；
4. 最小运行示例；
5. 关键数据结构解释；
6. 为什么这样设计；
7. Alternative 方案；
8. 当前 Known Limitations；
9. 进入下一阶段前的验收命令。

上一阶段测试不通过，不进入下一阶段。

---

# 四、任务拆分

## 运行环境总表
| Task | 内容 | 运行环境 |
|---|---|---|
| 01–02 | 骨架 / Baseline / Manual Gen | 本机 CPU |
| 03–04 | Minimal Qwen / KV Cache | 本机 CPU（对齐用 FP32） |
| 05–08 | Engine / Scheduler / Paged KV / Runner | 本机 CPU |
| 09–11 | API / Metrics / Prefix Cache | 本机 CPU |
| 12 | **Benchmark + Ablation** | **云端 GPU（必需）** |
| 13–14 | Docker / CI / 接入知微 | 本机 CPU |

**Task 01–11 必须能在本机 CPU 上跑通并验收，不得要求 GPU。**

---

## Task 01：项目骨架 + HF Baseline
交付：
- pyproject
- config
- loader
- baseline
- README 初稿

验收：
Qwen2.5-0.5B 能正常生成。

---

## Task 02：Manual Generation Loop + Sampling
实现：
- forward loop
- greedy
- temperature
- top-k
- top-p
- EOS

禁止：
最终使用 `generate()`。

验收：
greedy 输出与 HF baseline 基本一致。

---

## Task 03：Minimal Qwen Decoder
实现：
- embedding
- rmsnorm
- rope
- gqa attention
- swiglu
- decoder layer
- lm head

验收：
逐层 allclose / top-1 agreement。

本机 CPU 补充：
- 用 **FP32** 对齐，建议 `atol=1e-4, rtol=1e-4`；
- 同时统计 top-1 agreement，比纯 allclose 更能反映生成是否一致；
- 上云改用 FP16 时容差放宽到 `atol=1e-2`，两套阈值都要写进测试配置。

---

## Task 04：Contiguous KV Cache
实现：
- prefill
- decode
- past KV
- latency benchmark

验收：
KV 版本输出正确，性能优于 no-cache baseline。

本机 CPU 补充：
- 正确性部分（KV 版本输出与 no-cache 一致）必须在 CPU 上验证通过；
- 性能对比在本机也能跑，但 CPU 上的加速比**不能写进简历**，仅作逻辑验证；
- benchmark 脚本需支持 `--device cpu|cuda` 参数。

---

## Task 05：Request + Engine Core
实现：
- request model
- request registry
- request states
- engine loop

验收：
可同时维护多个请求。

---

## Task 06：Continuous Batching Scheduler
实现：
- waiting queue
- running queue
- FCFS
- token budget
- sequence budget
- dynamic batch

验收：
8~16 个请求动态加入退出。

---

## Task 07：BlockPool + Paged KV
实现：
- KVBlock
- BlockPool
- FreeQueue
- BlockTable
- allocate
- append
- free
- usage

允许：
PyTorch gather。

必须说明：
这不是完整 PagedAttention Kernel。

验收：
10000 次 stress test 无泄漏。

---

## Task 08：ModelRunner 接入 Paged KV
实现：
- prefill runner
- decode runner
- block metadata
- gather/read/write KV

验收：
多请求输出正确。

---

## Task 09：Async Engine + Streaming API
实现：
- asyncio
- FastAPI
- SSE
- OpenAI-compatible schema
- cancellation

验收：
OpenAI Python Client 可直接调用。
Client disconnect 后 KV 正确回收。

---

## Task 10：Metrics + Tracing
实现：
- TTFT
- TPOT
- ITL
- E2E
- running/waiting
- KV utilization
- output tokens/s
- request trace

验收：
一次请求可输出完整时间线。

---

## Task 11：Prefix Cache
实现：
- block hash
- hash table
- ref count
- reuse
- eviction/free policy

验收：
共享前缀 workload 中可观察 cache hit。

---

## Task 12：Benchmark + Ablation　【运行环境：云端 GPU，必需】

必须先确认：**Task 01–11 已在本机 CPU 全部跑通**。云端只做 benchmark，不用来 debug。

必须包含：
- HF baseline
- no-cache
- KV
- continuous batching
- paged KV
- prefix cache

并发：
1/2/4/8/16/32

workloads：
- 64→256
- 2048→32
- 512→128
- shared prefix

输出：
- csv/json
- chart
- benchmark report

上云执行约定：
- 代码通过 Git 同步（push → 云端 clone → 跑完 push → 本机 pull），不手工传文件；
- 优先魔搭 ModelScope（A10 24GB，免费额度，国内直连）；
  备选 AutoDL（1.5–2 元/时）；Kaggle 需注意 T4 不支持 BF16，只用 FP16；
- 时间预算 8–15 GPU 小时，上云前先写好一键脚本，**一轮跑完所有配置**；
- 显存类指标若环境无 GPU，输出 `N/A (no GPU)`，禁止填 0；
- 跑完立即把原始 CSV/JSON 与图表 commit 回仓库，云端随时可能被回收；
- 每张图必须附实验记录（GPU 型号、CUDA、PyTorch、dtype、block_size、并发数）。

---

## Task 13：Docker + CI + README
必须包含：
- Dockerfile
- pytest
- CI
- architecture
- known limitations
- benchmark method
- benchmark result

---

## Task 14：接入知微
把知微的 LLM Provider：
```text
Qwen/DeepSeek external API
```

扩展为：
```text
LiteInfer OpenAI-Compatible API
```

用于完整 Demo。

---

# 五、代码质量要求
- 核心类有 type hints；
- 关键方法有 docstring；
- 配置集中管理；
- 禁止大段全局状态；
- Scheduler / Cache / ModelRunner 解耦；
- 测试独立；
- benchmark 和 production path 分离；
- 日志避免在热路径大量 print。

---

# 六、最终必须提供的设计文档
至少：
- architecture.md
- generation_loop.md
- kv_cache.md
- scheduler.md
- paged_kv.md
- async_serving.md
- benchmark.md
- known_limitations.md
- interview_notes.md

---

# 七、最终性能数字规则
任何：
- x 倍提升
- xx% 提升
- P95 xx ms
- xx tokens/s

必须来自真实 benchmark 输出。
不得提前硬编码进 README 或简历。
