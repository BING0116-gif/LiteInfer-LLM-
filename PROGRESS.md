# LiteInfer 开发进度

> **每个新开发窗口的 AI 先读这个文件**，它记录了当前进度、已有文件与关键接口。
> 每完成一个 Task，把 AI 给出的记录追加到下方「Task 记录」章节。

## 总体进度

| Task | 内容 | 运行环境 | 状态 | Commit |
|---|---|---|---|---|
| 01 | 项目骨架 + HF Baseline | CPU | ✅ 已完成 | |
| 02 | Manual Generation Loop + Sampling | CPU | ✅ 已完成 | |
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

## Task 01：项目骨架 + HF Baseline（2026-09-08 补录）

**新增文件**

```text
pyproject.toml
liteinfer/config.py               # EngineConfig 集中配置
liteinfer/device.py               # get_device / resolve_dtype / peak_memory_mb
liteinfer/model/loader.py         # load_model_and_tokenizer
liteinfer/model/baseline.py       # HFBaseline（全项目唯一 generate() 调用点）
tests/test_config.py / test_device.py / test_no_hardcoded_cuda.py / test_baseline.py
examples/baseline_demo.py
```

**关键接口签名**

```python
EngineConfig(model_id, device="cpu", dtype=torch.float32, hf_cache_dir=None,
             max_new_tokens=64, seed=42, trust_remote_code=False, local_files_only=False)
EngineConfig.from_env(**overrides)
load_model_and_tokenizer(cfg) -> LoadedModel(model, tokenizer, config, device, dtype)
HFBaseline.from_config(cfg) / .generate(prompt, max_new_tokens=None, greedy=True) -> GenerationOutput
```

**验收命令**：`pytest -m model -q`（Qwen2.5-0.5B 正常生成，CPU FP32）

**踩过的坑**

- Windows 无开发者模式时 HF symlink 静默失败留 0 字节 snapshot，需设
  `HF_HUB_DISABLE_SYMLINKS=1`（config.py 里已 setdefault）
- transformers 4.56 起 `torch_dtype` 改名 `dtype`，loader 用 try/except 双兼容
- HF 缓存目录要传 `<HF_HOME>/hub`（对齐 huggingface_hub 规范布局），
  直接传根目录会同模型在磁盘存两份

---

## Task 02：Manual Generation Loop + Sampling（2026-09-08）

**新增文件**

```text
liteinfer/sampling/params.py          # SamplingParams（frozen + 构造期校验）
liteinfer/sampling/sampler.py         # Sampler：T→top-k→top-p→softmax→multinomial
liteinfer/model/generator.py          # ManualGenerator 手写循环（生产路径）
tests/test_sampler.py                 # 19 条纯 logits 单测（0.09s）
tests/test_manual_generation.py       # 7 条端到端（marker=model）
examples/generation_demo.py           # greedy/sample 两模式 demo
docs/design/generation_loop.md        # Task 02 设计文档
```

**修改文件**

```text
liteinfer/__init__.py                 # 惰性导出 SamplingParams/Sampler/ManualGenerator
```

**关键接口签名**

```python
SamplingParams(max_tokens, temperature=1.0, top_k=-1, top_p=1.0, seed=None)  # T=0 即 greedy
Sampler().sample(logits: Tensor[vocab], params, generator=None) -> int
ManualGenerator(model, tokenizer, cfg) / .from_config(cfg)
ManualGenerator.generate(prompt, params=None) -> GenerationOutput  # params=None → greedy
GenerationOutput(text, prompt_tokens, output_tokens, finish_reason,
                 latency_s, tokens_per_s, device, dtype)
```

**验收命令**（CPU 全部跑通）

```bash
pytest tests/test_sampler.py -q   # 19 passed
pytest -m model -q                # 11 passed（greedy 与 HF baseline 逐字一致）
python examples/generation_demo.py --mode greedy
```

**踩过的坑**

- Qwen2.5 的 EOS 是 `generation_config.eos_token_id`（<|im_end|>=151645），
  不是 `tokenizer.eos_token_id`，对齐 HF 时必须优先取前者（且可能是 list）
- 采样统计类测试（"所有候选都应出现"）必须用概率相近的 logits 构造，
  概率悬殊时尾部 token 有限次采样抽不到属正常，会误报失败
- examples 脚本直接 `python xxx.py` 找不到 liteinfer 包，需
  `set PYTHONPATH=d:\LiteInfer`
- 实际环境 transformers 5.14.1，loader 的 dtype/torch_dtype 双参数兼容已覆盖

**身份转变声明**：HFBaseline 自 Task 02 起降级为对齐参照物，退出生产路径；
后续生产推理走 ManualGenerator（Task 04 起由 ModelRunner 接管 forward 策略）。

---

<!-- 后续每个 Task 完成后，把 AI 生成的记录追加到这里 -->
