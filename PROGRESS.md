# LiteInfer 开发进度

> **每个新开发窗口的 AI 先读这个文件**，它记录了当前进度、已有文件与关键接口。
> 每完成一个 Task，把 AI 给出的记录追加到下方「Task 记录」章节。

## 总体进度

| Task | 内容 | 运行环境 | 状态 | Commit |
|---|---|---|---|---|
| 01 | 项目骨架 + HF Baseline | CPU | ✅ 已完成 | |
| 02 | Manual Generation Loop + Sampling | CPU | ✅ 已完成 | |
| 03 | Minimal Qwen Decoder | CPU（FP32 对齐） | ✅ 已完成 | |
| 04 | Contiguous KV Cache | CPU | ✅ 已完成 | |
| 05 | Request + Engine Core | CPU | ✅ 已完成 | |
| 06 | Continuous Batching Scheduler | CPU | ✅ 已完成 | |
| 07 | BlockPool + Paged KV | CPU | ✅ 已完成 | |
| 08 | ModelRunner 接入 Paged KV | CPU | ✅ 已完成 | |
| 09 | Async Engine + Streaming API | CPU | ✅ 已完成 | |
| 10 | Metrics + Tracing | CPU | ✅ 已完成 | |
| 11 | Prefix Cache | CPU | ✅ 已完成 | |
| 12 | Benchmark + Ablation | **云端 GPU** | ✅ 已完成 | |
| 13 | Docker + CI + README | CPU | ⬜ 未开始 | |
| 14 | 接入知微 | CPU | ⬜ 未开始 | |

## 环境备忘

- 开发机：**无 NVIDIA GPU**，Task 01–12 工具链必须在 CPU 上跑通；
  Task 12 全矩阵（5 引擎 × 4 workload × 6 并发）上云执行，CPU 用 smoke 口径验收
- Python 3.12.10（venv 在 `D:\LiteInfer\.venv`）；PyTorch 装 CPU 版（torch 2.14.0+cpu）
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

## Task 03：Minimal Qwen Decoder（2026-09-08）

**新增文件**

```text
liteinfer/model/minimal/            # __init__/rmsnorm/rotary/attention/mlp/layer/model/weights
liteinfer/model/alignment.py        # 两套容差（CPU 1e-4 / GPU 1e-2）+ top1_agreement
tests/test_minimal_operators.py     # 16 条算子单测（无模型，~10s）
tests/test_minimal_alignment.py     # 7 条对齐测试（marker=model）
docs/design/minimal_qwen_decoder.md
examples/decoder_demo.py
```

**修改文件**

```text
liteinfer/__init__.py               # 惰性导出 MinimalQwenForCausalLM / load_minimal_from_hf / alignment_tolerances
```

**关键接口签名**

```python
load_minimal_from_hf(cfg) -> MinimalLoaded(minimal, hf_model, device, dtype)
MinimalQwenForCausalLM(input_ids, position_ids=None) -> logits  # 无 KV Cache 全序列 forward
build_causal_mask(seq_len, device, dtype) -> [1,1,S,S]          # additive, finfo.min
alignment_tolerances(device) -> (atol, rtol); top1_agreement(a, b) -> float
```

**验收命令**（CPU 全部跑通）

```bash
pytest tests/test_minimal_operators.py -q   # 16 passed
pytest -m model -q                          # 18 passed（含 Task 01/02 不回归）
python examples/decoder_demo.py             # logits max|diff|=2.2e-5, top-1=100%
```

**踩过的坑**

- Qwen2.5-0.5B **无 QK-Norm**（transformers 5.14 Qwen2 模块无 q_norm/k_norm，
  safetensors 仅 290 键）；QK-Norm 是 Qwen3 系的，代码已按权重键存在性自适应
- Qwen2.5-0.5B **是 tied embedding**（tie_word_embeddings=true），对齐测试的
  参数量对比要用 state_dict numel 口径，不能用 parameters()
- transformers 5.x 标准 RoPE 也落盘 rope_scaling={'rope_type': 'default'}，
  判变体要查 rope_type 而非判空
- 5.x 的 attention forward hook 输出是 (output, attn_weights) 元组
- 模块命名镜像 HF 后键映射是恒等的，不要再"聪明地"剥 model. 前缀
- 设备查表键（如容差表）也会被 test_no_hardcoded_cuda 逮住，需拼接构造

---

## Task 04：Contiguous KV Cache（2026-09-08）

**新增文件**

```text
liteinfer/cache/__init__.py          # 导出 KVCacheConfig / LayerKVCache / ContiguousKVCache
liteinfer/cache/contiguous.py        # 连续 KV 缓存本体：配置(字节公式/容量)、单层视图(append 原地写入+零拷贝读)、整模型缓存
liteinfer/model/eos.py               # resolve_eos_ids(model, tokenizer) -> frozenset[int]，Task 02/04 共用唯一真相
liteinfer/model/cached_generator.py  # CachedGenerator(prefill/decode 主链) + CachedGenerationOutput(继承 GenerationOutput)
tests/test_kv_cache.py               # 19 条无模型快测（字节/append+read/mask 偏移/容量/小模型整段=增量）
tests/test_kv_generation.py          # 8 条真实模型测试（marker=model）
benchmark/kv_cache_benchmark.py      # --device/--max-new-tokens/--repeat/--warmup/--json，两模式同模型对照
examples/kv_cache_demo.py            # 最小运行示例
docs/design/kv_cache.md              # 设计文档（9 项交付 + Alternative + Known Limitations）
```

**修改文件**

```text
liteinfer/model/minimal/attention.py # forward 增 kv_cache/write_pos；RoPE 后原地写入缓存并取回完整视图
liteinfer/model/minimal/layer.py     # 透传 kv_cache/write_pos
liteinfer/model/minimal/model.py     # build_causal_mask 增 past_len；两个 forward 增 kv_caches/write_pos
liteinfer/model/generator.py         # _resolve_eos_ids 委托 eos.resolve_eos_ids（行为不变）
liteinfer/model/minimal/weights.py   # MinimalLoaded 补 tokenizer 字段，load_minimal_from_hf 一并返回
liteinfer/__init__.py                # 惰性导出 KVCacheConfig / ContiguousKVCache / CachedGenerator
liteinfer/model/minimal/__init__.py  # 模块 docstring 更新（Task 04 已复用 attention/layer）
```

**关键接口签名**

```python
KVCacheConfig(num_layers, num_kv_heads, head_dim, max_seq_len, dtype, device)  # frozen
    .bytes_per_token() -> int       # 2 × L × KVH × D × itemsize
    .total_bytes()    -> int
    .from_hf_config(hf_cfg, max_seq_len, dtype, device)  # 无 head_dim 时按 hidden//num_heads 推导
LayerKVCache.append(k_new[1,n,KVH,D], v_new, start) -> (k[1,start+n,KVH,D], v)  # 视图片段，零拷贝历史
LayerKVCache.read(length) -> (k[1,length,KVH,D], v)
ContiguousKVCache(cfg).layer_caches -> list[LayerKVCache]; .nbytes ; .reset()
build_causal_mask(seq_len, device, dtype, past_len=0) -> [1,1,S,S+past_len]
QwenSelfAttention.forward(h, position_ids, attention_mask=None, kv_cache=None, write_pos=0) -> Tensor
QwenDecoderLayer.forward(h, position_ids, attention_mask=None, kv_cache=None, write_pos=0) -> Tensor
MinimalQwenForCausalLM.forward(input_ids, position_ids=None, kv_caches=None, write_pos=0) -> logits  # 始终只返回 logits
CachedGenerator(model, tokenizer, cfg, eos_ids=None) / .from_config(cfg)
CachedGenerator.generate(prompt, params=None, use_cache=True, max_seq_len=None) -> CachedGenerationOutput
    # 字段：text, prompt_tokens, output_tokens, finish_reason, latency_s, tokens_per_s, device, dtype,
    #       prefill_latency_s, decode_latency_s, cached_tokens, cache_bytes
resolve_eos_ids(model, tokenizer) -> frozenset[int]
```

**验收命令**（CPU 全部跑通）

```bash
set PYTHONPATH=d:\LiteInfer
pytest tests/test_kv_cache.py -q          # 19 passed（无模型，~8s）
pytest tests/test_no_hardcoded_cuda.py -q # 1 passed（benchmark/ 未引入裸 cuda 字面量）
pytest -q                                 # 74 passed（含 Task 01~03 不回归）
pytest -m model -q                        # 26 passed（KV 对齐 + 端到端逐字一致 + Task 01/02/03 不回归）
python examples/kv_cache_demo.py          # 输出一致 True，加速比 > 1（本机 ~1.6~1.9x）
python benchmark/kv_cache_benchmark.py --device cpu --max-new-tokens 16 32
```

实测（Qwen2.5-0.5B, CPU FP32, repeat=2, warmup=1）：

```text
[len= 16] no-cache   7.806s ( 2.07 tok/s) | kv-cache   4.766s ( 3.46 tok/s) | speedup 1.64x
[len= 32] no-cache  18.695s ( 1.71 tok/s) | kv-cache   9.708s ( 3.63 tok/s) | speedup 1.93x
peak memory: N/A (no GPU)
```

**踩过的坑**

- **CPU 冷启动抖动**：首次 forward 含线程池初始化/内存分配/算子选择，比热态慢 3~5 倍
  （demo 首次 8s vs 重复测量 median 4.8s@len16）。不 warmup 时 benchmark 测到的是"谁先跑"，
  不是谁的 KV 复用更好；强制 `--warmup` + `--repeat` + 中位数。模型测试 fixture 也用 module
  scope + 2-token warmup 预热，否则速度断言会假阴性。
- **inference tensor 陷阱**：`torch.inference_mode()` 内新建的张量是 inference tensor，之后在
  外部对其原地 `copy_` 会抛 RuntimeError。缓存必须在进入 `inference_mode()` **之前**分配，
  decode 循环内只做 `copy_`。
- **decode 的 position_ids 必须是绝对位置** `[[cached_len]]`：RoPE 是绝对位置编码，从 0 开始
  会让续写整体错位；测试 `test_decode_needs_absolute_position_ids` 钉死。
- **缓存存旋转后的 K**（RoPE 之后写入）：存旋转前的 K 会让 decode 时历史需重算旋转，错误表现
  隐蔽（文本"看着还行"），用 `test_cache_stores_rotated_keys` 数值钉住；并反向 assert 非旋转前 K。
- **GQA 缓存按 num_kv_heads(2) 而非 Q 头(14)**：bytes_per_token = `2×L×KVH×D×itemsize`，
  按 Q 头存会放大 7 倍显存。
- **同模型消融，不拿 HF 比性能**：`use_cache=False` 跑同一个 MinimalQwen，只留 KV 复用一个变量；
  HF 仅作正确性参照（`CachedGenerator` 文本与 `ManualGenerator(HF)` 逐字一致）。
- **EOS 真相共用**：Qwen2.5 终止符是 `generation_config.eos_token_id`=<|im_end|>(151645)，不是
  tokenizer 的 151643；抽成 `eos.resolve_eos_ids` 让两条链共用，避免"长度不同"伪装成数值问题。
- **Windows GBK 控制台无法打印非 ASCII 符号**（如 ✓）：benchmark 改用 ASCII "OK"。
- 测试里原写 `torch.shares_memory`（不存在）→ 改用
  `k.untyped_storage().data_ptr() == layer.k_buf.untyped_storage().data_ptr()` 判断共享内存。
- **prefill/decode 拆两段计时**（而非合一段）：两段瓶颈不同（prefill compute-bound、decode
  memory-bound），合起来会掩盖 TTFT/TPOT 差异；Task 10 指标也按两段统计。

**下一阶段提示（Task 05/06/07）**

- 缓存生命周期已外部化（generator 持有），Task 07 把它换成 block（paged）即可，生成循环不动。
- 容量按 `prompt_len + max_tokens` 逐请求预分配 → 这是 contiguous 的内部/外部碎片来源，正是
  Task 07 分页与共享 block 的动机；decode 每步仍读全量历史 KV（O(T)），真实 PagedAttention 属 08。
- 模型始终只返回 logits（不返回 past_key_values），Task 03 的 `assert_close(mine, hf)` 断言零改动全绿。

---

## Task 05：Request + Engine Core（2026-09-08）

**新增文件**

```text
liteinfer/engine/__init__.py     # 导出 Request/RequestStatus/RequestRegistry/EngineCore 等
liteinfer/engine/request.py      # Request / RequestStatus / RequestState / RequestRegistry / RequestOutput / RequestStepResult
liteinfer/engine/core.py         # EngineCore：submit/step/run/cancel/get_request
tests/test_engine.py             # 快测（request/registry + 假模型状态机）+ model 标记（与 CachedGenerator 逐字一致 + 多请求）
examples/engine_demo.py          # 提交 3 个并发 prompt，run 并对照 CachedGenerator
docs/design/engine_core.md       # 设计文档（9 项交付 + Alternative + Known Limitations）
```

**修改文件**

```text
liteinfer/__init__.py            # 惰性导出 EngineCore / Request / RequestStatus / RequestRegistry
liteinfer/model/loader.py        # _load_tokenizer：fast 失败回退 use_fast=False（修复本机 transformers 5.14.1 tokenizers bug）
PROGRESS.md                      # 本记录 + 进度表 05 置 ✅
```

**关键接口签名**

```python
Request(request_id, prompt, params, status=WAITING, prompt_tokens=0,
        generated=[], finish_reason=None, output_text="")
RequestStatus              # WAITING / PREFILL / DECODE / FINISHED / CANCELLED（str 枚举）
RequestRegistry.add/get/remove/active()/count_by_status()
EngineCore(model, tokenizer, cfg, eos_ids=None) / .from_config(cfg)
EngineCore.submit(prompt, params=None) -> str            # request_id
EngineCore.step() -> list[RequestStepResult]             # 所有在飞请求各推进一个 token
EngineCore.run(max_steps=None) -> dict[str, RequestOutput]
EngineCore.cancel(request_id) / .get_request(id) / .active_requests()
RequestOutput(request_id, text, prompt_tokens, output_tokens, finish_reason,
             latency_s, tokens_per_s, device, dtype,
             prefill_latency_s, decode_latency_s, cached_tokens, cache_bytes)
```

**验收命令**（CPU 全绿）

```bash
set "HF_HOME=D:\LiteInfer\hf_cache"
set PYTHONPATH=d:\LiteInfer
pytest tests/test_engine.py -q -m "not model"   # 12 passed（无模型）
pytest -m model -q                               # 含 Task 05 共 2 passed，且与 Task 01-04 不回归（28 passed）
pytest -q                                       # 86 passed（全快测不回归）
pytest tests/test_no_hardcoded_cuda.py -q        # 1 passed（无裸 cuda）
python examples/engine_demo.py                   # 3 并发请求全部与 CachedGenerator 逐字一致
```

**踩过的坑**

- 本机 `sentencepiece` 缺失 + `transformers 5.14.1 + tokenizers 0.22.2` 的 fast tokenizer
  后端构建 bug（`Couldn't instantiate the backend tokenizer`），导致**仓库内所有 model 标记测试**
  无法加载 tokenizer（连 Task 04 既有的 test_kv_generation 也一并挂掉）。两步修复：
  (1) `pip install sentencepiece`；(2) `loader._load_tokenizer` 先试 fast、失败回退
  `use_fast=False`（tiktoken 后端，CPU 上编码结果一致，仅速度略慢）。正常环境仍优先 fast。
- `set HF_HOME=D:\LiteInfer\hf_cache` 不带引号会把尾随空格带进变量，缓存目录变
  `hf_cache \hub` → 找不到本地权重/tokenizer，回退后去网络下载。务必 `set "HF_HOME=..."`（带引号）。
- 编辑 `loader.py` 时曾把 `model = _from_pretrained(...)` 等几行误吞进 `_load_tokenizer`
  函数体，导致 `load_model_and_tokenizer` 隐式返回 None → `loaded.model` 抛 AttributeError；
  已把模型加载与 return 挪回原函数。教训：替换大段代码后务必读回确认函数边界。
- 假模型必须按「输入 token 的纯函数」产出 next token，否则多请求共享单模型会串台；用
  `nxt=(last+1)%vocab` 保证顺序无关、互不干扰，能真正验证并发维护。

**下一阶段提示（Task 06）**

- 本 Task 对多请求是「时间片交错」（每 step 推进所有在飞请求一个 token），不具备 waiting/running
  队列、FCFS、token/sequence budget 准入与抢占——这些属 Task 06 Scheduler。
- `EngineCore.step()` 已是「schedule 出本批活跃请求 + 逐个 execute」的形状，Task 06 把它内部的
  "全部 active 都步进"替换为 Scheduler 的准入决策即可，引擎主循环不动。
- 每个 `RequestState` 已持有专属 `ContiguousKVCache`，Task 07 仅把该字段换成 paged block。

---

## Task 06：Continuous Batching Scheduler（2026-09-08）

**新增文件**

```text
liteinfer/scheduler/__init__.py     # 导出 Scheduler / SchedulerConfig / ScheduledBatch / SchedulerRequestInfo
liteinfer/scheduler/config.py       # SchedulerConfig（max_num_seqs / max_num_batched_tokens，frozen）
liteinfer/scheduler/scheduler.py    # Scheduler：waiting/running 双队列 + FCFS 准入 + 双预算；ScheduledBatch/SchedulerRequestInfo
tests/test_scheduler.py             # 纯调度单测（无模型）
tests/test_engine_scheduler.py       # 引擎集成：fake 模型 16 请求动态批 + 真模型与 CachedGenerator 逐字一致
examples/scheduler_demo.py          # 最小运行示例：seq budget 限制下逐步准入
docs/design/scheduler.md            # 设计文档（9 项交付 + Alternative + Known Limitations）
```

**修改文件**

```text
liteinfer/config.py                 # EngineConfig 增加 scheduler: SchedulerConfig 字段（集中配置）
liteinfer/engine/core.py            # submit 入 waiting 队列 + token budget fail-fast；step 委托 Scheduler.schedule；cancel 从调度器移除；新增 _snapshot()
liteinfer/__init__.py               # 惰性导出 Scheduler / SchedulerConfig
PROGRESS.md                         # 本记录 + 进度表 06 置 ✅
```

**关键接口签名**

```python
SchedulerConfig(max_num_seqs: int = 16, max_num_batched_tokens: int = 2048)  # frozen
Scheduler(cfg: SchedulerConfig)
Scheduler.enqueue(request_id)                       # 提交 -> waiting 队尾（FCFS）
Scheduler.schedule(requests: dict[str, SchedulerRequestInfo]) -> ScheduledBatch
Scheduler.remove(request_id)                        # 取消/回收 -> 从两队列移除
Scheduler.num_waiting / .num_running               # 观测属性
SchedulerRequestInfo(request_id, prompt_len, output_len, status)
ScheduledBatch(prefill_ids, decode_ids)             # .all_ids = prefill + decode
# EngineCore 变化：
EngineCore.scheduler: Scheduler                     # 由 cfg.scheduler 构造
EngineCore.submit(...) -> rid                       # 入队 waiting，不再立即"在飞"
EngineCore.step() -> list[RequestStepResult]        # 只推进 Scheduler 放行的本批
```

**验收命令**（CPU 全绿）

```bash
set "HF_HOME=D:\LiteInfer\hf_cache"
set PYTHONPATH=d:\LiteInfer
pytest tests/test_scheduler.py -q                          # 纯调度单测全绿
pytest tests/test_engine_scheduler.py -q -m "not model"    # fake 模型 16 请求动态批全绿
pytest tests/test_no_hardcoded_cuda.py -q                  # 无裸 cuda 字面量
pytest -m model -q                                        # 真模型：scheduler 下与 CachedGenerator 逐字一致
pytest tests/test_engine.py -q                             # Task 05 不回归
pytest -q                                                 # 全快测不回归
python examples/scheduler_demo.py                         # 展示 seq budget 限制下的逐步准入
```

实测（Qwen2.5-0.5B, CPU FP32, fake 模型 16 请求 + 真模型逐字一致）：

```text
pytest 全快测: 104 passed, 30 deselected
pytest -m model: 30 passed（含 Task 06 与 Task 01-05 不回归）
scheduler_demo: running 恒 <=3（max_num_seqs=3），waiting 随 finished 释放 slot 而清空
```

**踩过的坑**

- **循环导入**：scheduler.py 在模块顶层 `from liteinfer.engine.request import RequestStatus`
  会触发 `engine/__init__` -> `core` -> `scheduler`（半初始化），ImportError。改为在
  `schedule()` 内做局部 import；类型注解靠 `from __future__ import annotations` 惰性求值，
  模块顶层无需真正导入 RequestStatus。
- **快照状态一致性**：调度器只认 `SchedulerRequestInfo` 快照里的 status，不反向持有引擎状态；
  submit 不立即"在飞"，而是进 waiting，由 schedule 决定何时准入——否则序列预算无法生效。
- **token budget 只闸门 NEW prefill**：running 的 decode 请求每步必被调度（cost=1 必放得下），
  预算主要用于限制单步能 prefill 多少 token，避免超长 prompt 占满整步饿死其它请求。
- **fail-fast 而非静默饿死**：prompt 长度 > max_num_batched_tokens 时 submit 直接抛
  ValueError，比排进队列永不准入更易排查。
- **engine_demo 无需改动**：默认 SchedulerConfig 预算足够大，Task 05 行为不变（已验证不回归）。

**下一阶段提示（Task 07）**

- 每个 `RequestState.cache` 仍是 `ContiguousKVCache`；Task 07 仅把该字段换成 paged block
  （KVBlock/BlockPool/BlockTable），`step` 循环与调度逻辑完全不动。
- 终态请求已能从 running 集合释放 slot（调度层面）；真正的 block 级显存回收属 Task 07。
- Scheduler 产出的 `ScheduledBatch` 已是「本步该跑哪些请求」的明确边界，Task 08 ModelRunner
  把它合并为一次 batch 前向即可。

---

## Task 07：BlockPool + Paged KV（2026-09-08）

**新增文件**

```text
liteinfer/cache/paged.py          # FreeQueue / BlockPool / KVBlock / BlockTable / PagedKVCache
tests/test_paged_kv.py            # 16 条单测 + 10000 次无泄漏压测（无模型，~48s）
examples/paged_kv_demo.py         # 多请求 append/读回/free 最小示例
docs/design/paged_kv.md           # 设计文档（9 项交付 + Alternative + Known Limitations）
```

**修改文件**

```text
liteinfer/cache/__init__.py       # 导出 FreeQueue/BlockPool/KVBlock/BlockTable/PagedKVCache
liteinfer/__init__.py             # _CACHE_NAMES 加入 PagedKVCache（惰性导出）
```

**关键接口签名**

```python
KVCacheConfig(...)                         # 复用 Task 04：num_layers/num_kv_heads/head_dim/dtype/device
FreeQueue(total) / .pop() / .push(idx) / .size / .empty / .total
BlockPool(cfg, block_size, num_blocks)
    .alloc(n)->list[int]  .free(ids)  .num_total  .num_free  .num_used  .usage()  .nbytes
KVBlock(pool, block_id)
    .write_token(layer, offset, k_vec[KVH,D], v_vec)  .k_view(layer, offset)  .v_view(...)
BlockTable(pool, block_size, blocks=[], num_tokens=0)
    .append(tokens_k[Layers,n,KVH,D], tokens_v)->int(新分配块数)
    .gather(layer, length)->[1, length, KVH, D]        # torch.gather 沿 block+offset 选槽
    .free()  .num_tokens  .__len__
PagedKVCache(cfg, block_size, num_blocks)
    .new_block_table()->BlockTable  .append_tokens(table,k,v)->int
    .free_table(table)  .num_blocks_total/free/used  .usage()  .nbytes
```

**验收命令**（CPU 全绿）

```bash
set "HF_HOME=D:\LiteInfer\hf_cache"
set PYTHONPATH=d:\LiteInfer
python -m pytest tests/test_paged_kv.py -q        # 16 passed（含 10000 次压测，~48s）
python -m pytest tests/test_no_hardcoded_cuda.py -q  # 1 passed
python -m pytest -q                               # 120 passed（Task 01-06 不回归，30 model deselected）
python examples/paged_kv_demo.py                  # 多请求读回逐位一致，释放后 free==total（零泄漏）
```

实测：

```text
tests/test_paged_kv.py: 16 passed in 48.10s
test_stress_10000_no_leak: 随机 10000 次建表→写随机长度→gather 逐位相等→释放，结束 num_free==total
pytest -q: 120 passed, 30 deselected
paged_kv_demo: [final] used=0 free=32  -> OK 零泄漏
```

**踩过的坑**

- **`write_seq` shape 误用**：`append` 按 token 逐层写，单 token 单层 K/V 形状是 `[KVH, D]`
  （不是 `[n, KVH, D]`），最初把 `tokens_k[layer, t]` 当 `[n, ...]` 传入导致 `n=KVH=2`、
  偏移越界抛 ValueError；改为 `write_token(layer, offset, k_vec, v_vec)` 明确单 token 语义。
- **gather 的 expand 维度**：`torch.gather(K_src, 0, idx_b)` 的输出样本维是 `length`，不是
  `num_blocks`；`idx_b` 必须 `expand(length, block_size, KVH, D)` 而非 `expand_as(K_src)`，
  否则在 dim0（10 vs 4）报 "expanded size must match"。输出形状 `[L, block_size, ...]` 才对。
- **双重释放检测**：FreeQueue 用 `set` 做成员判定，`push` 时若下标已在空闲集合即抛 ValueError；
  没有这层防护，double free 会虚高 free 计数、掩盖真实泄漏，压测会"假绿"。
- **`expand` 非零维限制**：`sel_b.view(-1,1,1,1)` 只能把 dim0 外的维 expand 到目标大小，
  dim0 必须等于 length，故 gather 输入 `idx_b` 形状固定为 `(L, bs, KVH, D)`。
- **docstring 反斜杠转义警告**：`examples` 文档里的 `D:\LiteInfer\hf_cache` 触发
  `SyntaxWarning: invalid escape sequence`，改用正斜杠 `D:/LiteInfer/hf_cache` 消除。

**下一阶段提示（Task 08）**

- 本 Task 已交付「缓存管理层 + gather 读接口」，`BlockTable.gather(layer, length)` 输出形状
  `[1, length, KVH, D]` 与 Task 04 `LayerKVCache.read` 完全一致，Task 08 可直接替换。
- Task 08 ModelRunner 的接入点：把 `RequestState.cache` 从 `ContiguousKVCache` 换成
  `BlockTable`（或包一层 `PagedKVCache`），并在 `QwenSelfAttention.forward` 中用
  `gather` 取到当前长度的 K/V 代替 `kv_cache.read`；prefill 一次写多 token、decode 每步写 1 token，
  都走 `BlockTable.append`。
- 物理布局 `[num_blocks, num_layers, block_size, KVH, D]` 已对齐 docs/02 §7，Task 08 无需改布局。
- **禁止在 Task 07 内改 `attention.py` / `engine/core.py`**（用户明确不越界 Task 08）。
- **Task 06 遗留的 set 顺序抖动**：`Scheduler._running` 原为 `set`，导致 running decode 请求输出顺序随 hash 随机；在你机器上触发 `test_running_decode_always_scheduled_under_budget` 失败（期望 `["a","b"]`，实际 `["b","a"]`）。已修复为 `dict[str, None]`（保留 insertion order + O(1) 成员判定），全量快测重新稳定通过。

---

## Task 08：ModelRunner 接入 Paged KV（2026-09-08）

**新增文件**

```text
liteinfer/model/runner.py        # KVCacheView 协议 / PagedLayerCache 适配器 / ModelRunner / infer_kv_dims
tests/test_paged_runner.py       # 13 条快测 + 4 条 model 标记
examples/paged_runner_demo.py    # 多请求 + 块生命周期 + 与连续 KV 逐字对照
docs/design/paged_runner.md      # 设计文档（9 项交付 + Alternative + Known Limitations）
```

**修改文件**

```text
liteinfer/cache/paged.py                    # +BlockTable.write_token / gather_v，内部抽出 _gather
liteinfer/config.py                         # +block_size(16) / +num_blocks(None=按调度预算推导)
liteinfer/engine/core.py                    # forward 改走 runner；块表 prefill 时惰性分配；终态/取消 _release 归还块
liteinfer/engine/request.py                 # RequestState.cache -> BlockTable|None；+cache_bytes
liteinfer/model/minimal/attention.py        # 仅类型标注：kv_cache 放宽为 KVCacheView（计算逻辑零改动）
liteinfer/model/minimal/layer.py            # 同上
liteinfer/model/minimal/model.py            # 同上（kv_caches: list[KVCacheView] | None）
liteinfer/__init__.py                       # 惰性导出 ModelRunner / PagedLayerCache
```

**关键接口签名**

```python
KVCacheView(Protocol)                    # append(k[1,n,KVH,D], v, start) -> (k,v)[1,start+n,KVH,D]
                                         # read(length) -> (k, v)[1,length,KVH,D]
PagedLayerCache(table: BlockTable, layer_idx: int)   # 伪装成 LayerKVCache，鸭子类型替换
BlockTable.write_token(layer, offset, k_vec[KVH,D], v_vec)   # 单 token 单层写，按需分块
BlockTable.gather(layer, length) / .gather_v(...)    # -> [1, length, KVH, D]
infer_kv_dims(model) -> (num_layers, num_kv_heads, head_dim)
ModelRunner(model, cfg, block_size=None, num_blocks=None)
    .paged: PagedKVCache            # 共享块池，构造时一次分配（inference_mode 之外）
    .new_block_table() -> BlockTable
    .prefill(input_ids, table) -> logits[1,P,V]
    .decode(token_id, position, table) -> logits[1,1,V]   # position 即写入偏移（绝对位置）
    .free_table(table) / .bytes_for(table) -> int
    .block_size / .num_layers
# EngineCore 变化：
EngineCore.runner: ModelRunner
EngineCore.new_block_table()      # 取代原 new_cache（不再预分配连续缓存）
EngineCore._release(st)           # 先记账 cache_bytes 再 free_table，st.cache=None
```

**验收命令**（CPU 全绿）

```bash
set "HF_HOME=D:\LiteInfer\hf_cache"
set PYTHONPATH=d:\LiteInfer
python -m pytest tests/test_paged_runner.py -q -m "not model"   # 13 passed（~20s）
python -m pytest tests/test_no_hardcoded_cuda.py -q             # 1 passed
python -m pytest -q                                             # 133 passed（Task 01-07 不回归，34 deselected）
python -m pytest -m model -q                                    # 34 passed（~313s）
python examples/paged_runner_demo.py --max-tokens 8             # identical=True，used_blocks=0
```

实测（Qwen2.5-0.5B, CPU FP32）：

```text
block_size=16 num_blocks=18 pool=7.1 MB
提交 3 个请求 -> waiting=3 used_blocks=0
step 1 之后: running=2 used_blocks=2
全部结束:   used_blocks=0 free_blocks=18
paged 与 contiguous 三请求文本 100% 逐字一致 -> OK
```

**踩过的坑**

- **`BlockTable.append` 与逐层前向的粒度冲突（本 Task 最大的坑）**：Task 07 的
  `append` 要求一次给出所有层 `[num_layers, n, KVH, D]`，但注意力是**逐层**前向的，
  第 i 层凑不齐 `num_layers` 这一维。解决办法是新增 `write_token(layer, offset, k, v)`
  （单 token 单层），并把**块分配判据设为 token 绝对位置** `offset // block_size`——
  于是同一 token 被 24 层各写一次时只有第一层真正 `alloc`，分配次数与层数无关
  （`test_block_allocated_once_per_token_not_per_layer` 钉住：9 token/bs=4 → 3 块）。
- **`gather` 只读 K**：Task 07 的 `gather` 硬编码读 `k_blocks`。V 需要独立方法，
  于是抽出私有 `_gather(blocks, layer, length)`，让 `gather` / `gather_v` 共用，
  既不动 Task 07 已压测的签名，又避免读 V 时误读成 K。
- **释放之后就查不到字节数**：`_to_output` 原用 `st.cache.nbytes`。块 `free` 后
  `num_tokens` 归零，字节数再也问不出来。改为 `_release` 里**先记账再释放**，
  存进新增的 `RequestState.cache_bytes`（Task 10 的 KV utilization 会用）。
- **dataclass 字段顺序**：想给 `RequestState.cache` 加默认值 None，但它前面的
  `prompt_ids` 没有默认值 → "non-default argument follows default argument"。
  不去重排字段（会破坏既有关键字调用），改为在 `submit` 里显式传 `cache=None`。
- **类型标注的循环导入风险**：`attention.py` 若要运行时 `from liteinfer.model.runner
  import KVCacheView`，会经过 `liteinfer.model` 包初始化，有环。用
  `if TYPE_CHECKING:` + `from __future__ import annotations` 惰性求值绕开。
- **假模型测不出分页**：Task 05/06 的 `FakeLM.forward` 直接忽略 `kv_caches`，
  所以块分配/归零全是空跑（断言"释放后 used==0"在分配都没发生时也会假绿）。
  新写了 `_WritingFakeLM`，forward 里显式 `for c in kv_caches: c.append(k, v, write_pos)`，
  才真正覆盖写路径与块生命周期；并加
  `test_blocks_actually_used_during_generation` 断言"过程中 used>0"，防止空跑通过。
- **Windows cmd 没有 `tail`**：`pytest ... | tail -40` 直接报"不是内部或外部命令"，
  改用不带管道的命令。
- **别宣称性能收益**：gather 读是真实拷贝，比 Task 04 contiguous 的零拷贝视图更慢
  （每 step 每层 K/V 各多一次 gather）。本阶段只验收正确性与块回收，
  吞吐数字全部留给 Task 12 的 GPU benchmark。

**下一阶段提示（Task 09）**

- 引擎的 `step()` 已经返回 `list[RequestStepResult]`（含 `token_id` / `token_text` /
  `finished`），正是 Task 09 流式输出的天然数据源，无需改引擎。
- `EngineCore.run()` 是同步阻塞的；Task 09 的 AsyncEngine 需要把它拆成
  "可被 asyncio 调度的 step"，并支持 client disconnect → `cancel(request_id)`
  （本 Task 已保证 cancel 会立刻归还物理块，docs/02 §10 的防泄漏要求已满足）。
- 块池容量目前静态（`num_blocks` 构造时定死）；Task 09 做并发准入时若想按
  "剩余可用块数"做调度闸门，`BlockPool.num_free` 已经是可直接消费的观测项。

---

## Task 09：Async Engine + Streaming API（2026-09-08）

**新增文件**

```text
liteinfer/engine/async_engine.py    # StreamChunk / AsyncStream / AsyncEngine（命令队列 + 后台循环 + 每请求输出队列）
liteinfer/server/__init__.py        # 导出 create_app（刻意不进 liteinfer/__init__）
liteinfer/server/schemas.py         # OpenAI 兼容 pydantic v2 模型
liteinfer/server/sse.py             # SSE 帧编码/解析 + [DONE] + 禁用缓冲响应头
liteinfer/server/app.py             # create_app：/v1/completions /v1/chat/completions /v1/models /health /cancel
liteinfer/server/main.py            # uvicorn 启动入口（python -m liteinfer.server.main）
tests/_fakes.py                     # 测试共用的"会写 KV"的假模型/假 tokenizer（跨文件共享）
tests/test_async_engine.py          # 12 条快测 + 1 条 model
tests/test_server_api.py            # 16 条快测（含真实 uvicorn 断连回收）
examples/async_engine_demo.py       # asyncio 并发流式 + 取消 + 与 CachedGenerator 逐字对照
examples/server_demo.py             # 起真服务 + OpenAI Python Client 调非流式/流式/chat/断连
docs/design/async_serving.md        # 设计文档（9 项交付 + Alternative + Known Limitations）
```

**修改文件**

```text
liteinfer/engine/__init__.py        # 导出 AsyncEngine / StreamChunk
liteinfer/__init__.py               # _ASYNC_NAMES 惰性导出 AsyncEngine / StreamChunk
pyproject.toml                      # dev/serving 补 FastAPI 栈；asyncio_mode = "auto"
PROGRESS.md                         # 本记录 + 进度表 09 置 ✅
```

**关键接口签名**

```python
StreamChunk(request_id, token_id, text, finished, finish_reason, index=0)   # text 是增量
AsyncStream(engine, request_id)     # __aiter__ / aclose；关闭即触发 abort
AsyncEngine(core)
    .submit(prompt, params=None) -> str            # 投命令 + await future 拿 rid
    .stream(request_id) -> AsyncStream
    .generate(prompt, params=None) -> AsyncStream  # = submit + stream（返回流，不是 generator）
    .abort(request_id)          # await 到取消生效（HTTP 取消端点）
    .abort_nowait(request_id)   # 只投递命令（生成器清理路径）
    .start() / .shutdown() / .is_running() / .pending_streams / .core
create_app(engine=None, cfg=None, model_id=None) -> FastAPI   # 注入 engine 即可不加载模型
# HTTP：
#   GET  /health                        {status, model, device, dtype, waiting, running, kv_blocks_*}
#   GET  /v1/models                     OpenAI ModelList
#   POST /v1/completions                prompt: str | list[str]，stream -> SSE
#   POST /v1/chat/completions           messages -> chat template（无模板则降级角色拼接）
#   POST /v1/requests/{request_id}/cancel
```

**验收命令**（CPU 全绿）

```bash
set "HF_HOME=D:\LiteInfer\hf_cache"
set PYTHONPATH=d:\LiteInfer
python -m pytest tests/test_async_engine.py -q   # 12 passed
python -m pytest tests/test_server_api.py -q     # 16 passed
python -m pytest tests/test_no_hardcoded_cuda.py -q   # 1 passed
python -m pytest -q                             # 161 passed, 35 deselected
python -m pytest -m model -q tests/test_async_engine.py   # 1 passed（~50s）
python examples/async_engine_demo.py --max-tokens 8
python examples/server_demo.py --max-tokens 8
```

实测（Qwen2.5-0.5B, CPU FP32）：

```text
pytest -q:            161 passed, 35 deselected（Task 01-08 不回归，新增 28 条）
async_engine_demo:    3 并发流式 identical=True；取消后 status=cancelled used_blocks=0
server_demo:          OpenAI Client 非流式/流式/chat 全通；断连后 kv_blocks_used=0
```

**踩过的坑**

- **流式丢最后一个 token（最隐蔽的一个）**：`EngineCore` 达到 `max_tokens` 时把"最后一个 token"
  和 `finished=True` 合并在同一个 `RequestStepResult` 里；最初对 finished chunk 一律 `text=""`，
  结果实测 `'345'` 而不是 `'3456'`。改为按 OpenAI 协议拆两帧：token 帧 + 只带 finish_reason 的
  空帧；EOS 那步本身没有 token（`token_id is None`），单独走"只发终态帧"分支。
- **httpx 的 `ASGITransport` 形不成"断连"**：实测第一条 SSE 帧到达时请求已经 `finished`
  （它把应用跑到结束才返回），所以"读到一半断开"在 ASGI 直连下不可能发生。硬验收必须起
  **真实 uvicorn + 真 socket**（`tests/test_server_api.py::test_client_disconnect_frees_blocks`）。
- **TestClient 会把应用跑在另一个线程的事件循环里**，而 AsyncEngine 的队列/future 必须与引擎
  循环同 loop；跨 loop 结算 future 会挂死。改用 `httpx.AsyncClient + ASGITransport` 手工跑
  `app.router.lifespan_context(app)`，全在同一 loop 内。
- **嵌套 async generator 的清理时机不确定**：外层（SSE）被关闭时，若内层流只靠 GC 兜底，
  其 `finally` 由 `loop.call_soon_threadsafe` 调度，验证不了"断连一定回收"。改为显式
  `it = engine.stream(rid).__aiter__()` + `finally: await it.aclose()`。
- **清理路径不能 `await`**：`AsyncStream._iterate` 的 finally 若 await 一个"由引擎循环 resolve
  的 future"，在生成器 finalizer/事件循环关闭时可能永远等不到。拆成 `abort()`（等生效）与
  `abort_nowait()`（只投递），清理路径用后者。
- **假 chat 测试的 tokenizer 陷阱**：FakeTokenizer 只能编码数字字符，若走 chat 降级拼接
  `"user: 2\nassistant:"` 会 `int('u')` 崩溃。给它加 `apply_chat_template`（取最后一条 content），
  降级分支另用 `object()` 当 tokenizer 单独覆盖。
- **Windows GBK 控制台**：示例里的"对照"打印成乱码，改用 ASCII 标签 `[parity]`。
- **`asyncio.Queue` 可在无 running loop 时构造**（3.10+ 已移除构造期的 loop 绑定），
  因此 `AsyncEngine` 可以在同步 fixture 里创建，队列在第一次 `get()` 时绑定到使用它的 loop。

**发现的环境问题（非 Task 09 引入，未擅自修改）**

`pytest -m model -q`（35 条一次性跑）会在第 1~16 个测试后 `Windows fatal exception: access
violation`，崩溃点 `transformers/core_model_loading.py::_materialize_copy`。**逐文件跑 8 个
文件全部通过**（1+4+7+7+8+2+2+4 = 35 passed）。原因：每个测试模块都有 module 级 `loaded`
fixture，FP32 的 0.5B（HF 模型 + MinimalQwen 副本）各约 2GB，本机 16.9GB 仅剩约 5GB 可用时
连续多次加载会撑爆。建议：逐文件跑，或把 model fixture 改成 session 级只加载一次
（涉及改动 Task 01-08 的既有测试文件，本次未动）。

**下一阶段提示（Task 10）**

- `AsyncEngine` 已把每一步的结果原样包成 `StreamChunk` 投递出来，`RequestStepResult` 里
  `token_id/token_text/finished/finish_reason` 齐全，Task 10 的 TTFT/TPOT/ITL 只需在
  `_dispatch` 前后打点（prefill 首 token 时间、相邻 token 间隔）。
- `RequestState` 已有 `prefill_latency_s` / `decode_latency_s` / `cached_len` / `cache_bytes`，
  `RequestOutput` 已透出后三者，KV utilization 的分母可用 `engine.core.runner.paged.num_blocks_*`。
- `/health` 已暴露 `waiting` / `running` / `kv_blocks_used` / `kv_blocks_total`，
  Prometheus 之类的采集口可以直接接这里。
- 流式响应目前**不带 usage**（需要 `stream_options.include_usage`），Task 10 补指标时一起做。

---

## Task 10：Metrics + Tracing（2026-09-09）

**新增文件**

```text
liteinfer/observability/__init__.py   # 导出 RequestMetrics/MetricsRegistry/RequestTrace/build_trace 等
liteinfer/observability/metrics.py    # 纯函数指标还原（TTFT/ITL/TPOT/分位数）+ RequestMetrics + MetricsRegistry
liteinfer/observability/trace.py      # RequestTrace：由打点字段还原事件时间线（render ASCII / to_dict JSON）
tests/test_metrics.py                 # 21 条快测（合成时间戳纯函数 + FakeLM 引擎集成）+ 1 条 model
tests/test_server_metrics.py          # 11 条快测：/metrics、trace 端点、SSE include_usage
examples/metrics_demo.py              # 单请求完整时间线 demo（验收主体）
docs/design/metrics_tracing.md        # 设计文档（9 项交付 + Alternative + Known Limitations）
```

**修改文件**

```text
liteinfer/engine/request.py   # RequestState +prefill_start_s/+prefill_end_s/+token_times（打点字段）
liteinfer/engine/core.py      # _prefill/_after_emit 打点；_release 汇入 MetricsRegistry；+trace_of/+metrics_snapshot
liteinfer/server/schemas.py   # +StreamOptions(include_usage)；CompletionResponse/ChatCompletionChunk 的 usage 字段说明
liteinfer/server/app.py       # +GET /metrics、+GET /v1/requests/{id}/trace；/health +kv_utilization；SSE usage 帧
liteinfer/__init__.py         # 惰性导出 RequestMetrics / MetricsRegistry / RequestTrace
PROGRESS.md                   # 本记录 + 进度表 10 置 ✅
```

**关键接口签名**

```python
RequestState 新字段: prefill_start_s / prefill_end_s (float, 0=未发生) ; token_times: list[float]  # 与 generated 下标对应
ttft_of(wall_start, token_times) -> float      # 首 token - enqueue（含排队+prefill）
inter_token_latencies(token_times) -> list[float]
tpot_of(token_times) -> Optional[float]        # 单 token 请求为 None（不是 0）
RequestMetrics.from_state(st) -> RequestMetrics  # 终态一次性汇总（frozen，含 itl p50/p95/max、e2e、tokens_per_s、cache_bytes、token_times）
MetricsRegistry(throughput_window_s=10.0)
    .record(m)                                  # 按 request_id 覆盖（幂等）
    .get(rid) / .has(rid) / __len__
    .snapshot(num_waiting=, num_running=, kv_blocks_used=, kv_blocks_total=, now=None) -> dict
        # 含 kv_utilization、output_tokens_per_s(滚动窗口)、ttft/tpot 均值、
        # gpu_memory_mb(None on CPU) + gpu_memory_mb_display("N/A (no GPU)")
RequestTrace(events, ttft_s, tpot_s, e2e_s, ...).render() / .to_dict()
build_trace(st) -> RequestTrace                # enqueue -> prefill_start -> prefill_end -> token×n -> finished/cancelled
# EngineCore 变化：
EngineCore.metrics: MetricsRegistry
EngineCore.trace_of(request_id) -> RequestTrace      # 未知 id 抛 KeyError
EngineCore.metrics_snapshot() -> dict                # 调度器/块池观测项 + 注册表聚合
# Server 变化：
GET /metrics                          # JSON 快照（非 Prometheus 文本格式，依赖零新增）
GET /v1/requests/{request_id}/trace   # 404 if unknown；非终态也可查
/health                               # +kv_utilization
SSE: stream_options={"include_usage": true} 时流末尾追加 choices=[] + usage 一帧（Task 09 遗留补齐）
```

**验收命令**（CPU 全绿）

```bash
set "HF_HOME=D:\LiteInfer\hf_cache"
set "PYTHONPATH=d:\LiteInfer"
python -m pytest tests/test_metrics.py -q -m "not model"    # 21 passed
python -m pytest tests/test_server_metrics.py -q            # 11 passed
python -m pytest tests/test_no_hardcoded_cuda.py -q         # 1 passed
python -m pytest -q                                         # 193 passed, 36 deselected（Task 01-09 不回归）
python -m pytest -m model -q tests/test_metrics.py          # 1 passed（~39s）
python examples/metrics_demo.py --max-tokens 16
```

实测（Qwen2.5-0.5B, CPU FP32, demo 输出节选）：

```text
TTFT=510.23ms  TPOT=358.48ms  ITL p50/p95/max=333.42/473.85/502.72ms  E2E=5.8876s
tokens_per_s=2.72   prefill/decode=0.5047/5.3377s   kv bytes=516096
trace: enqueue(+0.000) -> prefill_start(+0.000) -> prefill_end(+0.505) -> token×16 -> finished(+5.888)
自洽验算: 0.510 + 0.358×15 = 5.88 ≈ E2E 5.888（尾巴=终态处理 ~0.1ms）
gpu_memory_mb = N/A (no GPU)；结束后 kv_blocks_used=0（utilization=0.0）
```

**踩过的坑**

- **MetricsRegistry 用 dict 而非 list 是必须的**：合成测试里两个 RequestState
  共用了 request_id="r1"，record 静默覆盖导致 requests_total 断言 1 != 2；
  dict 键覆盖语义顺带让 `_release` 的潜在重复触达天然幂等。
- **SamplingParams 有构造期校验**（max_tokens>=1）：合成"0 个 token"的
  waiting 请求时 `SamplingParams(max_tokens=0)` 直接抛 ValueError，测试要
  用 `max(1, output_tokens)`，输出 0 个 token 靠 generated/token_times 空
  列表表达，不靠 params。
- **调度器 running 集合是惰性清理**（Task 06 既有行为）：请求终态后、
  下一次 `schedule()` 之前 `num_running` 仍计入它——run() 刚结束时断言
  `num_running==0` 会失败。非本 Task 范围，未擅自修改调度器；测试只断言
  `num_waiting==0` 并注释原因（Known Limitations #1）。
- **打点字段用 0.0 做"未发生"哨兵**：perf_counter 取值恒为正，0.0 可安全
  作为哨兵；trace 构建时按 `> 0` 过滤未发生的事件（waiting 请求只有
  enqueue 一条）。
- **滚动窗口吞吐必须基于 token 级时间戳**：仅请求级 tokens_per_s 聚合值
  算不出"最近 N 秒输出多少 token"；RequestMetrics 全量保留 token_times
  （frozen tuple），snapshot 里逐 token 判窗口归属。分母恒为窗宽而非窗口
  时长，保证与稳态值可比。`snapshot(now=...)` 参数化时钟，窗口测试才能
  确定性（"旧 token 排除在窗外"不用 sleep 等真实时间流逝）。
- **单 token 请求的 TPOT 必须是 None 不是 0**：与 OpenAI usage 语义对齐；
  均值聚合（ttft_s_mean/tpot_s_mean）也要跳过无定义的请求，否则 0 会稀释
  均值——"N/A 优于 0"的原则同样适用于延迟指标。
- **取消路径也要记账**：waiting 中即被取消的请求（cache=None，`_release`
  原本 early return）原本会漏出指标注册表；在 early-return 分支里也
  record，失败路径的请求数才完整。

**下一阶段提示（Task 11 Prefix Cache）**

- 打点已就位：prefix cache 的命中效果可直接消费——`RequestState.cached_len`
  + `RequestMetrics`/`RequestOutput.cached_tokens`（语义：KV 里已有的 token
  数）；prefix cache 落地后 TTFT 的下降就是最直接的验收观测。
- `BlockTable` 已有物理块句柄，hash 表（block hash -> block）可挂在
  `PagedKVCache` 层；释放块时"不归还而是挂入 hash 表"的引用计数策略
  （docs/02 §11）注意与 `_release` 的"先记账再释放"顺序共存。
- `/metrics` 的 `kv_utilization` 分母是总物理块；prefix cache 会"故意占住"
  块，届时 utilization 语义需区分"占用（含缓存）"与"在飞请求持有"。

---

## Task 11：Prefix Cache（2026-09-09）

**新增文件**

```text
liteinfer/cache/prefix.py          # block_hash（链式 blake2b）+ PrefixCache（哈希表/ref count/LRU 驱逐）
tests/test_prefix_cache.py         # 24 快测 + 1 model（哈希/命中/ref/LRU/位级 gather/引擎集成/压测/真模型）
examples/prefix_cache_demo.py      # 共享前缀 workload，开/关缓存对照
docs/design/prefix_cache.md        # 设计文档（9 项交付 + Alternative + Known Limitations）
```

**修改文件**

```text
liteinfer/cache/paged.py              # BlockTable +prefix_cache 字段 / adopt() / 分配与释放改道（gather/write 零改动）
liteinfer/cache/__init__.py           # 导出 PrefixCache / block_hash
liteinfer/model/runner.py             # ModelRunner 持有 PrefixCache；prefill(+write_pos=0 向后兼容)
liteinfer/config.py                   # EngineConfig +enable_prefix_cache(默认 False) + from_env(LITEINFER_PREFIX_CACHE)
liteinfer/engine/request.py           # RequestState/RequestOutput +prefix_hit_tokens；RequestState +prompt_token_ids
liteinfer/engine/core.py              # _prefill lookup/adopt/后缀 forward/register；_decode 块写满注册；_to_output/metrics_snapshot 透出
liteinfer/observability/metrics.py    # RequestMetrics +prefix_hit_tokens；snapshot +prefix_cached_blocks/prefix_hit_tokens_total
liteinfer/__init__.py                 # 惰性导出 PrefixCache
```

**关键接口签名**

```python
block_hash(parent: int, tokens: Sequence[int]) -> int   # h_i = blake2b(h_{i-1} + 块内 token ids)，无符号 64 位链
PrefixCache(pool: BlockPool, block_size: int)
    .lookup(token_ids) -> (list[block_id], hit_len)     # 收养：ref+1 并移出 LRU；prompt 恰好整块全中时放弃最后一块
    .register(token_ids, table) -> int                  # 新注册块数；已存在哈希跳过（先到先得）
    .release_block(block_id)                            # ref-1；归 0 进 LRU 不还池；未追踪块直接还池
    .alloc_new() -> block_id                            # 池空 -> LRU 驱逐；无可驱逐 fail fast
    .stats() -> dict                                    # lookups/hits/hit_tokens/evictions/cached/evictable
BlockTable.adopt(block_ids, num_tokens)                 # 命中块插到 blocks 头部（仅限空表）
BlockTable(..., prefix_cache=PrefixCache|None)          # 分配/释放改道由表内部消费，调用方无感
ModelRunner.prefill(input_ids, table, write_pos=0)      # 命中时 write_pos=hit_len 只算后缀
ModelRunner.prefix: PrefixCache | None                  # cfg.enable_prefix_cache 决定
EngineConfig(enable_prefix_cache=False)                 # 默认关：不破坏"请求结束后块全回收"契约；ablation 现成对照组
RequestOutput/RequestMetrics.prefix_hit_tokens: int     # 可观察 cache hit 的直接载体
# /metrics 新键：prefix_cached_blocks（None=未开启）、prefix_hit_tokens_total
```

**验收命令**（CPU 全绿）

```bash
set "HF_HOME=D:\LiteInfer\hf_cache"
set "PYTHONPATH=d:\LiteInfer"
python -m pytest tests/test_prefix_cache.py -q -m "not model"   # 24 passed
python -m pytest tests/test_no_hardcoded_cuda.py -q             # 1 passed
python -m pytest -q                                             # 217 passed, 37 deselected（Task 01-10 不回归）
python -m pytest -m model -q tests/test_prefix_cache.py         # 1 passed（真模型命中+逐字一致）
python -m pytest -m model -q tests/test_paged_runner.py         # 4 passed（runner 改动回归）
python examples/prefix_cache_demo.py --max-tokens 16
```

实测（Qwen2.5-0.5B, CPU FP32, block_size=16）：

```text
===== prefix OFF =====            ===== prefix ON  =====
hit=  0 | prefill= 1593.7 ms      hit=  0 | prefill= 1600.3 ms   # 首个请求必全算
hit=  0 | prefill= 1526.2 ms      hit= 32 | prefill=  829.5 ms   # 命中 2 块，~1.9x
hit=  0 | prefill= 1675.5 ms      hit= 32 | prefill=  877.5 ms
两组输出文本逐字一致；stats: hits=2/3, hit_tokens=64, evictions=0
```

**踩过的坑**

- **blake2b 摘要是无符号 64 位**：`int.from_bytes(digest)` 的结果最大 2^64-1，
  链式哈希第二层把父哈希 `struct.pack(">q", ...)`（有符号）直接溢出抛
  struct.error——快测 10 条连带挂掉。改 `">Q"`。教训：自研哈希链先把
  "任意层级"的边界值测一遍，别只测第一层。
- **prompt 恰好 N 个完整块且全命中时必须放弃最后一块**（留 token 现算，
  logits 只能由 forward 产出）。单块 prompt 因此命中恒为 0，首版两个测试
  用 1 块 prompt 造"释放后再命中"的场景，直接踩中这条设计规则——测试
  场景要用 ≥2 块的 prompt。
- **注册时机必须"KV 已完整落块"**：prefill 后 candidate 的 KV 还没写
  （下一步 decode 才写它）。若在 _after_emit 里按 prompt+generated 注册，
  同批后 prefill 的请求会收养到"最后槽位是零"的半成品块。decode 的注册点
  放在 forward 之后、`cached_len % block_size == 0` 时，用
  `prompt_token_ids + generated`（此刻两者拼接恰好等于已写序列）。
- **FakeLM 不读 KV，引擎级"输出一致"测不出收养块的内容错误**（FakeLM 的
  logits 是输入 token 的纯函数，KV 内容是全零）。补了一条 cache 层位级
  测试：收养 + 后缀续写后 gather 与"从零写一遍"逐位相等；真模型端到端
  逐字一致做最终兜底。
- **测试无法跨引擎命中**：PrefixCache 挂在 ModelRunner/PagedKVCache 实例上，
  新建第二个 EngineCore 是全新缓存。所有"第二请求命中"的断言必须发生在
  **同一个引擎**内（串行 run 两次，或同批 submit）。
- **RequestOutput 没有 generated 字段**（那是 Request/内部态的）：观测断言用
  `output_tokens` / `text`，或走 `core.get_request(rid).generated`。

**下一阶段提示（Task 12）**

- `enable_prefix_cache` 开关即 ablation 的 "paged vs paged+prefix" 对照组；
  `PrefixCache.stats()` 与 `/metrics` 的 `prefix_hit_tokens_total` 可直接写 CSV。
- 共享前缀 workload（docs/07 Task 12 的 4 个 workload 之一）现成可用，
  demo 里的"串行同前缀 3 请求"形状可直接搬进 benchmark 脚本。

---

## Task 12：Benchmark + Ablation（2026-09-09）

**新增文件**

```text
benchmark/__init__.py                # 声明为包，测试可 from benchmark.xxx import ...
benchmark/liteinfer_benchmark.py     # 矩阵驱动器（measure-only）：workload 规格/提示词构造/5 驱动 cell/汇总/CSV+JSON
benchmark/benchmark_charts.py        # -> charts/*.png（Agg；每图脚注附实验记录 docs/05 §8）
benchmark/benchmark_report.py        # -> report.md（诚实声明 + 实验记录表 + 结果表 + 真实数据驱动的结论节）
benchmark/run_full_matrix.sh         # 云端一键脚本：全 5 引擎 × 4 workload × 6 并发，一轮跑完并 commit
tests/test_benchmark_matrix.py       # 19 条无模型快测
examples/benchmark_demo.py           # CPU smoke 全链路示例（验收主体）
docs/design/benchmark.md             # 设计文档（9 项交付 + Alternative + Known Limitations）
```

**修改文件**

```text
pyproject.toml                       # [dev] +matplotlib>=3.9（Task 12 图表硬依赖，本机已装）
PROGRESS.md                          # 本记录 + 进度表 12 置 ✅
```

**关键接口签名**

```python
# benchmark/liteinfer_benchmark.py
ENGINES = ("hf", "nokv", "kv", "batch", "prefix")          # 5 真实驱动覆盖 6 命名消融
FULL_WORKLOADS / SMOKE_WORKLOADS                            # 4 个 workload × full/smoke 口径
workload_spec(name, smoke) -> dict   # smoke 缩放保形（kind 不变、长度缩小）
build_prompt_text(tokenizer, n_tokens, macro=_PROMPT_MACRO) -> str   # encode-重复-截断-decode
build_sp_prompts(tokenizer, shared_len, tail_len, count, tail_suffix=..., macro=...) -> list[str]
percentile(values, q) / mean_opt(values)                   # None 感知（N/A 优于 0）
CellRow(engine, workload, concurrency, ...36 字段)         # .to_csv()/from_csv()；缺失列空串<->None
run_cell(...) / run_matrix(args) -> (rows, env)            # env=实验记录（docs/05 §14）
_engine_cfg(base, concurrency, prompt_len, max_tokens, enable_prefix_cache) -> EngineConfig
main(argv) -> int  # 全矩阵：--workloads decode prefill typical sp --concurrency 1 2 4 8 16 32 \
                  #          --engines hf nokv kv batch prefix [--device --dtype --smoke ...]
render_charts(csv, env, outdir) -> list[Path]   # 指标×5 + sp_prefix_ttft 对比图
build_report(csv, env, charts_dir, out) -> str  # markdown 报告
```

**验收命令**（CPU 全部跑通）

```bash
set "HF_HOME=D:/LiteInfer/hf_cache"
set PYTHONPATH=d:/LiteInfer
python -m pytest tests/test_benchmark_matrix.py -q      # 19 passed
python -m pytest tests/test_no_hardcoded_cuda.py -q      # 1 passed（benchmark/ 无裸 cuda）
python -m pytest -q                                     # 236 passed, 37 deselected
python examples/benchmark_demo.py --smoke               # 测量+图表+报告全链路，产物落 benchmark/results/demo/
```

实测（Qwen2.5-0.5B, CPU FP32, smoke 36 cell 全数通过）：
kv 稳态 ~2.6~3.2 tok/s；sp@conc=4 prefix TTFT p50 4642.9ms vs batch 11812.5ms（下降 60.7%）；
prefix 命中 token 随并发增长（typical conc2/4 = 24/36）。全部 cell parity_ok=1（minimal 系
逐字一致），batch 终态 used=0、prefix free+used==total（块账目平衡），显存列 N/A (no GPU)。

**踩过的坑**

- **消融映射（设计取舍，已获用户确认）**：continuous batching 与 paged KV 自
  Task 06/08 合并进 EngineCore 一个驱动，仓库无两套独立历史实现可分别跑；采用
  "一线驱动双测量轴"（吞吐=V2 效果、KV 利用率=V3 效果），报告显式声明而非重写旧逻辑。
- **2048 长 prompt 的两个显式配置**：默认块池按 max_new_tokens+128 token 推导，
  长 prompt 撞池抛错 → `_engine_cfg` 按 ceil((pl+ml)/bs)+4 每序列 × 并发显式传
  num_blocks；默认 token budget 2048 会让 2048 prompt 的 submit ValueError →
  budget=max(2048, pl+8)。这两处不显式配置会直接炸，是本 Task 集成的第一道坎。
- **prefix cell 的缓存块统计要在 `del core` 之前读**：PrefixCache 挂在
  ModelRunner 上，删了就读不到；prefix 终态 used≠0 是预期（缓存块占池），
  泄漏契约改为验 `free+used==total`，只有 batch 才真正要求 used==0。
- **报告模板把 `text += _insights(rows)` 误删过**：该语句被模板改写吞掉后结论节
  静默消失，`test_report_insights_only_from_data` 当场逮到（缺少"数据不足"分支）。
  教训：改完大段模板必须重跑对应单测，不能只看 diff。
- **顺序驱动"并发"=排队逐个跑**：hf/nokv/kv 并发 N 是 N 个请求背靠背，吞吐天然
  平坦；引擎驱动同时提交才能看出批处理收益——两口径都用 out_tokens/wall，
  cell 间可比。HF 无拆段计时，其 TTFT/TPOT/ITL 为 N/A（"N/A 优于 0"延续）。
- **PShell 无 `&&`**：`set PYTHONPATH=... && python ...` 直接 ParseError，须
  `$env:PYTHONPATH='d:/LiteInfer'; python ...`；后台任务的 stdout 是块缓冲，
  盯进度用 `Get-Process python` 的 CPU 而非日志。

**下一阶段提示（Task 13）**

- 全矩阵命令 = `benchmark/run_full_matrix.sh`（云端设 LITEINFER_DEVICE/LITEINFER_DTYPE），
  产物先 commit 再 push（云端随时被回收）；本地跑完 demo 后可删 `benchmark/results/demo`。
- `results.csv` 列序是跨脚本契约（charts/report 消费），加列需同步三处，单测
  `test_csv_fieldnames_stable` 会兜底。
- Task 13 Docker/CI 可直接把 `pytest` 与 `benchmark_demo --smoke` 当冒烟验收接进去。

---

## Task 13：Docker + CI + README（2026-09-09）

**新增文件**

```text
Dockerfile                          # CPU 优先镜像（ARG TORCH_INDEX_URL 可切 GPU wheel）
.dockerignore                       # 排除 .venv/hf_cache/pip_cache/docs/.git 等
.github/workflows/ci.yml            # 质量门：pytest 快测 + A1 门禁；docker-build 手动触发
docs/known_limitations.md           # 全部引用既有 docs/PROGRESS 的真实局限（不新编造）
tests/test_deployment_assets.py     # 6 条部署资产约束测试 + 镜像上下文 skipif 守卫
```

**修改文件**

```text
README.md                  # 重写：定位/诚实声明/架构图/快速开始(venv+Docker)/Benchmark/局限/目录/文档索引
PROGRESS.md                # 本记录 + 进度表 13 置 ✅ + 环境备忘改 Python 3.12.10（实测）
```

**关键资产要点**

```dockerfile
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu   # GPU 构建: --build-arg 指 cu 版 index
ENV HF_HOME=/model-cache                                    # 容器内缓存路径，运行时 -v 挂载本机 hf_cache
RUN pip install torch --index-url "${TORCH_INDEX_URL}"      # 必须先单独装，editable 才不会重新拉 GPU wheel
CMD ["python", "-m", "liteinfer.server.main", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# .github/workflows/ci.yml
# push/PR → ubuntu-latest: 装 CPU torch → pip install -e ".[dev]" → pytest -q -m "not model" → test_no_hardcoded_cuda
# docker-build job 仅 workflow_dispatch（镜像构建慢，本机 Docker 已可独立验证）
```

- `tests/test_deployment_assets.py`：Dockerfile 含 HF_HOME 且无设备字面量、.dockerignore 排除重路径、
  CI 含 pytest+A1 门禁、README 含 docs/07 Task 13 必含章节、known_limitations.md 存在且含 PagedAttention 诚实声明。
  镜像内在 module 级 skipif（部署资产按设计不进镜像，见踩坑 #4）。

**验收命令**（全部真实执行过）

```bash
set "HF_HOME=D:\LiteInfer\hf_cache"; set PYTHONPATH=d:\LiteInfer （PowerShell 用 $env: 赋值）
pytest -q                                   # 242 passed, 37 deselected（基线 236 + 新增 6）
pytest tests/test_deployment_assets.py -q   # 6 passed
docker build -t liteinfer:cpu .             # 构建成功（torch 2.14.0+cpu / transformers 5.16.1 / sentencepiece 0.2.2）
docker run --rm liteinfer:cpu python -m pytest -q     # 236 passed, 6 skipped, 37 deselected
docker run --rm -v D:/LiteInfer/hf_cache:/model-cache liteinfer:cpu python examples/async_engine_demo.py --max-tokens 8
    # [final] 与 CachedGenerator 全部一致 + 取消后零泄漏: True -> OK
docker run --rm liteinfer:cpu python -m liteinfer.server.main --help   # usage 正常
```

**踩过的坑**

- **本机 venv 从未装过 dev extras**：`fastapi/matplotlib` 缺失 → pytest 收集 3 个文件直接
  `ModuleNotFoundError`（Task 01-08 只需核心依赖，09-12 的 server/图表 测试在别处又复装过）。
  不猜直接看 `pip list` 确诊，`pip install -e ".[dev]"` 后 242 全绿。教训：venv 变更后先
  `pip list` 而不是盲跑测试。
- **PShell 至今无 `&&`**（Task 12 已记录，又踩）：`set HF_HOME=... && pytest` 直接 ParseError；
  统一用 `$env:HF_HOME=...; $env:PYTHONPATH=...; pytest` 的分行写法。
- **PyYAML 1.1 把 CI 的 `on:` 解析成 bool True**：对 `sorted(d.keys())` 做混合类型排序崩
  TypeError。GitHub 官方 Actions 接受裸 `on:`；本地验证脚本用对具体键访问即可，不要 sorted。
- **部署资产测试在镜像内全部 FileNotFoundError**：`.dockerignore` 按设计排除 `.github/` 与
  `docs/`，Dockerfile 本身也不 COPY 进 `/workspace`，镜像内当然没有这些文件。这不是 bug，
  是测试上下文边界没划清。加 module 级 `pytestmark = pytest.mark.skipif(not Dockerfile.exists())`
  守卫：仓库上下文真实断言、镜像上下文跳过，两种上下文各司其职。
- **首次 docker build 极慢的根因是 torch CPU wheel 下载走 download-r2.pytorch.org 只有
  ~150kB/s**（196MB 约 28 分钟）→ 一次构建后层缓存命中即秒级；所以 CI 的 docker job
  设成 workflow_dispatch 手动触发，不消耗每次 push。
- **torch 先装 + editable 后装的顺序是硬约束**：`pip install torch --index-url cpu` 之后再
  `pip install -e ".[dev]"`，pip 识别 torch 已满足依赖（log 显示 Requirement already satisfied），
  绝不会从默认 PyPI 回拉带 GPU 的 wheel。
- **PowerShell 里 `docker ... --help` 显示 exit -1**：是 --help 的正常退出语义，别当成失败；
  看的是 stdout usage，不是 exit code。

**下一阶段提示（Task 14 接入知微）**

- 部署面已就绪：LiteInfer 的 `/v1/chat/completions` 对外就是 OpenAI 兼容 base，知微的 LLM
  Provider 从外部 API 指向 LiteInfer 即可，无需改端点协议。
- `benchmark_demo --smoke` 与 `async_engine_demo` 在容器内已跑通：Task 14 验证真模型时
  `docker run -v D:/LiteInfer/hf_cache:/model-cache liteinfer:cpu ...` 是现成的快速路径。
- 服务默认 `--host 127.0.0.1`（容器内 CMD 已是 0.0.0.0）；知微同机接入可保持 0.0.0.0 或走端口映射。

---

<!-- 后续每个 Task 完成后，把 AI 生成的记录追加到这里 -->
