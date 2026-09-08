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

<!-- 后续每个 Task 完成后，把 AI 生成的记录追加到这里 -->
