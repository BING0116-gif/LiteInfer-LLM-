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

<!-- 后续每个 Task 完成后，把 AI 生成的记录追加到这里 -->
