# Task 04 Design Document：Contiguous KV Cache

## 0. 运行环境（补充条款 A6）

```text
运行环境：CPU（FP32）；代码不依赖 GPU，上云仅需改 EngineConfig.device/dtype
本机验证命令：
  set PYTHONPATH=d:\LiteInfer
  pytest tests/test_kv_cache.py -q
  pytest -m model tests/test_kv_generation.py -q
  python examples/kv_cache_demo.py
  python benchmark/kv_cache_benchmark.py --device cpu
备注：CPU 上的加速比仅用于验证 KV 复用逻辑，不得写进简历/汇报（docs/07 Task 04）
```

## 1. 目标与范围

把 Task 03 的"每步全序列重算"升级为 **prefill 一次 + decode 逐步复用历史 KV**：

- 做：按层预分配的连续 K/V 缓冲区、prefill/decode 两阶段、past KV 复用、
  带偏移的因果掩码、延迟 benchmark、同模型消融对照；
- 不做：请求/调度/批处理（Task 05/06）、分页与共享（Task 07/08）、
  PagedAttention kernel（本项目明确不声称实现）。

## 2. 关键数据结构

```text
ContiguousKVCache                       # 外部持有，所有层共享一次预分配
├── KVCacheConfig                       # (L, KVH, D, max_seq_len, dtype, device)
├── k_cache: [L, max_seq_len, KVH, D]   # K
├── v_cache: [L, max_seq_len, KVH, D]   # V
└── LayerKVCache × L                    # 单层视图（缓冲区切片，不是副本）
      .append(k_new, v_new, start)      # 写入 [start, start+n)，返回 [:start+n] 视图
      .read(length)                     # 只读历史视图
```

单次前向的数据流（以 decode 一步为例，T = 已缓存长度）：

```text
input_ids [1,1]  position_ids [[T]]
      │
      ├─ embedding → 逐层 QwenDecoderLayer
      │       └─ self_attn: 投影 → QK-Norm → transpose → RoPE
      │                     → kv_cache.append(k, v, T)   ← 唯一一次数据搬运 O(1)
      │                     → repeat_kv → scores[q=1, k=T+1] → (无 mask)
      │                     → softmax(fp32) → out
      └─ lm_head → logits [1,1,vocab]
```

形状约定：

| 张量 | 形状 | 说明 |
|---|---|---|
| `k_new` / `v_new` | `[1, n, KVH, D]` | 本次新算的 token（prefill n=S，decode n=1） |
| 缓存物理布局 | `[max_seq_len, KVH, D]` | 与 docs/02 §7 一致；切成 block 即 Task 07 的 paged 布局 |
| attention 内 `k`/`v` | `[1, T, KVH, D]` → transpose → `[1, KVH, T, D]` | 消费前才转置，转置只改步长 |
| `build_causal_mask(S, past_len)` | `[1, 1, S, S+past_len]` | `key_pos <= query_pos` 可见 |

容量核算（Qwen2.5-0.5B，FP32）：

```text
bytes/token = 2 × 24 层 × 2 KV 头 × 64 head_dim × 4 B = 24 KB
32 token 的缓存 ≈ 768 KB（vs 模型权重约 2 GB）
```

## 3. 为什么这样设计

1. **缓存由外部持有，模型只负责读写。**
   KV 的生命周期跨多次 forward，到 Task 07 还要跨请求共享（同一物理 block
   被多条序列引用）。缓存在模型内部的话，这些优化无从下手。模型侧只接收
   `kv_caches` 列表并往里写，所有权留给 generator / 未来的 engine。
2. **追加式写入而不是"返回 past_key_values 再 cat"。**
   常见写法每步 `torch.cat([past, new])` 会把整段历史复制一遍（decode 时长度
   就是 T），本方案每步只写 n 个新 token（decode n=1），再返回 `[:start+n]`
   的**视图**，历史零拷贝。测试 `test_read_returns_view_without_copy` 用
   storage 指针钉住了这一点。
3. **缓存在 RoPE 之后写入。**
   缓存里存的是"旋转后的 K"，与推理期语义一致；若存旋转前的 K，decode 时
   必须先把历史重算一遍旋转，缓存就白做了。`test_cache_stores_rotated_keys`
   直接比对缓存内容与手工重放的旋转结果。
4. **`build_causal_mask` 增加 `past_len` 而不是新增函数。**
   带历史时 query 长度 ≠ key 长度，`triu` 的方阵语义失效；统一用
   `key_pos <= query_pos` 表达，`past_len=0` 时与原实现逐值等价（有单测）。
   decode（`seq_len == 1`）时单个 query 恒可见全部历史，直接省掉掩码加法。
5. **模型始终只返回 logits 张量，不返回 `past_key_values`。**
   历史已由外部缓存承载；再返回一份会把返回类型变成 `Tensor | tuple`，
   Task 03 那些 `assert_close(mine, hf)` 的断言就全得改。向后兼容是硬约束：
   Task 03 的 16 条算子单测与 7 条对齐测试一行未动，全部通过。
6. **消融基线用同一个 MinimalQwen（`use_cache=False`），不用 HF 模型。**
   拿 HF 比会把"算子实现差异"混进"KV 复用收益"，消融结论不可信。
   HF 仍然作为**正确性**参照（端到端文本与 `ManualGenerator` 逐字一致）。
7. **EOS 解析抽成 `liteinfer/model/eos.py` 供两条链共用。**
   Task 02 已踩坑：Qwen2.5 的终止符是 `<|im_end|>`(151645) 而不是
   tokenizer 的 `<|endoftext|>`(151643)。两边各自解析，只要一边选错，
   "输出是否一致"就会以"长度不同"的形式失败，看起来像数值问题。
8. **benchmark 强制 warmup + repeat + 中位数。**
   实测冷启动比热态慢 3~5 倍（线程池初始化/内存分配/算子选择），不预热时
   测出来的是"谁先跑"；单次测量在 CPU 上也不足以支撑结论。
9. **显存指标无 GPU 时输出 `N/A (no GPU)`**（补充条款 A3），禁止填 0。

## 4. Alternative 方案

| 方案 | 取舍 |
|---|---|
| 模型返回 `past_key_values`，调用方拼回缓存 | 否——每步 O(T) 历史拷贝；且返回类型污染既有断言 |
| 让缓存长在 `QwenSelfAttention` 内部 | 否——Task 07 的 block 共享与 prefix cache 需要外部管理生命周期 |
| 缓存存旋转前的 K，decode 时再旋转 | 否——历史重算是 O(T)，且错误表现隐蔽（文本"看着还行"） |
| `nn.functional.scaled_dot_product_attention` + `is_causal` | 暂不用——Task 04 要保持算子路径可对齐、可单步对照；后续 Task 可替换，接口已是 additive mask |
| 每步 `torch.cat` 拼接历史 | 否——等价于方案 1，多一次全量读写 |
| 缓存按 Q 头数分配 | 否——GQA 下 KV 头只有 2 个（Q 有 14 个），按 Q 头存会放大 7 倍显存 |
| 直接用 HF 的 `DynamicCache` | 否——那不是"自主管理 KV Cache"（docs/07 最终必须完成 10 项之第 3 项） |
| 缓存容量写死到 config | 暂不——Task 04 按 `prompt + max_tokens` 逐请求分配；Task 06 引入调度器时再统一为 `max_model_len` |

## 5. Known Limitations

1. **batch=1**：`LayerKVCache.append` 显式拒绝 batch>1。多条序列各有不同长度，
   需要分页与 block table —— 正是 Task 07 的内容。
2. **预分配定长 → 内部碎片**：容量按 `prompt + max_tokens` 一次性分配，
   提前 EOS 结束时尾部空间浪费；多条请求无法共享尾部 block（Task 07 解决）。
3. **decode 每步仍要读全量历史 KV**：本阶段只消除了"重算"，没消除"重读"。
   真正的收益上限取决于内存带宽，GPU 上的加速比会显著高于 CPU。
4. **非零拷贝的注意力**：`append` 是 O(1)，但 attention 仍把历史 K/V 读进
   matmul；不存在自定义 kernel，也不声称实现了 PagedAttention。
5. **CPU 性能数字不可外推**：本机实测约 3 tok/s（0.5B FP32），且冷启动抖动
   达数倍；所有性能结论必须来自 Task 12 的 GPU benchmark。
6. **缓存不跨请求复用**：prefix cache（Task 11）尚未实现，相同 system prompt
   会被重复 prefill。
7. **无 chunked prefill**：超长 prompt 必须一次塞进 prefill，峰值内存随
   prompt 长度平方增长（causal mask 与 scores）。

## 6. 验收标准与实测（本机 CPU / FP32）

正确性（必须全绿，且都在 CPU 上验证）：

| 判据 | 命令 | 结果 |
|---|---|---|
| 缓存本体与掩码单测（无模型） | `pytest tests/test_kv_cache.py -q` | 19 passed |
| prefill/decode logits 与 HF 对齐（atol=rtol=1e-4） | `pytest -m model tests/test_kv_generation.py -q` | 8 passed |
| 端到端文本与 `ManualGenerator`(HF) 逐字一致 | 同上 | 一致 |
| 自带 no-cache 路径与 KV 路径一致 | 同上 | 一致 |
| 全量不回归 | `pytest -m model -q` / `pytest -q` | 26 passed / 74 passed |
| 无裸设备字面量 | `pytest tests/test_no_hardcoded_cuda.py -q` | 通过（含新增 `benchmark/`） |

性能（仅逻辑验证，不可写进简历）：

```text
model=Qwen/Qwen2.5-0.5B device=cpu dtype=torch.float32 repeat=2 warmup=1
[len= 16] no-cache   7.806s ( 2.07 tok/s) | kv-cache   4.766s ( 3.46 tok/s) | speedup 1.64x
[len= 32] no-cache  18.695s ( 1.71 tok/s) | kv-cache   9.708s ( 3.63 tok/s) | speedup 1.93x
明细：kv-cache 32: prefill 0.387s / decode 8.355s，cache 909312 B
     no-cache 32: prefill 0.397s / decode 18.244s
peak memory: N/A (no GPU)
```

两条观察，与理论一致：

1. 加速比随生成长度上升（1.64x → 1.93x）：no-cache 是 O(T²)，KV 是 O(T)；
2. KV 版的 per-token 成本基本恒定（3.46 → 3.63 tok/s），no-cache 则明显下降
   （2.07 → 1.71 tok/s）——这正是"每步重算整条序列"的代价。

## 7. 关键接口签名

```python
KVCacheConfig(num_layers, num_kv_heads, head_dim, max_seq_len, dtype, device)
    .bytes_per_token() -> int          # 2 × L × KVH × D × element_size
    .total_bytes() -> int
    .from_hf_config(hf_cfg, max_seq_len, dtype, device)

LayerKVCache.append(k_new[1,n,KVH,D], v_new, start) -> (k[1,start+n,KVH,D], v)
LayerKVCache.read(length) -> (k[1,length,KVH,D], v)
ContiguousKVCache(cfg).layer_caches -> list[LayerKVCache] ; .nbytes ; .reset()

build_causal_mask(seq_len, device, dtype, past_len=0) -> [1,1,S,S+past]
QwenSelfAttention.forward(hidden, position_ids, attention_mask=None,
                          kv_cache=None, write_pos=0) -> Tensor
MinimalQwenForCausalLM.forward(input_ids, position_ids=None,
                               kv_caches=None, write_pos=0) -> logits[1,S,V]
CachedGenerator(model, tokenizer, cfg, eos_ids=None) / .from_config(cfg)
CachedGenerator.generate(prompt, params=None, use_cache=True, max_seq_len=None)
    -> CachedGenerationOutput(text, prompt_tokens, output_tokens, finish_reason,
                              latency_s, tokens_per_s, device, dtype,
                              prefill_latency_s, decode_latency_s,
                              cached_tokens, cache_bytes)
resolve_eos_ids(model, tokenizer) -> frozenset[int]     # Task 02/04 共用
```

## 8. 进入下一阶段前的验收命令

```bash
set PYTHONPATH=d:\LiteInfer
pytest -q                                     # 74 passed（无模型，秒级）
pytest -m model -q                            # 26 passed（含 Task 01~03 不回归）
python examples/kv_cache_demo.py              # 输出一致 True，加速比 > 1
python benchmark/kv_cache_benchmark.py --device cpu
python benchmark/kv_cache_benchmark.py --device <gpu> --dtype fp16   # Task 12 上云后
```
