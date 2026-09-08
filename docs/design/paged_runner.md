# Task 08 设计文档：ModelRunner 接入 Paged KV

**运行环境**：CPU（本机无 NVIDIA GPU，全程 `EngineConfig.device="cpu"`、`dtype=float32`）

---

## 1. 源代码

| 文件 | 职责 |
|---|---|
| `liteinfer/model/runner.py` | **新增**。`KVCacheView` 协议、`PagedLayerCache`（分页适配器）、`ModelRunner`（共享块池 + prefill/decode）、`infer_kv_dims` |
| `liteinfer/cache/paged.py` | 修改。`BlockTable.write_token`（单 token 单层写入 + 按需分块）、`gather_v`（读 V）、内部抽出 `_gather` |
| `liteinfer/engine/core.py` | 修改。forward 改走 `self.runner`；块表**首次 prefill 才分配**；终态/取消时 `_release` 归还物理块 |
| `liteinfer/engine/request.py` | 修改。`RequestState.cache` 语义改为 `BlockTable | None`，新增 `cache_bytes` |
| `liteinfer/config.py` | 修改。新增 `block_size`（默认 16）与 `num_blocks`（默认按调度预算推导） |
| `liteinfer/model/minimal/{attention,layer,model}.py` | 修改。仅类型标注：缓存参数从 `LayerKVCache` 放宽为 `KVCacheView` 协议 |
| `liteinfer/__init__.py` | 修改。惰性导出 `ModelRunner` / `PagedLayerCache` |

---

## 2. Unit Tests

`tests/test_paged_runner.py`，13 条快测 + 4 条 model 标记测试：

**快测（无模型）**

| 测试 | 钉住的行为 |
|---|---|
| `test_append_read_matches_contiguous` | 分页适配器与 Task 04 连续实现**逐位相等**（含跨块边界） |
| `test_read_after_partial_history_matches` | 只读部分历史（length < 已写入）也一致 |
| `test_gather_v_reads_value_not_key` | V 走独立读路径，不会误读成 K |
| `test_block_allocated_once_per_token_not_per_layer` | 块分配只与 token 位置有关，与层数无关（9 token / bs=4 → 3 块） |
| `test_exhausted_pool_fails_fast` | 池耗尽抛 `ValueError`，不静默越界写 |
| `test_invalid_inputs_rejected` | 形状/参数校验 |
| `test_bytes_for_matches_token_count` | 字节记账 = token 数 × `bytes_per_token()` |
| `test_runner_allocates_and_frees` | ModelRunner 分配/归还块 |
| `test_engine_releases_blocks_on_finish` | 多请求全部终态后 `num_blocks_used == 0` |
| `test_cancel_releases_blocks` | 取消路径立刻归还（docs/02 §10） |
| `test_waiting_requests_hold_no_blocks` | waiting 队列不占物理块（分页相对预分配的价值） |
| `test_multi_request_blocks_isolated` | 并发请求各自块表互不覆盖 |
| `test_block_size_respected` | `block_size` 配置生效 |

**model 标记（Qwen2.5-0.5B，CPU FP32）**

- `test_single_request_matches_contiguous`
- `test_multi_request_each_matches_contiguous`（含运行中动态加入第三个请求）
- `test_blocks_actually_used_during_generation`（防止上面的"归零"是空跑通过）
- `test_runner_dims_from_real_model`（24 层 / **2 个 KV 头**（不是 14 个 Q 头）/ head_dim 64）

---

## 3. Design Document

本文件。

---

## 4. 最小运行示例

```bash
set "HF_HOME=D:/LiteInfer/hf_cache"
set PYTHONPATH=d:/LiteInfer
python examples/paged_runner_demo.py --max-tokens 8
```

实测输出（Qwen2.5-0.5B, CPU FP32）：

```text
block_size=16 num_blocks=18 pool=7.1 MB
提交 3 个请求 -> waiting=3 used_blocks=0
step 1 之后: running=2 used_blocks=2
全部结束:   used_blocks=0 free_blocks=18

[prompt] The capital of France is
  paged     : ' Paris. It is the largest city in'  (8 tokens, length)
  contiguous: ' Paris. It is the largest city in'  (8 tokens, length)
  identical : True   cache_bytes=294912
...
[final] 输出全部一致: True | 块泄漏: False -> OK
```

---

## 5. 关键数据结构

### `PagedLayerCache(table, layer_idx)`

把共享 `BlockTable` 的**某一层**包装成单层 KV 缓存，对外只暴露两个方法：

```python
append(k_new[1,n,KVH,D], v_new, start) -> (k[1,start+n,KVH,D], v[1,start+n,KVH,D])
read(length)                           -> (k[1,length,KVH,D], v[1,length,KVH,D])
```

写入走 `BlockTable.write_token(layer, offset, k_vec, v_vec)`，读取走 `gather` / `gather_v`。

### `BlockTable.write_token(layer, offset, k_vec, v_vec)`（新增）

```python
logical = offset // block_size
while logical >= len(self.blocks):        # 按需分配，判据是 token 位置
    self.blocks.append(KVBlock(pool, pool.alloc(1)[0]))
self.blocks[logical].write_token(layer, offset % block_size, k_vec, v_vec)
self.num_tokens = max(self.num_tokens, offset + 1)
```

### `ModelRunner`

```
ModelRunner
├── paged : PagedKVCache          # 共享块池，构造时一次分配（inference_mode 之外）
│   └── pool : BlockPool           # [num_blocks, num_layers, block_size, KVH, D]
├── new_block_table() -> BlockTable      # 每请求一张，空表不占块
├── prefill(input_ids, table) -> logits  # 一次写 P 个 token
├── decode(token_id, position, table)    # 每步写 1 个 token（position 即写入偏移）
├── free_table(table) / bytes_for(table)
└── _handles(table) -> [PagedLayerCache(table, i) for i in range(num_layers)]
```

### 一次 decode 的数据流

```text
EngineCore._decode(st)
  → runner.decode(st.next_id, position=st.cached_len, st.cache)
      → model(step_input[1,1], position_ids=[[position]],
              kv_caches=[PagedLayerCache(table, i)], write_pos=position)
          → QwenSelfAttention.forward (第 i 层)
              ├─ q/k/v 投影 → QK-Norm(可选) → RoPE      # 旋转后才是要缓存的 K
              ├─ kv_cache.append(k[1,1,KVH,D], v, position)
              │     ├─ write_token(i, position, k_vec, v_vec)   # 落物理块
              │     └─ read(position+1) → gather/gather_v       # 取完整历史
              ├─ repeat_kv（GQA 展开 2 → 14 头）
              └─ scores → mask → softmax → @V → o_proj
```

---

## 6. 为什么这样设计

### 6.1 让分页适配器伪装成 `LayerKVCache`（最关键的一个决定）

`QwenSelfAttention` 里缓存相关的代码只有一行：

```python
k, v = kv_cache.append(k.transpose(1, 2), v.transpose(1, 2), write_pos)
```

只要 `PagedLayerCache` 提供同形的 `append` / `read`，**注意力层的计算逻辑一个字都不用改**，
Task 03 那些与 HF 逐层对齐的断言也不会被动到。分页这笔账被完全关在了 `cache` 与 `runner`
两个包里——`model/minimal/` 只改了类型标注。

代价是 `read` 从"零拷贝视图"退化成"gather 出来的副本"（见 Known Limitations）。

### 6.2 为什么要新增 `write_token`（而不是复用 `BlockTable.append`）

Task 07 的 `BlockTable.append(tokens_k, tokens_v)` 要求一次给出**所有层**
（`[num_layers, n, KVH, D]`）。但模型是**逐层**前向的，第 i 层只能拿到自己那层的
K/V，凑不齐 `num_layers` 这一维。所以必须有"单层单 token"粒度的写入入口。

反过来，块分配以 **token 绝对位置** `offset // block_size` 为唯一判据，
因此同一 token 被 24 层各写一次时，**只有第一层真正 `alloc`，其余 23 层复用同一个物理块**——
分配次数与层数无关（有单测钉住）。

### 6.3 块表为什么"首次 prefill 才分配"

Task 04 是 `submit` 时按 `prompt_len + max_tokens` 预分配一整段连续缓存。分页下如果
`submit` 就建表，虽然空表本身不占块，但把生命周期拉长会让"waiting 队列里的请求"
也持有表；改成被准入时才建，语义上正好对应 docs/02 §2 的 `Scheduler admission → PREFILL`。
更重要的是：**分页的优势就是不为未知长度预留**，提前建表会把这个优势讲丢一半。

### 6.4 为什么先记账再释放

`_release` 里先 `bytes_for(table)` 再 `free_table(table)`。因为 `free` 之后
`BlockTable.num_tokens` 归零，字节数再也问不出来，而 `RequestOutput.cache_bytes`
要在请求结束后仍可读（Task 10 的 KV utilization 指标会消费它）。

### 6.5 块池容量为什么按调度预算推导

```python
per_seq_tokens = max_new_tokens + 128        # 128 是给 prompt 的余量
num_blocks     = max(16, max_num_seqs * ceil(per_seq_tokens / block_size))
```

池太小会在生成中途 `alloc` 失败（体验最差），太大则一次性占掉几百 MB。
按"够跑满 `max_num_seqs` 条序列"推导，两头都照顾到；仍不够时 `BlockPool.alloc`
会 fail fast 抛 `ValueError`，比静默 OOM 好排查。

---

## 7. Alternative 方案

| 方案 | 为什么没选 / 何时选 |
|---|---|
| **A. 模型返回 `past_key_values`，由 Runner 统一写块表** | 能让写入彻底离开模型层（更"干净"）。但这会把 `forward` 的返回类型从 `Tensor` 变成 `Tensor \| tuple`，Task 03 所有 `assert_close(mine, hf)` 的对齐断言都得跟着改；而且每层的 K/V 要一路传到顶层再写回，多一次跨层搬运。等真要做 **prefix cache / chunked prefill** 时，块表写入需要跨请求协调，届时会重新考虑它。 |
| **B. 直接把请求拼成一个 `[B, T]` 大张量做真正的 batch 前向** | 才是"一次前向跑完全批"，也是 vLLM 吞吐的来源。但需要变长 attention（padding + 逐请求 mask + 逐请求 gather），会重写 `QwenSelfAttention` 的 mask 语义，风险集中在"最容易出错"的地方。当前 `ModelRunner.execute` 的**边界**（输入 ScheduledBatch、输出逐请求 logits）已经留好，未来替换内部实现即可，引擎与调度器不动。 |
| **C. 不改 `BlockTable`，给 `BlockPool` 加"按层连续"的第二套布局** | 等于放弃分页的物理块共享，退化成 contiguous，Task 11 的 prefix cache 就没法做了。 |
| **D. 用 `torch.Tensor` 的 `index_select` 代替 `gather`** | 语义等价。沿用 Task 07 已压测过的 `gather` 实现，是为了让"读路径"只有一份经过 10000 次压测验证的代码。 |

---

## 8. 当前 Known Limitations

1. **不是 PagedAttention CUDA Kernel**。读路径是 `torch.gather` 重建连续 K/V，
   逻辑等价、CPU 上逐位可验证，但**没有 fused kernel 的访存优化**。
2. **CPU 上比分页前更慢**。`LayerKVCache.read` 返回的是缓冲区切片（零拷贝视图），
   而 `BlockTable.gather` 每次都要造索引张量并真实搬运数据，每个 decode step 每层的
   K/V 各多一次拷贝（0.5B 约 96 次 gather/step）。**本阶段不宣称任何性能收益**，
   吞吐数字一律留给 Task 12 的 GPU benchmark。
3. **`execute` 尚未合并成单次 batch 前向**。当前 `ModelRunner.prefill/decode`
   是逐请求调用模型（引擎循环形状与 Task 06 一致），调度器给出的 `ScheduledBatch`
   决定"本步跑谁"，但还不是"一次 matmul 跑完"。见 Alternative B。
4. **块池容量是静态的**。构造时一次 `torch.zeros` 分配完，运行中不会扩缩容；
   `num_blocks` 推导式里的 `+128` 是 prompt 余量的粗略估计，超长 prompt 仍会 fail fast。
5. **没有 prefix cache / 块共享**。每张块表独占物理块，`ref_count` 属 Task 11。
6. **没有 chunked prefill**。超长 prompt 会在单个 step 里一次性 prefill 完，
   可能长时间占住该 step（Task 09 之后若要控尾延迟需再拆）。

---

## 9. 进入下一阶段前的验收命令

```bash
set "HF_HOME=D:\LiteInfer\hf_cache"
set PYTHONPATH=d:\LiteInfer

python -m pytest tests/test_paged_runner.py -q -m "not model"   # 13 passed
python -m pytest tests/test_no_hardcoded_cuda.py -q             # 1 passed
python -m pytest -q                                             # 133 passed（Task 01-07 不回归）
python -m pytest -m model -q                                    # 34 passed（含分页/连续逐字一致）
python examples/paged_runner_demo.py --max-tokens 8             # 输出一致 True，块零泄漏
```
