# Task 11 设计文档：Prefix Cache

运行环境：**CPU**（Task 01–11 约束；无 GPU 依赖，设备/dtype 全部经 `EngineConfig` 进入）。

本机验证命令：

```bash
set "HF_HOME=D:\LiteInfer\hf_cache"
set "PYTHONPATH=d:\LiteInfer"
python -m pytest tests/test_prefix_cache.py -q -m "not model"   # 24 passed
python -m pytest tests/test_no_hardcoded_cuda.py -q             # 1 passed
python -m pytest -q                                             # 217 passed（全快测不回归）
python -m pytest -m model -q tests/test_prefix_cache.py         # 1 passed（真模型）
python examples/prefix_cache_demo.py --max-tokens 16
```

## 1. 源代码

```text
liteinfer/cache/prefix.py          # block_hash（链式哈希）+ PrefixCache（哈希表/引用计数/LRU）
liteinfer/cache/paged.py           # BlockTable：+prefix_cache 字段 / adopt() / 分配与释放改道
liteinfer/cache/__init__.py        # 导出 PrefixCache / block_hash
liteinfer/model/runner.py          # ModelRunner 持有 PrefixCache；prefill(+write_pos)
liteinfer/config.py                # EngineConfig.enable_prefix_cache（默认 False）
liteinfer/engine/request.py        # RequestState/RequestOutput +prefix_hit_tokens、+prompt_token_ids
liteinfer/engine/core.py           # _prefill lookup/adopt/后缀 forward/register；_decode 块写满注册
liteinfer/observability/metrics.py # RequestMetrics +prefix_hit_tokens；snapshot +prefix 观测
liteinfer/__init__.py              # 惰性导出 PrefixCache
```

## 2. Unit Tests

`tests/test_prefix_cache.py`，25 条（24 快测 + 1 model）：

- **哈希**：确定性、前缀敏感（父哈希不同则哈希不同）、内容敏感；
- **注册/命中**：注册后命中同一物理块、断链即停、prompt 恰好整块时必须放弃最后一块、
  不足一块的 prompt 不可缓存、重复注册先到先得；
- **引用计数/LRU**：释放进 LRU 不还池、再命中移出 LRU、未追踪块直接还池、
  池空按 LRU 驱逐且从哈希表摘除、被在飞请求收养（ref>0）的块不可驱逐；
- **数值正确性**：收养 + 后缀续写后 gather 与"从零写一遍"**逐位相等**（FakeLM
  不读 KV，引擎级对比测不出收养块的内容错误，这条是复用不改数值的直接证据）；
- **引擎集成（FakeLM）**：第二请求命中 32 token 且文本逐字一致、关闭开关零命中
  且 `used_blocks==0`、同批并发提交也命中、多轮对话（decode 块注册）命中更长前缀、
  300 轮随机前缀压测零泄漏、metrics 快照透出 prefix 观测；
- **真模型（model）**：Qwen2.5-0.5B 第二请求命中且与关闭缓存的引擎**逐字一致**，
  prefill 耗时下降。

## 3. 关键数据结构解释

```python
block_hash(parent, tokens) -> int      # h_i = blake2b(h_{i-1} 的 8 字节 + 块内 token ids)
PrefixCache(pool, block_size)
    _by_hash:  dict[hash, block_id]    # 哈希表：内容 -> 物理块（同内容只有一个代表块）
    _hash_of:  dict[block_id, hash]    # 反查表：驱逐时 O(1) 摘除
    _refs:     dict[block_id, ref]     # 引用计数（仅被追踪的块）
    _lru:      OrderedDict[block_id]   # ref==0 的可驱逐/可复用块，队首最旧
    lookup(token_ids) -> (ids, hit_len)      # 收养：ref+1 并移出 LRU
    register(token_ids, table) -> int        # 已存在哈希跳过；返回新注册块数
    release_block(block_id)                  # ref-1；归 0 进 LRU；未追踪直接还池
    alloc_new() -> block_id                  # 池空 -> LRU 驱逐；无可驱逐 fail fast
BlockTable.adopt(block_ids, num_tokens)      # 命中块插到 blocks 头部（仅限空表）
```

复用路径（`EngineCore._prefill`）：

```text
lookup(prompt) → 命中 2 块 → table.adopt(ids, 32) → prefill(后缀 16 token, write_pos=32)
→ 注册新算出的完整块 → decode 每写满一块即注册
请求终态 _release → free_table → 逐块 release_block（缓存块 ref-1，未追踪块还池）
```

## 4. 为什么这样设计

- **链式内容哈希**：父哈希进链使"块内容相同但前缀不同"必然不同哈希——这是
  两个请求可安全共享同一物理块的充分判据；blake2b 而非内置 `hash()`，避免
  PYTHONHASHSEED 随机盐导致跨进程不可复现。
- **哈希表挂 PagedKVCache 层**（Task 10 遗留提示预留的位置）：块粒度的复用
  天然属于块池管理方；`BlockTable` 只持有一个可选引用做分配/释放改道，
  `paged.py` 的 gather/write 逻辑零改动（开闭原则，Task 07 的 16 条单测不动）。
- **只缓存完整块**（docs/02 §11）：半块的上下文尚未封口；且最后一个 token 的
  logits 必须由 forward 产出——因此 prompt 恰好 N 个完整块全命中时放弃最后一块。
- **ref==0 不还池而是进 LRU**：这是"白捡"的前提；块池耗尽时按 LRU 驱逐，
  被驱逐块物理 id 直接挪给新属主（不经过 FreeQueue，账目仍平衡）。双重释放
  防护（Task 07）对未追踪块依旧生效。
- **注册时机必须"KV 已完整落块"**：prefill 后 candidate 的 KV 尚未写入
  （下一步 decode 才写），若提前注册，同批后 prefill 的请求可能收养到
  "最后槽位是零"的半成品块。decode 的注册点放在 forward 之后、按
  `cached_len % block_size == 0` 判断。
- **默认 `enable_prefix_cache=False`**：开启后终态请求的完整块"故意占住"池，
  `kv_blocks_used` 不再归零，会打破 Task 09/10 "请求结束后块全回收"的既有契约
  与测试；关闭时引擎行为与 Task 08/09/10 逐位一致，也为 Task 12 ablation
  （paged vs paged+prefix）保留现成对照组。

## 5. 最小运行示例

`examples/prefix_cache_demo.py`：3 个共享 system prompt 的请求，对照开关两组。

实测（Qwen2.5-0.5B, CPU FP32, block_size=16, --max-tokens 16）：

```text
===== prefix OFF =====
[prompt= 48 tok] hit=  0 tok | prefill=  1593.7 ms
[prompt= 47 tok] hit=  0 tok | prefill=  1526.2 ms
[prompt= 49 tok] hit=  0 tok | prefill=  1675.5 ms
===== prefix ON  =====
[prompt= 48 tok] hit=  0 tok | prefill=  1600.3 ms
[prompt= 47 tok] hit= 32 tok | prefill=   829.5 ms
[prompt= 49 tok] hit= 32 tok | prefill=   877.5 ms
prefix stats: {'lookups': 3, 'hits': 2, 'hit_tokens': 64, 'evictions': 0,
               'cached_blocks': 6, 'evictable_blocks': 6}
```

共享前缀 32 token 命中后 prefill 约 **1.9x** 加速，且 ON/OFF 两组输出文本逐字一致。

## 6/7. Alternative 方案

| 方案 | 取舍 |
|---|---|
| **按块内容哈希（本实现）** | 无需模型参与，CPU 上即可验证；代价是只缓存完整块 |
| 全序列单哈希（整个 prompt 一个 key） | 实现最简单，但前缀必须逐 token 相同，共享率低；且无法部分命中 |
| 精确 KV 比对（比张量内容） | 无哈希碰撞问题，但 O(n) 张量比较比 O(1) 哈希贵得多，且语义与 vLLM 不一致 |
| 独立缓存池（另开一块物理内存存共享前缀） | 不占用请求块池，但同一内容存两份、需要额外的拷贝路径，收益为负 |
| 驱逐策略 LFU/随机 | LRU 与"最近被用过的前缀更可能再被用"的负载直觉一致，且实现 O(1) |

## 8. Known Limitations

1. **默认关闭**：`EngineConfig.enable_prefix_cache=False`。开启后
   `kv_blocks_used` 含"故意占住"的缓存块，`kv_utilization` 语义变为
   "占用（含缓存）"；缓存持有的块数单看 `/metrics` 的 `prefix_cached_blocks`。
2. **哈希碰撞未防护**：64 位 blake2b 碰撞概率工程上可忽略（~2^-64），
   未做"命中后再比对 KV 内容"的二次校验（vLLM 同样不做）。
3. **写路径仍是逐 token 逐层 copy**：复用省的是 prefill 计算，不是写带宽；
   gather 读路径的固有开销（Task 08 已声明）不变。
4. **块池容量静态**：`num_blocks` 构造时定死，缓存块与在飞块共享同一池；
   极端情况下缓存块会被频繁驱逐（`n_evictions` 可观测）。
5. **批量 prefill 未合并**：本 Task 不改 Task 08 的逐请求执行模型，命中只
   减少单请求的 prefill 长度。
6. **这不是完整 PagedAttention CUDA Kernel**（延续 Task 07/08 声明）：
   前缀复用发生在 CPU 的块管理层，与 vLLM 的 GPU kernel 融合路径不同。

## 9. 进入下一阶段前的验收命令

见文档开头"本机验证命令"。全部在本机 CPU 通过：

```text
tests/test_prefix_cache.py -m "not model" : 24 passed
tests/test_no_hardcoded_cuda.py           : 1 passed
pytest -q                                 : 217 passed, 37 deselected（Task 01-10 不回归）
tests/test_prefix_cache.py -m model       : 1 passed
tests/test_paged_runner.py -m model       : 4 passed（runner 改动回归）
examples/prefix_cache_demo.py             : hit=32, prefill 1600ms -> 830ms，输出逐字一致
```

**下一阶段提示（Task 12）**：`enable_prefix_cache` 开关即 ablation 的
"paged vs paged+prefix" 对照组；`PrefixCache.stats()` 与
`/metrics` 的 `prefix_hit_tokens_total` 可直接写入 benchmark CSV。
