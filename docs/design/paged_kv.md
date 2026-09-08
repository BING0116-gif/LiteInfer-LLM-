# Task 07 设计文档：BlockPool + Paged KV Cache

> 运行环境：CPU / GPU 两者皆可（本机 CPU FP32 验证）
> 本机验证命令：见文末「进入下一阶段前的验收命令」
> 关联代码：`liteinfer/cache/paged.py`，单测 `tests/test_paged_kv.py`

---

## 1. 源代码

`liteinfer/cache/paged.py` 提供五个类，职责层层收敛：

| 类 | 职责 |
|---|---|
| `FreeQueue` | 空闲物理块下标的 FIFO；`pop/push/size/empty`；**双重释放检测** |
| `BlockPool` | 持有物理张量 `[num_blocks, num_layers, block_size, KVH, D]×2`；`alloc/free/usage/nbytes` |
| `KVBlock` | 单物理块句柄 `(pool, block_id)`；`write_token(layer, offset, k, v)` / `k_view/v_view` |
| `BlockTable` | 单请求「逻辑块列表 + num_tokens」；`append`（lazy 取块）/ `gather`（gather 读）/ `free` |
| `PagedKVCache` | 顶层管理器：组合 `BlockPool` + 每请求 `BlockTable`；`new_block_table/append_tokens/free_table/usage` |

所有 `device`/`dtype` 一律来自 `KVCacheConfig`（源头 `EngineConfig`），模块内不出现任何设备决策。

---

## 2. Unit Tests

`tests/test_paged_kv.py`（16 条，无真实模型，CPU 秒级）：

- `FreeQueue`：pop/push、双重释放抛错、空队列 pop 抛错；
- `BlockPool`：alloc/free 计数、耗尽抛错、双重释放抛错、usage 与 nbytes 公式；
- `KVBlock`：单 token 写入读取精确相等；
- `BlockTable`：单块（n<block_size）、跨块（n=40→3 块）重建逐位相等、部分读（length<n）、越界读抛错；
- 多请求隔离：两请求物理块不相交、各自读回正确；
- `PagedKVCache`：管理器 usage、gather 输出形状与 Task 04 `LayerKVCache.read` 一致（保证 Task 08 可无缝替换）；
- **压力测试 `test_stress_10000_no_leak`**：随机 10000 次「建表→写随机长度→gather 逐位相等→释放」，结束断言 `num_free == total`（零泄漏）。

全部通过：`16 passed in ~48s`（压测 10000 次为本机实测耗时）。

---

## 3. Design Document（本文件）

见其余各节。

---

## 4. 最小运行示例

`examples/paged_kv_demo.py`：一个池服务 3 个并发请求（长度 10/40/7，含跨块），写入确定值 K/V，gather 读回逐位一致，释放后 `free == total`。输出：

```text
[init] PagedKVCache(block_size=16, num_blocks=32, used=0, free=32, nbytes=262144)
[req 0] seq_len=10 blocks=1 allocated=1 readback_ok=True free_blocks=31
[req 1] seq_len=40 blocks=3 allocated=3 readback_ok=True free_blocks=28
[req 2] seq_len=7  blocks=1 allocated=1 readback_ok=True free_blocks=27
[free req 0] used=4 free=28
[free req 1] used=1 free=31
[free req 2] used=0 free=32
[final] PagedKVCache(block_size=16, num_blocks=32, used=0, free=32, nbytes=262144)
OK: 全部请求释放后空闲块数复原，零泄漏
```

---

## 5. 关键数据结构解释

**物理块池（BlockPool）**
```python
k_blocks: Tensor[num_blocks, num_layers, block_size, num_kv_heads, head_dim]
v_blocks: Tensor[num_blocks, num_layers, block_size, num_kv_heads, head_dim]
```
每个「物理块」= 一块 `[num_layers, block_size, KVH, D]` 的 K/V 存储，跨所有层。这是 docs/02 §7 推荐布局的直接落地。

**逻辑→物理映射（BlockTable.blocks）**
```python
blocks: list[KVBlock]   # blocks[i] 是逻辑块 i 对应的物理块
num_tokens: int          # 已写入 token 数
```
位置映射：`logical_block = pos // block_size`，`offset = pos % block_size`，`physical = blocks[logical_block].block_id`。

**FreeQueue**：物理块下标的空闲集合，`push` 时检测重复（双重释放 → 抛错），是「无泄漏」不变量的最底层保障。

---

## 6. 为什么这样设计

- **为什么用 block 而非 contiguous 整段**：Task 04 按 `prompt_len + max_tokens` 逐请求预分配连续段，内部/外部碎片大、请求间无法共享。分页把容量切成固定块，按需 lazy 取/还，请求结束即整块回收，碎片与泄漏面大幅收敛。
- **为什么 lazy 分配（append 时才取块）而非建表时预占**：序列长度在生成前未知，vLLM 也是随序列增长逐块向池要块；这样既能自然触发 `FreeQueue.alloc` 压力路径，又避免为每个请求预占满整段容量。
- **为什么用 `torch.gather` 做读**：`gather` 沿 (block, offset) 两维选槽重建连续 K/V，逻辑等价于 PagedAttention 的 KV 读取，且 CPU 可跑、可单测；Task 08 的 ModelRunner 直接复用 `BlockTable.gather`，无需改注意力实现即接入。
- **为什么双重释放要抛错**：双重释放会让 free 计数虚高、真实泄漏被掩盖，是分页系统最阴险的 bug 之一；`FreeQueue` 用 `set` 成员判定在 `push` 即刻暴露。
- **为什么在 inference_mode 外分配张量**：Task 04 已踩坑——模式内新建张量是 inference tensor，离开上下文后原地 `copy_` 会抛 RuntimeError；decode/append 每步都要写物理块，故池在构造时（模式外）一次性 `torch.zeros` 预分配。

---

## 7. Alternative 方案

- **复用 ContiguousKVCache 做单请求分页**：不引入 BlockPool，每个 block 用一个 ContiguousKVCache。缺点：块间无统一空闲管理、无法跨请求共享物理内存、FreeQueue/usage 指标难统一度量，与 vLLM 式分页背离。
- **append 时整段 `torch.cat` 而非 gather 读**：写时仍每次复制历史。与 Task 04 同样的 O(T) 流量问题，分页的内存复用优势被抵消。
- **用高级索引（fancy indexing）替代 `torch.gather`**：功能等价且更短，但 `gather` 更贴合「按索引选槽」的语义、更贴近 CUDA PagedAttention 原始形态，作为教学/迁移到 Task 08 的桥梁更清晰。

---

## 8. 当前 Known Limitations

- **不是完整的 PagedAttention CUDA Kernel**：`gather` 在 CPU 上用 `torch.gather` 重建连续 K/V 后再走普通 attention，并非 GPU 融合 kernel；算力路径与 vLLM 的 CUDA 实现完全不同。本 Task 仅交付缓存管理层与正确性验证。
- **尚未接入 EngineCore / attention**：`RequestState.cache` 仍是 Task 04 的 `ContiguousKVCache`，引擎 step 循环未改动；把 `BlockTable.gather` 接入注意力、把 `PagedKVCache` 作为 `cache` 字段替换，属 **Task 08 ModelRunner**。
- **无前缀共享 / 引用计数**：本 Task 只做「每请求独占物理块 + 释放回收」，没有 Task 11 的 prefix cache（块哈希 + ref count + 复用）。当前多请求不共享物理块。
- **无抢占式回收 / 块驱逐**：池耗尽时 `alloc` 直接抛错（fail fast），没有按优先级驱逐低优请求的块；这是 LiteInfer 后续（含 prefix cache）才需要的能力。

---

## 9. 进入下一阶段前的验收命令

```bash
set "HF_HOME=D:\LiteInfer\hf_cache"
set PYTHONPATH=d:\LiteInfer

python -m pytest tests/test_paged_kv.py -q        # 16 passed（含 10000 次压测）
python -m pytest tests/test_no_hardcoded_cuda.py -q  # 1 passed（无裸 cuda 字面量）
python -m pytest -q                               # 120 passed（Task 01-06 不回归）
python examples/paged_kv_demo.py                  # 多请求读回逐位一致，释放后零泄漏
```

**压测硬指标**：`test_stress_10000_no_leak` 结束后断言 `pool.num_blocks_free == pool.num_blocks_total`（零泄漏）为通过必要条件。
