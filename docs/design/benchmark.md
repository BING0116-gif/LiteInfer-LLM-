# Task 12 设计文档：Benchmark + Ablation

**运行环境**：本机 CPU（`EngineConfig.device="cpu"`、`dtype=float32`）验证流水线；
完整矩阵由 `benchmark/run_full_matrix.sh` 在云端 GPU 执行（`LITEINFER_DEVICE`/`LITEINFER_DTYPE` 走环境变量）。

---

## 1. 源代码

| 文件 | 职责 |
|---|---|
| `benchmark/liteinfer_benchmark.py` | **新增**。矩阵驱动器：workload 规格 / 提示词构造 / 5 驱动 cell 执行 / 汇总 / CSV+JSON 落盘（measure-only） |
| `benchmark/benchmark_charts.py` | **新增**。由 results.csv + env.json 渲 PNG（Agg 后端，脚注附实验记录） |
| `benchmark/benchmark_report.py` | **新增**。由 CSV+env（+charts 目录）组装 Markdown 报告 |
| `benchmark/run_full_matrix.sh` | **新增**。云端一键脚本：全 5 引擎 × 4 workload × 6 并发，一轮跑完并 commit 产物 |
| `benchmark/__init__.py` | **新增**。把 benchmark 声明为包，测试可 `from benchmark.xxx import ...` |
| `tests/test_benchmark_matrix.py` | **新增**。19 条无模型快测（workload/提示词/汇总纯函数/CSV 往返/引擎配置尺寸/图表渲染/报告组装） |
| `examples/benchmark_demo.py` | **新增**。CPU smoke 全链路示例：测量 -> 图表 -> 报告（验收主体） |
| `pyproject.toml` | 修改。`[dev]` 增加 `matplotlib>=3.9`（Task 12 图表硬依赖，本机已装） |

---

## 2. Unit Tests

`tests/test_benchmark_matrix.py`，19 条，全部无模型：

| 测试 | 钉住的行为 |
|---|---|
| `test_workload_spec_known_names` | 4 个 workload 的名字/kind/max_tokens 合法 |
| `test_workload_spec_smoke_shape_preserved` | smoke 口径**保形**（kind 不变、长度缩小） |
| `test_workload_spec_unknown_fails` | 未知 workload 名 fail fast |
| `test_planned_prompt_len` | sp 的总 prompt 长 = shared+tail |
| `test_build_prompt_text_macro_count` | FakeTokenizer 下提示词再编码 token 数精确 |
| `test_build_prompt_text_requires_positive` | n<=0 抛错 |
| `test_build_sp_prompts_share_head_and_differ` | 共享前缀一致 + 尾缀互不相同 |
| `test_percentile_basic` / `test_percentile_none_aware` | 分位数线性插值 + None 过滤/空序列 |
| `test_mean_opt` | 全 None 均值 = None（N/A 优于 0） |
| `test_aggregate_cell_pure` | 纯函数汇总：吞吐/分位数/None 列 |
| `test_aggregate_cell_engine_missing_cols_none` | 顺序驱动缺失的列汇总为 None 不是 0 |
| `test_engine_cfg_sizing_for_long_prompt` | 2048 prompt 的块池与 token budget 推导正确 |
| `test_csv_roundtrip` | CellRow 写入/读回逐字段一致（含 None <-> 空串） |
| `test_csv_fieldnames_stable` | CSV 列顺序 = 字段顺序（跨脚本契约） |
| `test_env_record_structure` | 实验记录字段齐备、无性能数字 |
| `test_charts_render_pngs` | 合成数据真实渲出 PNG（PNG magic 校验非空） |
| `test_report_generation` | 报告含 "N/A (no GPU)"（无 GPU 不填 0）与结果表 |
| `test_report_insights_only_from_data` | 缺数据时结论节输出"数据不足"而非编造 |

---

## 3. Design Document

本文件。

---

## 4. 最小运行示例（本机 CPU 验收主体）

```bash
set "HF_HOME=D:/LiteInfer/hf_cache"
set PYTHONPATH=d:/LiteInfer
python examples/benchmark_demo.py --smoke
```

等价于：`liteinfer_benchmark`（smoke 口径）-> `benchmark_charts` -> `benchmark_report`，
产物落在 `benchmark/results/demo/`（results.csv / env.json / charts/*.png / report.md）。
冒烟口径 = 4 个保形缩放 workload × 并发 1/2/4 × 引擎 kv/batch/prefix。

---

## 5. 关键数据结构

### `CellRow`（每个 (engine, workload, concurrency) 一行）

时间统一秒、缺失一律 `None`（CSV 写空串，展示层渲染 "N/A"）。三类列：

- **性能列**：`wall_s`、`req_per_s`、`out_tokens_per_s`、`ttft_p50/p95`、
  `tpot_p50/p95`、`itl_p50_mean/p95_mean`、`e2e_p50/p95`、`output_tokens_total/mean`；
- **KV 观测列**（仅引擎驱动）：`kv_bytes_mean`、`kv_peak_blocks`、
  `kv_blocks_used_end`、`kv_blocks_free_end`、`kv_blocks_total`、`kv_util_peak`；
- **校验/配置列**：`parity_ok`、`prefix_hit_tokens_mean`、`max_num_seqs`、
  `max_num_batched_tokens`、`num_blocks`、`block_size`。

### `env.json`（docs/05 §14 实验记录）

模型/设备/dtype/torch/transformers/python/平台/线程/时间戳 + 本轮 workload 口径
（full 或 smoke 的长度）+ 引擎与并发清单。图表脚注直接吃它。

### 消融映射（关键设计决策，已与用户确认）

| 命名消融 (docs/07) | 真实驱动 | 测量轴 |
|---|---|---|
| HF baseline | `hf`：顺序 HF `generate()` | E2E / 吞吐 |
| no-cache | `nokv`：`CachedGenerator(use_cache=False)` 顺序 | E2E / 吞吐 |
| KV (contiguous) | `kv`：`CachedGenerator(use_cache=True)` 顺序 | E2E / 吞吐 |
| continuous batching | `batch`：`EngineCore`（scheduler 动态批） | 吞吐 vs 并发 |
| paged KV | `batch`：同上驱动的块池层 | KV utilization / 峰值块 |
| prefix cache | `prefix`：`EngineCore(enable_prefix_cache=True)` | 命中 token / TTFT 下降 |

V2（continuous batching）与 V3（paged KV）在本仓库自 Task 06/08 起就已合并在
`EngineCore` 一个生产驱动里，仓库里不存在"独立的两套历史实现"可以分别拉出来跑。
为不重写已删业务逻辑，采纳"一线驱动、双测量轴"：同一运行的吞吐（V2 效果）
与 KV 利用率（V3 效果）是正交观察量，报告分两节呈现并显式声明合并事实。
`block_size` 消融（8/16/32/64，docs/05 §12）不在本轮自动化矩阵内，记为
Known Limitation 与后续工作。

---

## 6. 为什么这样设计

1. **measure / render / report 三层分离、以文件为契约**：每层可独立单测、
   可单独重跑（云端只需要 liteinfer_benchmark，图表/报告可在拉回本地后跑），
   避免一个"一站式脚本"既测又画又写，难以隔离故障。
2. **并发即请求数**：`num_requests = concurrency`。顺序驱动"排队逐个跑"、
   引擎驱动"同时提交"，同一吞吐口径（out_tokens / wall）下 cell 间可直接比较；
   GPU 上正是这个维度让 batch/prefix 出现相对顺序驱动的吞吐分离。
3. **引擎 cell 每 cell 新建 EngineCore（fresh 块池）**：块账目从零开始，
   结束时验 `free+used==total`（两组都验）+ batch 额外验 `used==0`（防泄漏
   契约，docs/02 §2）；prefix 引擎的共享块同样验账目平衡，避免"缓存吞块"。
4. **参照文本逐字校验**：四个 minimal 系驱动在同一 MinimalQwen 上 greedy
   生成，文本必须与参照逐字节一致——先证"测得是同一件事"，再谈性能
   （Task 04 kv_cache_benchmark 同款方法论）；hf 不一致仅警告（不同实现）。
5. **显式配置长 prompt 的池尺寸与 token budget**：默认推导（
   `max_new_tokens+128` token/序列）对 2048 prompt 会撞池抛错，budget 默认
   2048 会让 `submit` 直接 ValueError；`_engine_cfg` 按
   `ceil((prompt+max_tokens)/bs)+4` 每序列块数 × 并发显式推导，预算按
   `max(2048, prompt_len+8)` 对齐——这是"能测 2048 prompt 而不炸"的必要条件。

---

## 7. Alternative 方案

- **给 V2/V3 各造一套独立实现**（contiguous 预分配的批处理引擎当 V2、
  现有 EngineCore 当 V3）：工作量显著增大且要重造一段已被 Task 08 替换的
  旧逻辑，违背用户"不重写业务逻辑"约束。取舍：忠实于既有实现的合并形态，
  用双测量轴呈现两效果。
- **图表库用 pandas+matplotlib 或者纯 textplot**：pandas 对"读 CSV 分组画图"
  并不提供额外能力（矩阵小），保持 stdlib csv；textplot 无法满足 docs/05
  "必须画图"的交付要求，选 matplotlib（已是环境依赖，用户同意声明进 dev）。
- **每 cell 复用同一 EngineCore**：会跨 cell 累计 prefix 缓存块与指标注册表，
  "第二 cell"的数据被上一层污染。每 cell fresh 引擎的代价是构造开销（毫秒级，
  远小于一个 cell 的推理时长），换来账目确定性。

---

## 8. 当前 Known Limitations

- **CPU 数字仅验证流水线**：本机 0.5B@FP32 约 3 tok/s，全矩阵（5×4×6）在
  CPU 上会跑到小时级，故验收用 smoke 口径（缩放保形 + 并发 1/2/4 + 驱动
  kv/batch/prefix）。完整矩阵命令与数值只能来自云端 GPU。
- **V2/V3 合并**：continuous batching 与 paged KV 是同一驱动的两个测量轴，
  无法拆成两套独立实现对照（见 §5）。
- **HF 未拆段计时**：`hf` 行的 TTFT/TPOT/ITL 为 N/A（HF `generate()` 不暴露
  两段计时）；顺序驱动的 TTFT≈prefill 耗时（不含首 token 采样），是近似。
- **prefix 命中限同引擎实例内**：本 benchmark 每 cell 新建引擎，sp workload
  的命中/复用只表达"同一批 N 个请求共享前缀"，不表达跨批/跨服务的 prefix
  缓存持久性。
- **`prefix_cached_blocks` 列当前不携带值**：cell 结束后 core 已销毁，只剩
  evictable 块数可以可信读取，但把它放进列会与"释放后块数"语义混淆；命中
  效果由 `prefix_hit_tokens_mean` 与 TTFT 对比承担。
- **block_size 消融未自动化**（docs/05 §12 的 8/16/32/64），留作后续。
- **吞吐分母不含等待时间细分**：`req_per_s`/`out_tokens_per_s` 用总墙钟
  （含排队），与 vLLM 的 system-throughput 口径一致；不单独出"计算时间"。

---

## 9. 验收命令（本机 CPU 全绿）

```bash
set "HF_HOME=D:/LiteInfer/hf_cache"
set PYTHONPATH=d:/LiteInfer
python -m pytest tests/test_benchmark_matrix.py -q      # 19 passed（无模型）
python -m pytest tests/test_no_hardcoded_cuda.py -q      # 1 passed（benchmark/ 无裸 cuda）
python -m pytest -q                                      # 既有全量快测不回归
python examples/benchmark_demo.py --smoke                # 测量+图表+报告全链路，产物落 demo/
```