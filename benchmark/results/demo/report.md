# LiteInfer Benchmark 报告

> 诚实性声明：本报告全部数字来自 results.csv 的真实测量。
> CPU 上的数字仅验证流水线正确性，**不得**用于简历/汇报（docs/07 二、docs/05 §15）。
> 当前环境显存指标：N/A (no GPU)

## 1. 实验记录（docs/05 §14）

| 字段 | 值 |
|---|---|
| 模型 | Qwen/Qwen2.5-0.5B |
| 设备 | cpu |
| dtype | torch.float32 |
| PyTorch | 2.13.0+cpu |
| Transformers | 5.14.1 |
| Python | 3.12.10 |
| 平台 | Windows-11-10.0.26200-SP0 |
| CPU 核数 | 12 |
| Torch 线程 | 8 |
| block_size | 16 |
| 引擎顺序 | kv, batch, prefix |
| 并发数 | 1, 2, 4 |
| workload 口径 | decode(p=16,out=48); prefill(p=160,out=16); typical(p=48,out=24); sp(p=160,out=16) |
| smoke 模式 | True |
| 时间戳(UTC) | 2026-09-09T08:01:04+00:00 |
| 并发语义 | num_requests=concurrency：顺序驱动逐个跑，引擎驱动同时跑 |


## 2. 方法摘要

**驱动映射**（docs/07 Task 12 的 6 个命名消融 -> 5 个真实驱动，
continuous batching 与 paged KV 在本仓库合并进 EngineCore —— 见 docs/design/benchmark.md）：

| 命名消融 | 实际驱动 | 测量轴 |
|---|---|---|
| HF baseline | `hf`（顺序 HF generate） | E2E / 吞吐 |
| no-cache | `nokv`（顺序无缓存） | E2E / 吞吐 |
| KV (contiguous) | `kv`（顺序连续 KV） | E2E / 吞吐 |
| continuous batching | `batch`（EngineCore） | 吞吐 vs 并发 |
| paged KV | `batch`（EngineCore 块池） | KV utilization / 峰值块 |
| prefix cache | `prefix`（EngineCore + 前缀缓存） | 命中 token / TTFT 下降 |

- 并发定义：`num_requests = concurrency`。顺序驱动逐个排队跑，引擎驱动同时提交。
- 指标口径：TTFT=首 token - enqueue；TPOT=相邻 token 均值；ITL p50 为"每请求
  p50 的均值"；HF 未拆段计时，其 TTFT/TPOT/ITL 为 N/A。
- 所有 minimal 系 cell 已做输出逐字一致性校验（`parity` 列）；失败即整轮失败。
- 采样：greedy（temperature=0），模型/种子固定。

## 3. 全量结果

| engine | workload | conc | req/s | out tok/s | TTFT p50 | TPOT p50 | ITL p50 | E2E p50 | KV util | prefix hit | parity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| batch | decode | 1 | 0.055 | 2.638 | 934.9 | 367.1 | 342.1 | 18190.9 | 50.0% | 0.0 | OK |
| batch | decode | 2 | 0.065 | 3.100 | 1034.9 | 633.1 | 639.5 | 30789.2 | 50.0% | 0.0 | OK |
| batch | decode | 4 | 0.062 | 2.961 | 1636.5 | 1332.0 | 1304.7 | 64239.5 | 50.0% | 0.0 | OK |
| batch | prefill | 1 | 0.102 | 1.630 | 3745.2 | 404.5 | 379.5 | 9812.7 | 80.0% | 0.0 | OK |
| batch | prefill | 2 | 0.293 | 4.689 | 3275.5 | 230.1 | 193.7 | 6727.4 | 80.0% | 0.0 | OK |
| batch | prefill | 4 | 0.221 | 3.539 | 2301.0 | 1004.9 | 1166.9 | 17375.2 | 80.0% | 0.0 | OK |
| batch | sp | 1 | 0.076 | 1.218 | 4820.5 | 554.2 | 480.5 | 13133.6 | 87.5% | 0.0 | OK |
| batch | sp | 2 | 0.080 | 1.286 | 8407.2 | 1084.1 | 892.6 | 24669.5 | 87.5% | 0.0 | OK |
| batch | sp | 4 | 0.096 | 1.543 | 11812.5 | 1941.7 | 1527.2 | 40938.0 | 87.5% | 0.0 | OK |
| batch | typical | 1 | 0.104 | 2.502 | 1545.8 | 349.8 | 341.3 | 9591.5 | 55.6% | 0.0 | OK |
| batch | typical | 2 | 0.083 | 1.988 | 3339.1 | 897.0 | 835.4 | 23970.8 | 55.6% | 0.0 | OK |
| batch | typical | 4 | 0.100 | 2.411 | 4063.9 | 1529.5 | 1436.8 | 39241.9 | 55.6% | 0.0 | OK |
| kv | decode | 1 | 0.067 | 3.201 | 627.5 | 303.4 | N/A | 14986.3 | N/A | 0.0 | OK |
| kv | decode | 2 | 0.064 | 3.091 | 642.9 | 314.6 | N/A | 15525.6 | N/A | 0.0 | OK |
| kv | decode | 4 | 0.065 | 3.142 | 707.1 | 307.5 | N/A | 15329.5 | N/A | 0.0 | OK |
| kv | prefill | 1 | 0.147 | 2.350 | 3503.1 | 217.8 | N/A | 6790.1 | N/A | 0.0 | OK |
| kv | prefill | 2 | 0.274 | 4.390 | 762.0 | 190.5 | N/A | 3639.4 | N/A | 0.0 | OK |
| kv | prefill | 4 | 0.126 | 2.024 | 3306.3 | 304.4 | N/A | 7858.1 | N/A | 0.0 | OK |
| kv | sp | 1 | 0.099 | 1.584 | 4398.0 | 375.7 | N/A | 10069.7 | N/A | 0.0 | OK |
| kv | sp | 2 | 0.098 | 1.570 | 4350.9 | 386.2 | N/A | 10178.8 | N/A | 0.0 | OK |
| kv | sp | 4 | 0.105 | 1.676 | 4134.4 | 373.0 | N/A | 9762.3 | N/A | 0.0 | OK |
| kv | typical | 1 | 0.112 | 2.681 | 1524.8 | 320.6 | N/A | 8950.0 | N/A | 0.0 | OK |
| kv | typical | 2 | 0.114 | 2.746 | 1455.2 | 314.0 | N/A | 8734.5 | N/A | 0.0 | OK |
| kv | typical | 4 | 0.114 | 2.744 | 1472.0 | 308.2 | N/A | 8607.1 | N/A | 0.0 | OK |
| prefix | decode | 1 | 0.049 | 2.350 | 762.2 | 418.4 | 392.0 | 20427.6 | 50.0% | 0.0 | OK |
| prefix | decode | 2 | 0.073 | 3.506 | 951.1 | 561.5 | 634.2 | 27340.9 | 43.8% | 8.0 | OK |
| prefix | decode | 4 | 0.095 | 4.550 | 295.3 | 880.8 | 1243.2 | 41692.9 | 40.6% | 12.0 | OK |
| prefix | prefill | 1 | 0.110 | 1.760 | 3819.3 | 351.5 | 354.0 | 9091.8 | 80.0% | 0.0 | OK |
| prefix | prefill | 2 | 0.135 | 2.167 | 3739.5 | 723.7 | 696.8 | 14594.5 | 46.7% | 80.0 | OK |
| prefix | prefill | 4 | 0.163 | 2.605 | 4751.6 | 1289.6 | 1236.8 | 24095.2 | 30.0% | 120.0 | OK |
| prefix | sp | 1 | 0.107 | 1.706 | 3882.9 | 366.4 | 349.1 | 9379.6 | 87.5% | 0.0 | OK |
| prefix | sp | 2 | 0.133 | 2.132 | 4219.2 | 705.2 | 691.9 | 14797.1 | 50.0% | 96.0 | OK |
| prefix | sp | 4 | 0.149 | 2.382 | 4642.9 | 1438.5 | 1420.1 | 26252.5 | 31.2% | 144.0 | OK |
| prefix | typical | 1 | 0.125 | 3.011 | 1364.8 | 287.2 | 286.2 | 7970.9 | 55.6% | 0.0 | OK |
| prefix | typical | 2 | 0.109 | 2.628 | 1502.6 | 721.8 | 698.1 | 18103.8 | 38.9% | 24.0 | OK |
| prefix | typical | 4 | 0.120 | 2.879 | 2706.6 | 1310.8 | 1298.7 | 32855.2 | 30.6% | 36.0 | OK |


## 4. 校验与异常

- 全部 cell 的 `parity_ok` 均为 True：True。
- 引擎驱动 cell 结束后块账目检查（free+used==total 与 batch 的 used==0）通过后可
  产生数据；任何泄漏/账目不齐会直接抛错终止测量，不会混进结果。

## 5. Known Limitations

- CPU 单机数字只用于验证流水线正确性；真正的吞吐/延迟数字必须来自云端 GPU 跑次。
- prefix 与 batch 共用 EngineCore 驱动（V2/V3 合并），两效果的吞吐与 KV 利用率
  在同一运行的两个正交观察量里分别呈现，不能拆分出"独立的两套实现"做对照。
- prefix cache 命中只发生在同一引擎实例内（本 benchmark 每 cell 新建引擎），
  sp workload 的命中与 TTFT 下降都限定在 cell 内部的多请求共享前缀上。
- HF 驱动无 prefill/decode 拆段计时，其 TTFT/TPOT/ITL 列为 N/A。
- 顺序驱动的 TTFT≈prefill 耗时（不含首 token 采样），是近似值。
- block_size 消融（8/16/32/64）不在本 Task 的自动化矩阵内，作为后续工作留档。

## 6. 图表索引

- `e2e_p50_vs_concurrency.png`（脚注含实验记录）
- `kv_util_peak_vs_concurrency.png`（脚注含实验记录）
- `out_tokens_per_s_vs_concurrency.png`（脚注含实验记录）
- `sp_prefix_ttft_vs_concurrency.png`（脚注含实验记录）
- `tpot_p95_vs_concurrency.png`（脚注含实验记录）
- `ttft_p95_vs_concurrency.png`（脚注含实验记录）


## 7. 观察与结论（全部来自本报告 CSV 真实测量；不足之处明说）

- sp workload @concurrency=4: prefix 相对 batch 的 TTFT p50 下降 60.7%（11812.5 ms -> 4642.9 ms，hit tokens 见结果表）
- decode workload：batch 峰值吞吐 3.10 tok/s vs kv 顺序 3.20 tok/s（比率 0.97x，CPU 上仅供管线验证）
