# Task 10 设计文档：Metrics + Tracing

> 运行环境：CPU（本机验证）／GPU（Task 12 云端）两者皆可
> 本机验证命令见文末「验收命令」

## 1. 目标

为引擎补上 docs/02 §1 架构图中的"旁路：Metrics / Tracing"，交付 docs/07 §四
Task 10 清单的全部 8 项：TTFT、TPOT、ITL、E2E、running/waiting、KV
utilization、output tokens/s、request trace。硬验收：**一次请求可输出完整
时间线**。

## 2. 关键数据结构

### 2.1 打点层（engine，改动最小化）

`RequestState` 新增三个字段（`liteinfer/engine/request.py`）：

| 字段 | 含义 | 打点位置 |
|---|---|---|
| `prefill_start_s` | 被调度准入、prefill 开始（= 排队结束） | `EngineCore._prefill` |
| `prefill_end_s` | prefill forward 完成（不含采样） | 同上 |
| `token_times: list[float]` | 每个 token 的产出时刻，与 `Request.generated` 按下标一一对应 | `EngineCore._after_emit` |

既有字段 `wall_start`（= enqueue 时刻）与 `wall_end`（= 终态时刻）沿用
Task 05 的语义，不重复打点。

打点刻意**只存时刻、不做计算**：热路径上的开销是一次 `perf_counter()` 加一次
list append，指标还原全部推迟到请求结束后（或查询时）的纯函数里。

### 2.2 指标层（liteinfer/observability/metrics.py）

- `RequestMetrics`（frozen dataclass）：单请求完整延迟画像。核心字段：
  - `ttft_s = token_times[0] - wall_start`（覆盖排队 + prefill）；
  - `tpot_s = (last_token - first_token) / (n-1)`，**单 token 请求为 None**
    （与 OpenAI usage 语义一致，不用 0 冒充）；
  - `itl_p50_s / itl_p95_s / itl_max_s`：ITL 分位数（线性插值）；
  - `e2e_s = wall_end - wall_start`；`tokens_per_s = n / e2e`（E2E 口径，
    与 `GenerationOutput.tokens_per_s` 语义一致）；
  - `token_times` 全量保留——全局吞吐的滚动窗口统计必须基于 token 级时间戳。
- `MetricsRegistry`：`record(RequestMetrics)` 按 request_id 覆盖记录（幂等，
  防御 `_release` 重复触达）；`snapshot()` 输出 JSON 安全的全局快照：
  请求计数、waiting/running、KV blocks used/total 与 `kv_utilization`、
  `output_tokens_per_s`（滚动窗口，默认 10s）、TTFT/TPOT 均值、
  `gpu_memory_mb`。

### 2.3 时间线层（liteinfer/observability/trace.py）

`build_trace(RequestState) -> RequestTrace` 把打点还原为有序事件：

```text
enqueue -> prefill_start -> prefill_end -> token(0..n-1) -> finished/cancelled
```

每个事件带 `offset_s`（相对 enqueue）。`render()` 输出 ASCII 时间线文本
（Windows GBK 控制台可直接打印），`to_dict()` 供 HTTP 端点返回 JSON。

### 2.4 服务层（liteinfer/server）

| 端点 | 说明 |
|---|---|
| `GET /metrics` | 全局快照 JSON（委托 `EngineCore.metrics_snapshot()`） |
| `GET /v1/requests/{id}/trace` | 单请求时间线 JSON（未知 id → 404；非终态也可查） |
| `GET /health` | 增加 `kv_utilization` 字段 |
| SSE `stream_options.include_usage` | 流末尾追加一帧 `choices: []` + `usage`（OpenAI 协议，Task 09 遗留补齐） |

## 3. 为什么这样设计

1. **打点落在 RequestState 字段上，observability 只做纯还原**。
   备选方案是"通用 tracer"：core 在每个关键位置调用
   `observer.on_event(...)`。否决原因：把观测协议反向耦合进引擎热路径，且
   `EngineCore` 的单写者状态机本来就已天然携带全部关键时刻——从中**派生**
   时间线即可，core 无需感知 trace 的存在。代价是事件粒度固定为 5 类，但
   对延迟归因（排队/prefill/逐 token）已经完备；更细粒度（逐层 forward）
   是 profiler 的职责，不是 serving 指标。
2. **单写者天然无锁**。`record` 只发生在引擎循环线程（Task 09 确立的单写者
   纪律内），`snapshot` 可被 HTTP 线程并发读——dict 读侧在 CPython 下原子，
   快照允许少记一个刚结束的请求，不为观测引入锁拖累热路径。
3. **两条执行路径共享同一套打点**。同步 `EngineCore.run()` 与异步
   `AsyncEngine`（`_dispatch` 零改动）都经过 `_prefill`/`_after_emit`/
   `_release`，不存在"流式路径漏记"。
4. **`_release` 是终态唯一汇聚点**（finish 与 cancel 都走它），指标在此
   一次性汇总。waiting 中即被取消的请求（没碰过缓存）也必须记账，否则
   全局请求数会漏掉失败路径。
5. **滚动窗口吞吐基于 token 级时间戳**。仅有请求级 tokens_per_s 聚合值
   算不出"最近 N 秒输出了多少 token"；分母恒为窗宽而非窗口时长，窗口刚
   启动时数字偏低是吞吐定义的自然结果，且可与稳态值直接比较。
6. **不引入 prometheus_client**。本阶段只交付 JSON 观测口；Prometheus
   文本格式留给 Task 12/13 按采集器需要再加，不为一个端点加依赖。
7. **显存指标遵循补充条款 A3**：`peak_memory_mb()` 无 GPU 返回 None，
   snapshot 同时给出 `gpu_memory_mb: null` 与 `gpu_memory_mb_display:
   "N/A (no GPU)"`，绝不填 0。

## 4. Alternative 方案

| 方案 | 取舍 |
|---|---|
| 通用 observer/hook 回调（vLLM 的 metrics 风格） | 灵活、可插拔，但耦合热路径、需要定义回调协议与生命周期；本项目单进程单写者，状态即事实，纯还原更简单可靠。 |
| Prometheus client（Counter/Histogram） | 拉模式 + 指标类型语义规范，但多一个依赖，且 Histogram 的桶配置对 CPU 演示场景过重；Task 12 出报告时再接。 |
| OpenTelemetry span | 分布式标准，但单机引擎用不上 exporter/采样那套复杂度；`RequestTrace.to_dict()` 的 offset 语义已满足"完整时间线"验收。 |
| 指标在 `AsyncEngine._dispatch` 打点 | 只覆盖流式路径，同步 `run()` 无时间线；且 dispatch 在事件循环线程，与 engine 线程的时钟拼接有跨线程一致性负担。 |

## 5. Known Limitations

1. **调度器 running 集合惰性清理**：请求终态后、下一次 `schedule()` 之前，
   `num_running` 仍计入它（Task 06 既有行为，非本 Task 引入，未擅自修改）。
   观测上看是"刚结束的请求还被算作 running 一拍"。
2. **registry/metrics 按进程生命期累积**：开发/演示场景可接受；生产需要
   LRU/TTL 清理或外部采集器拉走后丢弃（留给 Task 13）。
3. **trace 无跨请求关联**（没有 trace_id 传播），单请求时间线是本阶段验收
   边界；多请求关联属分布式追踪范畴。
4. **ITL 采样受 `AsyncEngine` 投递延迟影响**：`token_times` 记录的是
   **产出**时刻（core 内），不含 SSE 网络投递耗时；用户侧观察到的 ITL
   会略大。这是"引擎指标"与"客户端体验"的口径差异，后者留给 Task 12
   的客户端侧 benchmark。
5. **窗口吞吐是进程内口径**：重启归零，不持久化。
6. `/metrics` 为 JSON，不是 Prometheus 文本格式（见 §3.6）。

## 6. 验收命令（本机 CPU）

```bash
set "HF_HOME=D:\LiteInfer\hf_cache"
set "PYTHONPATH=d:\LiteInfer"

python -m pytest tests/test_metrics.py -q -m "not model"
python -m pytest tests/test_server_metrics.py -q
python -m pytest tests/test_no_hardcoded_cuda.py -q
python -m pytest -q                          # 全量快测不回归
python -m pytest -m model -q tests/test_metrics.py
python examples/metrics_demo.py --max-tokens 16
```

## 7. 环境声明

- 设备/dtype 全部来自 `EngineConfig`（本机 cpu/float32），本模块未引入任何
  设备字面量（`test_no_hardcoded_cuda` 把关）；
- `torch.cuda.*` 仅出现在 `liteinfer/device.py` 既有入口内（有
  `is_available()` 守卫），observability 通过 `peak_memory_mb()` 消费，
  无 GPU 时自然降级为 None。
