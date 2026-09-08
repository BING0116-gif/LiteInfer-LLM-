# Task 06 Design：Continuous Batching Scheduler

> 运行环境：CPU（FP32）。本机验证命令见文末。
> 本 Task 只做**调度决策**（每步本批跑谁、各自 prefill/decode），不改执行形态——
> 仍是逐请求 forward（batch 合并属 Task 08，paged block 属 Task 07）。

---

## 1. 源代码

- `liteinfer/scheduler/config.py`：`SchedulerConfig`（frozen，纯 dataclass，不依赖 torch）
  - `max_num_seqs`：并发序列预算（sequence budget）
  - `max_num_batched_tokens`：单步可调度 token 总量预算（token budget）
- `liteinfer/scheduler/scheduler.py`
  - `Scheduler`：持有 `waiting: deque` + `running: set`，`schedule(snapshot) -> ScheduledBatch`
  - `SchedulerRequestInfo`：只读快照（prompt_len / output_len / status）
  - `ScheduledBatch`：`prefill_ids` / `decode_ids` / `all_ids`
- `liteinfer/scheduler/__init__.py`：导出
- `liteinfer/config.py`：`EngineConfig.scheduler` 字段（集中配置，默认 `SchedulerConfig()`）
- `liteinfer/engine/core.py`：`submit` 入 waiting 队列 + token budget fail-fast；
  `step` 委托 `Scheduler.schedule` 决定本批；`cancel` 从调度器移除；新增 `_snapshot()`
- `liteinfer/__init__.py`：惰性导出 `Scheduler` / `SchedulerConfig`

### 关键接口签名

```python
SchedulerConfig(max_num_seqs: int = 16, max_num_batched_tokens: int = 2048)  # frozen
Scheduler(cfg: SchedulerConfig)
Scheduler.enqueue(request_id)                       # 提交 -> waiting 队尾（FCFS）
Scheduler.schedule(requests: dict[str, SchedulerRequestInfo]) -> ScheduledBatch
Scheduler.remove(request_id)                        # 取消/回收 -> 从两队列移除
Scheduler.num_waiting / .num_running                # 观测属性（供指标/测试）
SchedulerRequestInfo(request_id, prompt_len, output_len, status)
ScheduledBatch(prefill_ids, decode_ids)  # .all_ids = prefill + decode
# EngineCore 新增/变化：
EngineCore.scheduler: Scheduler                     # 从 cfg.scheduler 构造
EngineCore.submit(...) -> rid                       # 入队 waiting，不再立即"在飞"
EngineCore.step() -> list[RequestStepResult]        # 只推进 Scheduler 放行的本批
```

---

## 2. Unit Tests

- `tests/test_scheduler.py`：纯调度单测（无模型），覆盖 FCFS 顺序、seq/token 双预算截断、
  终态释放 slot + FCFS 补位、cancel 移除、小预算不饿死、running 必调度、config 冻结。
- `tests/test_engine_scheduler.py`
  - 快测（FakeLM，无下载）：seq budget 限制并发、16 请求动态加入退出、中途 submit、
    cancel、token budget 过小 fail-fast、默认不限额与 Task 05 行为一致；
  - `marker=model`：Qwen2.5-0.5B 下 Scheduler 产出与 CachedGenerator 逐字一致
    （含并发 + 中途 submit）。

---

## 3. Design Document（关键数据结构）

- **waiting (deque)**：FCFS 队尾入队，队首出队，保证先到先服务。
- **running (set)**：已被准入、正在 prefill/decode 的请求；其 slot 占用计入序列预算。
- **ScheduledBatch**：把本步决策物化为「prefill 集合 + decode 集合」，引擎只消费 `all_ids`，
  不关心调度细节。引擎与调度器之间只通过 `SchedulerRequestInfo` 快照通信，互不持有对方内部状态。

---

## 4. 最小运行示例

`examples/scheduler_demo.py`：用 FakeLM（无权重）提交 12 个请求，序列预算=3，逐 step
打印 `admitted / running / waiting / finished`，直观看到 waiting 随 running 释放而清空。

---

## 5. 为什么这样设计

- **调度与执行解耦**：Scheduler 是纯逻辑、零张量，可脱离模型单测；引擎主循环形状
  （schedule -> execute）在 Task 05 已预留，本 Task 只替换「全部 active 都步进」为
  「Scheduler 准入出的本批」，**引擎主循环不动**。
- **running 必被调度、token budget 只闸门 NEW prefill**：沿用 vLLM 风格——已在飞的 decode
  请求一旦被挂起会造成停顿/语义错乱，故每步必调度；预算主要用于限制「单步能 prefill 多少
  token」，从而把长 prompt 与在飞 decode 解耦。
- **终态即释放 slot**：请求命中 EOS / `max_tokens` 后，下个 step 的 `schedule` 自动把它从
  running 移除，waiting 队首 FCFS 补位——这是 continuous batching 相比静态批「一个走完整批
  干等」的核心收益。
- **token budget 过小 fail-fast**：prompt 长度超过单步 token 上限时直接抛 `ValueError`，
  比静默饿死更易排查（明确是配置/输入问题，而非调度 bug）。

---

## 6. Alternative 方案

- **静态批（一次性固定 batch）**：实现最简单，但短请求要等长请求结束才能释放 slot，
  吞吐浪费明显，不满足「8~16 请求动态加入退出」验收。
- **在 EngineCore 内用列表+计数器硬实现预算**：可行，但把调度逻辑与引擎耦合，违反
  docs/07「Scheduler/Cache/ModelRunner 解耦」，且难以单测。独立 `Scheduler` 类更清晰。
- **抢占式调度（preemption）**：本 Task 不做——当前在飞请求都能在合理步数内结束，抢占
  主要用于显存压力极大的场景；引入会大幅增加复杂度，留待后续（若有真实显存压力再加）。

---

## 7. Current Known Limitations

- **执行仍逐请求 forward**：本 Task 只调度、不合并前向；真正的 batch 矩阵前向属 Task 08
  ModelRunner。因此本机**吞吐尚无实测提升**（CPU 上也不该有，符合 07 文档「CPU 加速比不写
  进简历」约定），调度价值在于「并发准入正确性 + 后续 batch 合并的结构基础」。
- **无 chunked prefill**：单 prompt 长度 > `max_num_batched_tokens` 时直接 fail-fast，
  不支持把超长 prompt 拆成多步 prefill（Task 07+ 视需要再加）。
- **Contiguous KV 缓存随 RequestState 持有，未真正回收**：终态请求释放 slot（调度层面），
  但其 `ContiguousKVCache` 张量要等 `RequestState` 被 GC 才回收；真正的 block 池与逐 block
  回收属 Task 07（BlockPool）。本 Task 不引入内存泄漏，但「slot 释放」与「显存释放」是两个
  层面，前者已验证、后者待 Task 07。
- **无优先级 / 无 SLA 感知**：纯 FCFS，先到先服务；不区分请求优先级。

---

## 8. 进入下一阶段前的验收命令

```bash
set "HF_HOME=D:\LiteInfer\hf_cache"
set PYTHONPATH=d:\LiteInfer

pytest tests/test_scheduler.py -q                          # 纯调度单测全绿
pytest tests/test_engine_scheduler.py -q -m "not model"    # fake 模型 16 请求动态批全绿
pytest tests/test_no_hardcoded_cuda.py -q                  # 无裸 cuda 字面量
pytest -m model -q                                         # 真模型：scheduler 下与 CachedGenerator 逐字一致
pytest tests/test_engine.py -q                             # Task 05 不回归
pytest -q                                                 # 全快测不回归
python examples/scheduler_demo.py                          # 展示 seq budget 限制下的逐步准入
```

---

## 9. Alternative 与 Limitations 之外：与上下游的接口契约

- **向上（Task 09 Async Engine）**：`step()` 返回 `list[RequestStepResult]`，调度器决策
  被封装在 `step` 内部，流式 API 无需感知调度细节。
- **向下（Task 07 BlockPool）**：每个 `RequestState.cache` 仍是 `ContiguousKVCache`；Task 07
  仅把该字段换成 paged block，`step` 循环与调度逻辑**完全不动**。
- **向下（Task 08 ModelRunner）**：调度器产出 `ScheduledBatch`，未来 `ModelRunner.execute(batch)`
  将其合并为一次 batch 前向；本 Task 的 `all_ids` 已是「本步该跑哪些请求」的明确边界。
