# Task 05：Request + Engine Core 设计文档

> 运行环境：CPU（本机 Intel 核显，无 CUDA）
> 本机验证命令见文末「验收命令」一节。
> 禁止硬编码设备：全仓库 grep 不到裸 `"cuda"`（见 `tests/test_no_hardcoded_cuda.py`）。

---

## 1. 源代码

新增模块 `liteinfer/engine/`：

- `request.py`：`Request`（用户态数据模型）、`RequestStatus`（状态枚举）、
  `RequestState`（引擎内部可变态）、`RequestRegistry`（注册表）、
  `RequestOutput` / `RequestStepResult`（对外结果）。
- `core.py`：`EngineCore`（多请求总控层），含 `submit / step / run / cancel / get_request`。
- `__init__.py`：导出上述类型。

`liteinfer/__init__.py` 增加 `EngineCore / Request / RequestStatus / RequestRegistry` 惰性导出；
`liteinfer/model/loader.py` 增加 tokenizer 加载的 fast→slow 回退（修复本机 transformers
5.14.1 + tokenizers 0.22.2 的 fast tokenizer 构建 bug，详见 Known Limitations）。

关键接口签名：

```python
# 请求模型（用户态，无张量/缓存句柄）
Request(request_id: str, prompt: str, params: SamplingParams,
        status: RequestStatus = WAITING, prompt_tokens: int = 0,
        generated: list[int] = [], finish_reason: str | None = None, output_text: str = "")
RequestStatus  # WAITING / PREFILL / DECODE / FINISHED / CANCELLED（str 枚举）

# 引擎
EngineCore(model, tokenizer, cfg, eos_ids=None)
EngineCore.from_config(cfg: EngineConfig) -> EngineCore
EngineCore.submit(prompt: str, params: SamplingParams | None = None) -> str          # 返回 request_id
EngineCore.step() -> list[RequestStepResult]                                          # 所有在飞请求各推进一个 token
EngineCore.run(max_steps: int | None = None) -> dict[str, RequestOutput]             # 跑到全部结束
EngineCore.cancel(request_id: str) -> None
EngineCore.get_request(request_id: str) -> Request
EngineCore.active_requests() -> list[Request]
```

---

## 2. Unit Tests

`tests/test_engine.py`：

- **快测（默认 `pytest -q`，无模型，12 passed）**
  - `TestRequestModel`：Request/RequestStatus/RequestRegistry 纯逻辑（默认值、终态判定、增删查、
    `active()` 排除终态、`count_by_status`）。
  - `TestEngineFakeModel`：用「假模型」驱动状态机，验证单/多请求、EOS/length 终止、中途 submit、
    cancel 的并发维护正确性。假模型的 next-token 是输入 token 的纯函数 **(cur+1)%vocab**，保证
    多请求互不干扰、与顺序无关。
- **真模型测试（marker=model，2 passed）**
  - `TestEngineParityWithCachedGenerator`：单请求 `EngineCore` 输出与 `CachedGenerator`
    **逐字一致**（text / finish_reason / output_tokens / cached_tokens）；2 个并发请求各自与各自
    `CachedGenerator` 输出一致。

---

## 3. 最小运行示例

`examples/engine_demo.py`：submit 3 个并发 prompt → `run()` → 每个请求输出与 `CachedGenerator`
逐字一致。运行：

```bash
set "HF_HOME=D:\LiteInfer\hf_cache"
set PYTHONPATH=d:\LiteInfer
python examples/engine_demo.py
```

---

## 4. 关键数据结构解释

- **`Request` vs `RequestState`**：`Request` 是用户能看到的全部状态（id、prompt、参数、状态、
  已生成 token、终止原因、文本），**不含任何张量/缓存句柄**——避免 Task 09 流式序列化时把
  CPU/GPU 张量泄露出去。`RequestState` 是引擎内部可变态，持有 `Request` 外加「推进生成所需的全部
  可变资源」：专属 `ContiguousKVCache`、prompt 张量、`cached_len`（已写入缓存的 token 数）、
  `next_id`（下一步要 forward 的 token）、可选 `torch.Generator`。
- **`RequestRegistry`**：`id -> Request` 的集中管理，只存用户态 `Request`，不存 `RequestState`
  （后者由 `EngineCore._states` 另表持有），保证外部查询安全。
- **`RequestStatus`**：`WAITING → PREFILL → DECODE → FINISHED / CANCELLED`，与 `docs/02 §2`
  生命周期一一对应；`PREFILL`/`DECODE` 区分是为 Task 06 调度器分别计入 TTFT/TPOT 预算预留。

---

## 5. 为什么这样设计

- **状态机与数据模型分离**：`Request` 无张量，状态演进只发生在 `EngineCore._advance`，便于单测
  与未来接入流式 API。
- **每请求专属 KV 缓存**：Task 04 已把缓存生命周期外部化，本 Task 让每个 `RequestState` 持有
  自己的 `ContiguousKVCache`（容量 `prompt_len + max_tokens`）。这正是 Task 07「分页 block」的
  替换点——届时只改 `RequestState.cache` 的构造，step 循环不动。
- **引擎复用 `MinimalQwen` 的 forward 契约**：模型只返回 logits、调用方持有缓存。step 循环对
  WAITING 请求做 prefill（产出首 token），对 DECODE 请求做「写缓存 + 采样」，**与此前
  `CachedGenerator` 的单请求循环逐位等价**——所以单请求输出能逐字对齐，验证了「引擎只是把单请求
  循环升级成了多请求状态机，语义没变」。
- **设备/dtype 一律从 `EngineConfig` 进入**：`EngineCore` 用 `cfg.device` / `resolve_dtype`，
  不依赖模型参数当前所在设备，避免「模型忘了 `.to(device)`」被静默吞掉（补充条款 A1/A2）。

---

## 6. Alternative 方案

- **方案 A：真·batch 合并前向（一次喂多个序列）**。本 Task 没做：当前 `MinimalQwen` 的 attention
  `LayerKVCache` 只支持 `batch=1`（越界即报错），真 batch 要改模型与缓存布局，属 Task 08
  `ModelRunner`。Task 05 用「时间片交错」推进所有在飞请求，**靠状态机正确性的验证**，不靠算力合并。
- **方案 B：让 `CachedGenerator` 直接吃一个请求列表**。否决：会把这个 Task 的「请求生命周期管理」
  塞回生成器，违背 docs/07 的「Scheduler/Cache/ModelRunner 解耦」。引擎应是独立总控层。
- **方案 C：把缓存放在注册表里随 Request 一起存**。否决：注册表只暴露用户态，缓存是内部资源，
  混在一起会泄露到外部查询。

---

## 7. 当前 Known Limitations

- **时间片交错 ≠ 真 batching**：每个 step 对活跃请求逐个 forward，CPU 上吞吐没合并收益；这是
  Task 08 的目标。
- **无调度**：本 Task 对所有已 submit 请求「全部同时推进」，没有 waiting/running 队列、FCFS、
  token/sequence budget——这些属 Task 06。因此「可同时维护多个请求」成立，但还做不到「按预算
  准入 / 抢占」。
- **tokenizer fast→slow 回退（环境补丁）**：本机 `transformers 5.14.1 + tokenizers 0.22.2`
  的 fast tokenizer 后端构建失败（`Couldn't instantiate the backend tokenizer`）。`loader.py`
  改为「先试 fast、失败回退 `use_fast=False`（tiktoken 后端）」。CPU 上两种 tokenizer 编码结果
  一致，不影响正确性，仅 tokenizer 速度略慢。正常环境仍优先用 fast。**注意**：本机 `HF_HOME`
  若带尾随空格会让缓存路径错位、回退后找不到本地文件而去网络下载，设置时务必
  `set "HF_HOME=D:\LiteInfer\hf_cache"`（带引号）。
- **`run()` 同步阻塞**：当前 `run()` 是同步循环，Task 09 的 asyncio/流式会在此基础上包一层队列。

---

## 8. 进入下一阶段前的验收命令

```bash
# 1) 快测（无模型，应 12 passed）
pytest tests/test_engine.py -q -m "not model"

# 2) 真实模型测试（与 CachedGenerator 逐字一致 + 多请求，应 2 passed）
set "HF_HOME=D:\LiteInfer\hf_cache"
pytest -m model -q

# 3) 防回归 + 无裸 cuda（全快测 86 passed；cuda 单测 1 passed）
pytest -q
pytest tests/test_no_hardcoded_cuda.py -q

# 4) 最小示例（3 个并发请求全部与 CachedGenerator 逐字一致）
set PYTHONPATH=d:\LiteInfer
python examples/engine_demo.py
```

---

## 9. 本阶段踩过的坑（环境向）

- 本机 `sentencepiece` 缺失会导致 Qwen tokenizer 加载失败；已 `pip install sentencepiece`
  （Qwen2.5 tokenizer 的后端依赖）。但这只解决了 slow 路径依赖，**fast 后端本身仍被
  transformers 5.14.1 + tokenizers 0.22.2 的 bug 卡死**，故 `loader.py` 加 fast→slow 回退。
- `set HF_HOME=D:\LiteInfer\hf_cache` 若不带引号，cmd 可能把尾随空格带进变量，
  导致缓存目录变成 `hf_cache \hub`，本地 tokenizer/权重找不到——务必加引号。
