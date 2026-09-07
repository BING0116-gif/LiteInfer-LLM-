# Task 01 设计文档：项目骨架 + HF Baseline

> 运行环境：**CPU**（本机无 NVIDIA GPU）
> 本机验证命令：
> ```bash
> # 快速测试（秒级，不需要模型）
> .venv/Scripts/python.exe -m pytest -q tests/test_config.py tests/test_device.py tests/test_no_hardcoded_cuda.py
> # 端到端验收（需要 hf_cache 中已有 Qwen2.5-0.5B）
> .venv/Scripts/python.exe -m pytest -q -m model tests/test_baseline.py
> .venv/Scripts/python.exe examples/baseline_demo.py --prompt "你好" --max-new-tokens 32
> ```

## 1. 本阶段目标

建立后续 13 个 Task 依赖的三块地基：

1. **配置中心**：`EngineConfig` 是 device / dtype / 模型 / 生成参数的唯一入口；
2. **设备与 dtype 抽象**：全仓库不出现裸设备字面量，`torch.cuda.*` 全部可降级；
3. **可测试的包结构**：src 布局 + editable install + pytest 分层（无模型的快速测试 / 需要模型的 `model` 标记测试）。

同时交付 HF Baseline：Qwen2.5-0.5B 经 `model.generate()` 正常生成，
作为 Task 02 手写生成循环的对齐参照。

**明确不做**（防止范围蔓延）：手写 forward、Sampler、KV Cache、Request、Scheduler、API。

## 2. 关键数据结构

### EngineConfig（frozen dataclass）

```python
EngineConfig(
    model_id: str = "Qwen/Qwen2.5-0.5B",
    device: str = "cpu",           # 云端只改这一处
    dtype: torch.dtype = torch.float32,
    hf_cache_dir: Optional[Path] = None,
    max_new_tokens: int = 64,
    seed: int = 42,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
)
```

- `frozen=True`：配置对象在引擎各模块间传递，禁止运行期被某处悄悄改掉
  —— Task 06 的 Scheduler 若发现配置被中途篡改，行为将不可复现。
- `dtype` 存 `torch.dtype` 而非字符串：避免每个使用方各自 parse，出现多份真相。
- `hf_cache_dir=None` 时走兜底链：`HF_HOME` 环境变量 > `D:/LiteInfer/hf_cache` > 仓库内 `hf_cache/`。
  兜底的原因：环境变量可能没设成，而 huggingface_hub 默认写 C 盘用户目录（C 盘仅剩约 10GB）。

### GenerationOutput

```python
GenerationOutput(text, prompt_tokens, output_tokens, latency_s, tokens_per_s, device, dtype)
```

latency 字段从第一天就进数据结构而不是 print：Task 10 的 Metrics 会复用同一形状。

### LoadedModel

`load_model_and_tokenizer()` 的返回值，携带**实际生效**的 device/dtype
（经过 CPU 守卫后的），与配置里的"声明值"区分开。

## 3. 为什么这样设计

### 3.1 设备抽象只依赖 `dev.type == "cpu"` 分支 + `torch.cuda.*` API

补充条款 A1 禁止硬编码设备。实现上让所有"是否在 GPU"的判断只写
`if dev.type == "cpu": ... else: ...`，配合 `torch.cuda.is_available()`，
全仓库**不需要**出现任何带引号的设备字符串——于是 A1 的验收
（grep 不到裸 `"cuda"` 字面量）可以做成一个常态化单测，
而不是靠 code review 人肉把关。

### 3.2 dtype 守卫放在加载前而不是配置层

云端代码从同一仓库 clone，环境变量可能残留 GPU 的 dtype 设置。
`resolve_dtype()` 在 `model.to(device)` 之前执行，
保证"CPU 上一律 float32"（补充条款 A2）永远不会被绕过，
同时留 warning 日志说明为什么配置被修正。

### 3.3 transformers 新旧版本的 dtype 参数兼容

transformers 4.56 起 `torch_dtype` 改名 `dtype`。用 try/except TypeError
探测而不是 inspect 签名：签名里的参数可能藏在 `**kwargs` 后面，
运行时探测比反射可靠。

### 3.4 测试分层与"自指陷阱"

- pyproject 里 `addopts = -m "not model"`：`pytest -q` 永远秒级反馈，
  需要模型的测试显式 `pytest -m model` 触发——保证"上一阶段测试不过
  不进入下一阶段"的检查成本低到不会被跳过。
- `test_no_hardcoded_cuda.py` 自身的匹配 pattern 用字符串拼接构造：
  如果直接写 `'"cuda"'`，这个测试文件就是全仓库唯一的违规者（自指陷阱）。

## 4. Alternative 方案

| 决策点 | 采用 | 备选 | 不采用的原因 |
|---|---|---|---|
| 设备判断 | `dev.type == "cpu"` 分支 | 定义 `_IS_CUDA = torch.cuda.is_available()` 全局常量 | 常量在 import 时冻结，测试里无法模拟"有卡/无卡"两种世界 |
| 配置方式 | frozen dataclass | pydantic Settings | 引入额外依赖；当前字段少，dataclass 足够，云/本地差异用环境变量覆盖 |
| 设备放置 | `model.to(device)` | `device_map="auto"`（accelerate） | 引入 accelerate 依赖且行为不透明；0.5B 单卡场景 `.to()` 完全够用 |
| dtype 兼容 | try/except TypeError | 读 `transformers.__version__` 分支 | 版本号比较脆弱（fork/nightly 版本串不可信） |
| 缓存兜底 | 三级兜底链 | 只依赖 HF_HOME | 环境变量未设时静默写爆 C 盘 |

## 5. Known Limitations

1. `HFBaseline.generate()` 是全项目**唯一**的 `model.generate()` 调用点，
   Task 02 后降级为对齐参照，不再是生产路径。
2. 只实现 greedy；temperature/top-k/top-p 属于 Task 02。
3. CPU 上的 latency/tokens_per_s **仅用于逻辑验证**，不代表 GPU serving 性能，
   不得写入 README 性能表或简历（docs/04 指标降级规则）。
4. `peak_memory_mb()` 无 GPU 时返回 `None`（渲染为 `N/A (no GPU)`），
   进程 RSS（`process_rss_mb`）只是粗粒度替代观测。
5. 未做批量生成（batch>1）；pad_token 的处理（无 pad 用 eos 兜底）在
   batch>1 时需要重新审视，留到 Task 06 连续批处理阶段。
6. `get_device()` 只做解析不做可用性检查（解析 ≠ 硬件就绪），
   硬件错误在 `model.to(device)` 时由 torch 抛出。

## 6. 踩坑记录

1. **Git Bash 向 Windows Python 传路径的坑**：`python -m venv /d/LiteInfer/.venv`
   在 MSYS bash 里 `/d/...` 不会被转成 `D:\...`（因为它是参数不是命令），
   结果 venv 被建到了 `C:\d\LiteInfer\.venv`。传给 Windows 原生 exe 的路径
   一律写 `D:/LiteInfer/.venv` 风格。
2. 首次 `venv` 创建中断后 `.venv` 内缺 pip，重建时用 `--clear` + 显式
   `ensurepip` 兜底。
3. **HF 缓存 symlink 坑（重要）**：Windows 无开发者模式时 symlink 静默失败，
   `hf_cache/hub/.../snapshots/` 下所有文件变成 0 字节空壳——blob 完整
   （988MB 权重都在）但 `config.json` 读不到，报
   `OSError: not a valid JSON file`。修法：删 snapshots 层（保留 blobs）+
   `HF_HUB_DISABLE_SYMLINKS=1` 重建，2 秒完成（blob 复用，不重下）。
   已在 `liteinfer/config.py` 里 setdefault 该环境变量。
4. **cache_dir 布局坑**：把 `hf_cache` 根目录直接传给
   `from_pretrained(cache_dir=...)` 会在根下另建一份 `models--*`（与
   `HF_HOME` 布局的 `hub/` 子目录不同），同一模型存两份各约 1GB。
   loader 统一传 `<resolved>/hub`。
