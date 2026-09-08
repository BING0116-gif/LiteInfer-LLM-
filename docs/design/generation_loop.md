# Task 02 Design Document — Manual Generation Loop + Sampling

## 运行环境

```text
运行环境：CPU（本阶段 01-11 全部 CPU 可跑通）
本机验证命令：
    pytest tests/test_sampler.py -q            # 纯单测，秒级
    pytest tests/test_no_hardcoded_cuda.py -q  # 设备硬编码把关
    pytest -m model -q                         # 端到端对齐（需要 hf_cache）
    python examples/generation_demo.py --mode greedy
    python examples/generation_demo.py --mode sample --temperature 0.7 --top-p 0.9 --seed 42
```

## 1. 架构与数据流

```text
prompt
  ↓ tokenizer
input_ids [1, seq]
  ↓ ┌──────────── 循环 max_tokens 次 ────────────┐
  ↓ │ model(input_ids) → logits [1, seq, vocab] │   ← 无 KV Cache，全序列 forward
  ↓ │ logits[0, -1] → Sampler.sample(...)       │   ← greedy / T / top-k / top-p
  ↓ │ EOS? → break (finish_reason="eos")        │
  ↓ │ append token → 拼回 input_ids             │
  ↓ └───────────────────────────────────────────┘
  ↓ decode(skip_special_tokens=True)
GenerationOutput(text, finish_reason, ...)
```

模块职责：

| 模块 | 职责 | 依赖 |
|---|---|---|
| `liteinfer/sampling/params.py` | 采样参数 + fail-fast 校验 | 无（连 torch 都不依赖） |
| `liteinfer/sampling/sampler.py` | logits → token 的纯函数 | torch, params |
| `liteinfer/model/generator.py` | forward 循环、EOS、append | loader, sampling |

## 2. 关键数据结构

### SamplingParams（frozen dataclass）

```python
SamplingParams(
    max_tokens: int,          # 必须 >= 1
    temperature: float = 1.0, # 0.0 == greedy（vLLM/OpenAI 通行语义）
    top_k: int = -1,          # -1 关闭
    top_p: float = 1.0,       # 1.0 关闭
    seed: Optional[int] = None,
)
```

构造期即校验（`__post_init__`），非法值在进入生成循环前就抛 `ValueError`。

### Sampler

```python
Sampler().sample(logits: Tensor[vocab], params, generator=None) -> int
```

流水线：`temperature 缩放 → top-k 过滤 → top-p 过滤 → softmax → multinomial`。
顺序与 HF 一致：过滤语义定义在"temperature 缩放后的分布"上，先缩放后过滤
才能保证过滤集合与采样分布一致。

top-p 的实现取 `cum - probs >= top_p` 判淘汰（前置累计概率已达标者出局），
天然保留恰好跨越阈值的 token，不需要 HF 实现里右移一位的修正写法。

随机性通过外部 `torch.Generator` 注入，Sampler 无状态 —— seed 是"请求级"
属性，并发服务下多个请求各带 seed，有状态采样器会互相污染随机序列。

### GenerationOutput

与 Task 01 `baseline.GenerationOutput` 字段对齐，新增 `finish_reason`
（`"eos"` / `"length"`），为 Task 05 请求状态机预留语义信号。

### ManualGenerator

```python
ManualGenerator(model, tokenizer, cfg)      # 依赖注入
ManualGenerator.from_config(cfg)            # 便捷构造
.generate(prompt, params=None) -> GenerationOutput  # params=None → greedy + cfg.max_new_tokens
```

## 3. 为什么这样设计

1. **Sampler 与模型彻底解耦**：它只吃 1D logits。Task 03 换自研 Decoder、
   Task 04 换 KV Cache 时采样层零改动；同时"纯函数"让它能用手工 logits
   做秒级数学性质单测，不必每次都拉起 0.5B 模型。
2. **EOS 在源头不进序列**：HF 的做法是 append EOS 后靠
   `skip_special_tokens` 在解码时剔除。我们在源头就不 append，语义更干净；
   文本结果与 HF 完全一致（对齐测试验证）。
3. **EOS 取 `generation_config.eos_token_id` 优先**：Qwen2.5 的终止符是
   `<|im_end|>`（151645），记录在 generation_config 而非 tokenizer；
   兼容 int / list 两种形态。漏掉这一点 greedy 对齐会错位。
4. **greedy 判据是 `temperature == 0.0`**：走 argmax 快路径，不建
   Generator、不做 softmax —— greedy 是生产默认路径，值得单独短路。
5. **共享权重对齐测试**：`HFBaseline` 与 `ManualGenerator` 注入同一份
   `LoadedModel`，排除"两份权重加载差异"的干扰，差异只能来自生成逻辑。
6. **每步全序列 forward（无 KV Cache）**：docs/03 模块 2 明确"先不考虑
   性能，只建立可控推理主链"。这是本阶段的刻意选择，不是遗漏。

## 4. Alternative 方案

| 决策点 | 采用 | 备选 | 不采用的原因 |
|---|---|---|---|
| top-p 淘汰判据 | `cum - probs >= p` | HF 的 `cumprobs > p` + 右移修正 | 语义等价但前者无 off-by-one 风险，可读性更好 |
| 随机源归属 | 请求级 Generator 注入 | Sampler 内部持全局 RNG | 全局 RNG 在并发/复现场景互相污染 |
| EOS 处理 | 源头不 append | append 后解码剔除 | 源头处理让 output_tokens 直接等于真实生成数 |
| 循环载体 | 逐步 `torch.cat` | 预分配 buffer + 原地写 | cat 在 O(n^2) 循环里开销可忽略，预分配是 Task 04 KV Cache 要解决的问题，现在做是过早优化 |
| temperature=0 | argmax 短路 | T=极小值走采样路径 | 数值上 T→0 会让 softmax 接近 one-hot，浮点误差下 multinomial 仍可能偏离 argmax；语义上 greedy 就不该进随机路径 |

## 5. Known Limitations

1. **无 KV Cache**：每步重算全前缀，O(n^2)。生成 32 token 的耗时随长度
   明显增长。Task 04 用 contiguous KV Cache 替换 forward 策略。
2. **单请求串行**：一次只生成一个序列，无 batching。Task 05/06 引入
   Engine Core 与 Continuous Batching。
3. **CPU FP32**：本机约定（补充条款 A2），CPU tokens/s 数据仅用于逻辑
   验证，禁止写进任何性能结论。
4. **top_p 接近 1.0 的浮点边界**：`top_p=1.0` 显式跳过过滤；但 `0.999`
   这类值在 float32 累加误差下，极端小概率的尾部 token 可能被误删。
   生产语义下影响可忽略，记录在案。
5. **`finish_reason` 粒度**：只有 eos/length，无 stop-string 中断（属于
   Task 05+ 的请求参数范畴）。

## 6. 进入下一阶段的验收命令

```bash
pytest tests/test_sampler.py -q
pytest -m model -q
python examples/generation_demo.py --mode greedy
```

通过标准：greedy 文本与 HF baseline 逐字一致；采样固定 seed 可复现；
全仓库无裸 `"cuda"` 字面量（`test_no_hardcoded_cuda.py` 守门）。
