# Task 03 Design Document：Minimal Qwen Decoder

## 0. 运行环境（补充条款 A6）

```text
运行环境：CPU（FP32 对齐）；代码不依赖 GPU，上云仅需改 EngineConfig.device/dtype
本机验证命令：
  pytest tests/test_minimal_operators.py -q
  pytest -m model tests/test_minimal_alignment.py -q
  set PYTHONPATH=d:\LiteInfer && python examples/decoder_demo.py
```

## 1. 目标与范围

用纯 PyTorch 从零实现 Qwen2 的 Decoder-only 前向计算图，替代"把 HF 模型当黑盒
forward"的 Task 02 状态。本阶段**只做 forward**：

- 做：embedding、RMSNorm、RoPE、GQA attention（含 QK-Norm）、SwiGLU MLP、
  decoder layer、lm_head、causal mask、HF 权重映射加载；
- 不做：KV Cache（Task 04）、采样（沿用 Task 02 Sampler）、Engine/Scheduler。

## 2. 关键数据结构

```text
MinimalQwenForCausalLM
├── model: MinimalQwenModel
│   ├── embed_tokens: Embedding(vocab, hidden)          # [B,S] -> [B,S,H]
│   ├── layers: N × QwenDecoderLayer                     # 0.5B: 24 层
│   │   ├── self_attn: QwenSelfAttention
│   │   │   ├── q/k/v_proj: Linear(hidden -> H*D / KVH*D, bias=attention_bias)
│   │   │   ├── [q_norm/k_norm: RMSNorm(head_dim)]       # QK-Norm，权重存在才建
│   │   │   ├── rotary_emb: RotaryEmbedding(head_dim, theta=1e6)
│   │   │   └── o_proj: Linear(H*D -> hidden, bias=False)
│   │   ├── mlp: QwenMLP(gate/up/down, SwiGLU)
│   │   └── input/post_attention_layernorm: RMSNorm(hidden)
│   └── norm: RMSNorm(hidden)                            # final norm
└── lm_head: Linear(hidden -> vocab, bias=False)         # 0.5B tied 到 embedding
```

中间张量形状（attention 内部）：

```text
q: [B, S, H, D] -> transpose -> [B, H, S, D]
k/v: [B, S, KVH, D] -> QK-Norm/RoPE -> repeat_kv -> [B, H, S, D]
scores/probs: [B, H, S, S]
```

## 3. 为什么这样设计

1. **模块命名镜像 HF state_dict**：`model.layers.{i}.self_attn.q_proj` 等键名与
   HF 完全一致，权重重映射只剩"剥离 `model.` 前缀"一条规则，
   `load_state_dict(strict=True)` 保证 checkpoint 每个权重都被消费——
   悄悄漏加载是数值对齐最难排查的失败模式，用机制而非人肉消灭它。
2. **所有超参从 HFConfig 读取**：hidden/head_dim/eps/theta/intermediate_size/
   bias 开关全部 `getattr` 兜底读取，不写死 0.5B 的数字；遇到未支持的
   `rope_scaling` 显式 NotImplementedError 而不是静默算错。
3. **数值路径逐处镜像 HF eager 实现**：RMSNorm 的 fp32 升精度、softmax 的
   fp32 计算后 cast、rotate_half 与 `cat(freqs, freqs)` 的 cos/sin 布局——
   对齐要求"数值路径一致"而非"数学等价"，FP32 下这两者差异是 0，
   到 FP16（云端）时差异会是 1e-2 量级，路径一致能显著压低噪声。
4. **RoPE 现算不缓存**：Task 03 目标是正确性；cos/sin 预计算与位置增量
   追加属于 KV Cache 阶段的优化，提前引入只会增加对齐噪声。
5. **additive mask 用 finfo.min 而非 -inf**：全 -inf 行 softmax 会出 NaN，
   finfo.min 经 softmax 后约为 0，与 HF 行为一致且稳定。
6. **QK-Norm 由权重决定，不由 config 决定**：本机实测 transformers 5.14.1 的
   Qwen2 实现已无 q_norm/k_norm 模块，Qwen2.5-0.5B 的 safetensors 原始键
   （290 个）里也没有这两组权重（QK-Norm 是 Qwen3 系才启用）。因此
   `use_qk_norm` 从 checkpoint 键存在性推断，strict 加载语义清晰。

## 4. Alternative 方案

| 方案 | 取舍 |
|---|---|
| 直接读 HF config json 手写超参 | 否——config 字段随 transformers 版本漂移，经 `PretrainedConfig` 对象读取最稳 |
| 复用 HF 的建模代码只换 forward | 否——那不是"自主实现核心执行链"，违反 docs/07 禁止事项精神 |
| `nn.functional.scaled_dot_product_attention` | 可用但放弃——sdpa 内核是黑盒，本阶段目标是打通每一步算子；Task 04 做 KV Cache 时再评估 |
| 权重用 copy_ 逐模块搬运 | 否——strict state_dict 一次加载，机制上保证无遗漏 |
| F.multi_head_attention_forward | 否——其内部布局与 Qwen（QK-Norm、GQA 布局）不完全匹配，且同样是黑盒 |

## 5. Known Limitations

1. 无 KV Cache：每步全序列 forward，O(n²)，性能数据无意义（Task 04 解决）。
2. 不支持 `rope_scaling` 变体（0.5B 为标准 RoPE，不受影响）。
3. 不支持 tied embedding 之外的 Qwen 变体细节；batch>1 未在真实模型上
   验证（本阶段对齐用单序列，batch 路径在 Task 06 批处理时再验）。
4. attention 用显式 matmul+softmax，显存 O(S²)（sdpa 可在后续 Task 直接替换，
   接口已按 additive mask 设计）。
5. QK-Norm 路径已实现但本机 checkpoint 无法验证（Qwen2.5-0.5B 无此权重）；
   未来对齐 Qwen3 checkpoint 时该路径才能实测。

## 6. 验收标准与实测

- 算子单测（无模型）：`pytest tests/test_minimal_operators.py -q`；
- 对齐测试（真实权重，CPU FP32，atol=rtol=1e-4）：
  `pytest -m model tests/test_minimal_alignment.py -q`；
- top-1 agreement = 100%（FP32 下）；
- `pytest -m model -q` 全量不回归（Task 01/02 测试不受影响）；
- 环境纪律：`pytest tests/test_no_hardcoded_cuda.py -q` 全绿，
  本模块零 `torch.cuda.*` 调用。
