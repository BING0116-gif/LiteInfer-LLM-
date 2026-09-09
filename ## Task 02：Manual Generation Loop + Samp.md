## Task 02：Manual Generation Loop + Sampling（2026-09-08）

**新增文件**

text
liteinfer/sampling/params.py          # SamplingParams（frozen + 构造期校验）
liteinfer/sampling/sampler.py         # Sampler：T→top-k→top-p→softmax→multinomial
liteinfer/model/generator.py          # ManualGenerator 手写循环（生产路径）
tests/test_sampler.py                 # 19 条纯 logits 单测（0.09s）
tests/test_manual_generation.py       # 7 条端到端（marker=model）
examples/generation_demo.py           # greedy/sample 两模式 demo
docs/design/generation_loop.md        # Task 02 设计文档

**修改文件**

text
liteinfer/__init__.py                 # 惰性导出 SamplingParams/Sampler/ManualGenerator

**关键接口签名**

python
SamplingParams(max_tokens, temperature=1.0, top_k=-1, top_p=1.0, seed=None)  # T=0 即 greedy
Sampler().sample(logits: Tensor[vocab], params, generator=None) -> int
ManualGenerator(model, tokenizer, cfg) / .from_config(cfg)
ManualGenerator.generate(prompt, params=None) -> GenerationOutput  # params=None → greedy
GenerationOutput(text, prompt_tokens, output_tokens, finish_reason, latency_s, tokens_per_s, device, dtype)

**验收命令**（CPU 全部跑通）

bash
pytest tests/test_sampler.py -q   # 19 passed
pytest -m model -q                # 11 passed（greedy 与 HF baseline 逐字一致）
python examples/generation_demo.py --mode greedy

**踩过的坑**

- Qwen2.5 的 EOS 是 generation_config.eos_token_id（<|im_end|>=151645），
    不是 tokenizer.eos_token_id，对齐 HF 时必须优先取前者（且可能是 list）
- 采样统计类测试（"所有候选都应出现"）必须用概率相近的 logits 构造，
    概率悬殊时尾部 token 有限次采样抽不到属正常，会误报失败
- examples 脚本直接 `python xxx.py` 找不到 liteinfer 包，需 `set PYTHONPATH=d:\LiteInfer`
- transformers 5.x 实际装的是 5.14.1，loader 的 dtype/torch_dtype 双参数
    兼容逻辑已覆盖，无感

**身份转变声明**：HFBaseline（Task 01）自 Task 02 起降级为对齐参照物，
退出生产路径；后续生产推理一律走 ManualGenerator（Task 04 起由
ModelRunner 接管 forward 策略）。
