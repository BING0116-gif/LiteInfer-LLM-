# LiteInfer 最终项目 Checklist

## 0. 环境约束（先看这条）
- [ ] 全仓库 grep 不到裸 `"cuda"` 字面量，设备统一走 `EngineConfig.device`
- [ ] `EngineConfig.dtype` 可配置（本机 CPU = FP32，云端 GPU = FP16）
- [ ] `torch.cuda.*` 调用均有 CPU fallback，无 GPU 时不崩
- [ ] 显存类指标在无 GPU 时输出 `N/A (no GPU)`，不填 0
- [ ] A–G、I–J、L–M 各项均已在本机 CPU 上验证通过
- [ ] H 中的 GPU Memory / KV Utilization 与 K 整节需在云端 GPU 完成

## A. 推理正确性
- [ ] HF Baseline 可运行
- [ ] Manual Generation 可运行
- [ ] 最终不依赖 model.generate()
- [ ] Greedy 与 HF 输出对齐
- [ ] Minimal Qwen Forward 对齐
- [ ] KV Cache 输出正确

## B. 模型模块
- [ ] RMSNorm
- [ ] RoPE
- [ ] GQA
- [ ] SwiGLU
- [ ] Decoder Layer
- [ ] LM Head

## C. Engine
- [ ] Request
- [ ] Request Registry
- [ ] Engine Loop
- [ ] Request lifecycle
- [ ] FINISHED 回收
- [ ] CANCELLED 回收

## D. Scheduler
- [ ] FCFS
- [ ] waiting queue
- [ ] running queue
- [ ] max_num_seqs
- [ ] token budget
- [ ] admission control
- [ ] Continuous Batching

## E. KV Cache
- [ ] contiguous KV
- [ ] BlockPool
- [ ] FreeQueue
- [ ] BlockTable
- [ ] allocate
- [ ] append
- [ ] free
- [ ] get_usage
- [ ] stress test
- [ ] no memory leak

## F. Paged KV
- [ ] logical-to-physical mapping
- [ ] block size configurable
- [ ] gather-based correctness
- [ ] README 明确不是完整 CUDA PagedAttention

## G. API
- [ ] FastAPI
- [ ] /v1/completions
- [ ] /v1/chat/completions
- [ ] OpenAI SDK compatible
- [ ] stream=false
- [ ] stream=true
- [ ] SSE
- [ ] cancellation

## H. Metrics
- [ ] Queue Time
- [ ] TTFT
- [ ] TPOT
- [ ] ITL
- [ ] E2E
- [ ] Requests/s
- [ ] Output Tokens/s
- [ ] P50
- [ ] P95
- [ ] GPU Memory
- [ ] KV Utilization

## I. Prefix Cache
- [ ] block hash
- [ ] cache hit
- [ ] ref_count
- [ ] full-block only
- [ ] shared-prefix benchmark

## J. Tests
- [ ] unit tests
- [ ] integration tests
- [ ] stress tests
- [ ] cancellation test
- [ ] concurrent request test
- [ ] correctness comparison

## K. Benchmark
- [ ] HF Sequential
- [ ] Manual no-cache
- [ ] KV
- [ ] Continuous Batching
- [ ] Paged KV
- [ ] Prefix Cache
- [ ] concurrency 1/2/4/8/16/32
- [ ] decode-heavy workload
- [ ] prefill-heavy workload
- [ ] chat workload
- [ ] shared-prefix workload
- [ ] charts
- [ ] raw CSV/JSON

## L. 工程
- [ ] pyproject.toml
- [ ] Dockerfile
- [ ] CI
- [ ] type hints
- [ ] docstrings
- [ ] config management
- [ ] logging
- [ ] no vLLM runtime dependency

## M. 文档
- [ ] README
- [ ] architecture
- [ ] kv cache design
- [ ] scheduler design
- [ ] paged kv design
- [ ] benchmark methodology
- [ ] known limitations
- [ ] interview notes

## N. 最终简历
- [ ] 所有性能数字来自真实 benchmark
- [ ] 不写“完整复刻 vLLM”
- [ ] 不写“实现 PagedAttention CUDA”除非真的做了
- [ ] 强调从零实现核心执行链
- [ ] 强调 Scheduler + KV + Serving + Benchmark
