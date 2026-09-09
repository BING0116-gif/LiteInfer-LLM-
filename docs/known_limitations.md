# LiteInfer Known Limitations

> 本文汇总全部「当前不做 / 做不到 / 需要注意」事项，全部来自仓库内已有的
> `docs/00_README.md`、`docs/02_Technical_Architecture.md`、各 `docs/design/*.md`
> 与 [PROGRESS.md](../PROGRESS.md) 的真实记录，保持诚实声明优先。

## 1. 架构与平台范围

- 仅支持**单机、单设备**（CPU 或一块 GPU）；不支持 Multi-GPU、Tensor Parallel、分布式推理。
- 仅支持 **Decoder-only Transformer、Qwen2 / Qwen2.5 架构**；不支持 MoE。
- 仅支持文本生成；不支持多模态。
- 基础设施围绕 PyTorch 实现，未提供 TensorRT / ONNX 后端。

## 2. 模型与推理功能边界

- **不依赖 `model.generate()`**：HF `generate()` 仅作为 Baseline 阶段的正确性参照物，
  生产推理全部走自主的 Manual Generation Loop。
- **Paged KV 是 PyTorch gather 实现，不是 CUDA PagedAttention Kernel**。
  每步对历史 KV 做一次真实拷贝（gather），比连续缓存的零拷贝视图慢；完整 Kernel 不在第一版范围。
- 支持 PyTorch 的 gather 作为块读取手段，未实现 CUDA kernel 级 PagedAttention。
- 不支持量化（AWQ/GPTQ/FP8）、LoRA Serving、Speculative Decoding、CPU KV Swap、Chunked Prefill（后续工作）。

## 3. Prefix Cache

- 只缓存**完整 Block**；prompt 恰好等于整数块时，最后一块不缓存（logits 只能由 forward 产出），
  单块 prompt 命中恒为 0。
- 命中只发生在**同一引擎实例**内（PrefixCache 挂在 ModelRunner 实例上），新建引擎即全新缓存。
- 启用 `enable_prefix_cache` 后缓存块会"故意占住"块池：结束请求后 `used_blocks != 0` 是预期，
  泄漏契约是 `free + used == total`；默认关闭，保证"请求结束后块全回收"契约不被破坏。

## 4. 调度与内存

- Scheduler 的 running 集合是**惰性清理**：请求终态后、下一次 `schedule()` 之前
  `num_running` 仍计入它；`run()` 刚结束断言 `num_running==0` 会失败。
- **token 预算 fail-fast**：prompt 长度超过 `max_num_batched_tokens` 时提交直接抛 ValueError，
  而不是排队饿死。
- 块池容量构造时静态定死（`num_blocks` 不可动态扩容），并发准入前需确保块数够用。
- 连续 KV Cache 按 `prompt_len + max_tokens` 逐请求预分配，存在内部/外部碎片
  （Paged KV 的动机，也是与实现相关的取舍）。

## 5. Benchmark 与指标

- **CPU 数字只验证流水线正确性，不进简历/汇报**；正式性能数字必须来自云端 GPU（FP16）。
- 显存类指标在无 GPU 时输出 `N/A (no GPU)`，禁止填 0。
- 6 个命名消融 → 5 个真实驱动：continuous batching 与 paged KV 在仓库中合并进
  EngineCore 一个驱动，两者的吞吐与 KV 利用率是同一运行的两个正交测量轴，不能拆分出
  "独立的两套实现"做对照。
- 顺序驱动（hf/nokv/kv）无 prefill/decode 拆段计时，其 TTFT/TPOT/ITL 为 N/A。
- 并发语义：`num_requests = concurrency`；顺序驱动逐个排跑，引擎驱动同时提交，两口径可比的是 out_tokens/wall。

## 6. 服务与部署

- GitHub Actions 免费 runner 无 GPU，CI 只跑**不依赖真实模型**的快速测试；
  模型测试需在本地（含下载 Qwen2.5-0.5B 约 1GB）或 GPU runner 上手动触发。
- Docker 镜像**默认 CPU**（torch 走 `download.pytorch.org/whl/cpu`）；
  GPU 构建需通过 `--build-arg TORCH_INDEX_URL=...` 显式切换，镜像内本身不写死设备。
- 事件循环约束：AsyncEngine 的队列/future 必须与引擎循环同 loop；
  测试中不要用会另起线程事件循环的 Client（本项目用 `httpx.AsyncClient` 挂在同一 loop）。

## 7. 测试相关注意事项

- 一次跑全部 `-m model` 测试会把 16.9GB 内存打满（每个模块加载 HF + MinimalQwen 两份 0.5B）：
  建议逐文件跑或把 model fixture 改为 session 级（涉及既有测试文件改动，未做大改）。
- Fast tokenizer 在缺 `sentencepiece` 时回退 `use_fast=False`；容器镜像已内置 sentencepiece。