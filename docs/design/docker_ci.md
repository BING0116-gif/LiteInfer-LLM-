# Task 13：Docker + CI + README 设计文档

> 运行环境：CPU（本机已验证）；镜像默认 CPU 版，GPU 构建由 build-arg 切换
> 本机验证命令：见「9. 验收命令」

## 1. 目标

把 Task 01–12 的引擎代码变成可交付、可复现的工程项目：

1. **Dockerfile**：一条命令构建可运行镜像，CPU 为默认目标（补充条款 A），GPU 可切；
2. **CI**：每次 push/PR 自动跑不依赖模型的质量门（pytest 快测 + 补充条款 A1 门禁）；
3. **README**：作品级首页——定位/诚实声明/架构/快速开始/部署/Benchmark/局限；
4. **known_limitations.md**：项目所有已知局限的单一权威清单（内容全部引用既有文档）。

## 2. 交付结构

```text
Dockerfile                     # 构建镜像（ARG TORCH_INDEX_URL，默认 CPU wheel）
.dockerignore                  # 排除 .venv/hf_cache/pip_cache/docs/.git（防 GB 级上下文）
.github/workflows/ci.yml       # tests(默认) + docker-build(workflow_dispatch)
docs/known_limitations.md      # 已知局限权威清单（7 组）
docs/design/docker_ci.md       # 本文档
tests/test_deployment_assets.py# 部署资产约束单测（6 条）
README.md                      # 重写（原为 Task 01 初稿，状态过时）
```

## 3. 关键数据结构

- **Dockerfile 层序**（层缓存顺序即约束顺序）：
  `FROM python:3.12-slim` → `ENV HF_HOME=/model-cache` → `ARG` + `RUN pip install torch`
  → `RUN pip install sentencepiece` → `COPY pyproject+README` → `COPY liteinfer`
  → `RUN pip install -e ".[dev]"` → `COPY tests/examples/benchmark` → `CMD 服务入口`。
  重依赖（torch 196MB）独占一层：业务代码改动不触发 torch 重下。
- **.dockerignore 排除表**：`.venv/`、`hf_cache/`、`pip_cache/`（体积）、`docs/`
  `PROGRESS.md`（运行不需要）、`.git/`、`benchmark/results/`、各类缓存目录。
- **ci.yml job 结构**：
  `tests`（ubuntu-latest，6 步：checkout → setup-python → 装 CPU torch → 装 dev →
  pytest 快测 → A1 门禁）+ `docker-build`（`if: workflow_dispatch`，只 build 不 push）。
- **test_deployment_assets.py**：6 条纯静态断言 + `pytestmark = pytest.mark.skipif`
  （镜像内跳过，因为部署资产按设计不进镜像）。

## 4. 关键设计决策与理由

1. **torch 先单独装（cpu index），再 `-e ".[dev]"`**：`pyproject` 只声明 `torch>=2.0`，
   默认 PyPI 会拉带 GPU 的 wheel（~2GB）；先装 CPU 版后 pip 识别已满足，editable 安装
   不会重拉。这让镜像体积/时间可控，且与"开发机无 GPU"的约束一致。
2. **CI 只跑 `-m "not model"`**：免费 runner 无 GPU，模型测试还要下载 Qwen 1GB；
   核心质量（240 条快测 + 设备字面量门禁）在 push 时已全量守护，model 测试留本地验收。
3. **docker-build 手动触发**：首次 build 因 torch 下载慢（本机实测 ~28 分钟），
   每次 push 都跑会拖慢反馈；本机 Docker 已可独立验证，CI 侧只做干净环境兜底。
4. **部署资产测试加双上下文守卫**：镜像/仓库两种上下文中这些文件存在性相反，
   不加守卫会导致"镜像内跑 pytest 假红"（见 6. 踩坑）。守卫让每个上下文只验证
   该验证的，语义诚实。
5. **README 诚实声明**：明确"不是完整 vLLM、Paged KV 是 gather 实现非 CUDA kernel"、
   "CPU 数字不进简历"；benchmark 结果表只引用 `report.md` 真实产物。

## 5. Alternative 方案（未采纳及原因）

| 方案 | 未采纳原因 |
|---|---|
| 双 Dockerfile（`Dockerfile.cpu` / `.gpu`） | 一份文件 + build-arg 已覆盖两种构建；双文件会增加维护面且容易漂移 |
| CI 里跑 `benchmark_demo --smoke` | 需要 matplotlib + 模型下载，免费 runner 慢且不稳定；smoke 已在本机/容器验证 |
| 镜像只装生产依赖（不含 dev） | 容器内验收（pytest/demo）是 Task 13 明确要求，`[dev]` 一次装齐让容器自足 |
| README 全文放性能结论表 | 报告级数据已集中在 `benchmark/results/`，README 只放结论 + 指向产物，避免双份维护 |

## 6. Known Limitations

- 镜像默认 root 用户运行（pip 的 root 警告存在）；未做非 root 用户与用户态权限隔离。
- CI 免费 runner 不跑 model 测试与 GPU benchmark（见 [docs/known_limitations.md](../known_limitations.md) §6）。
- GitHub Actions badge 指向仓库 `BING0116-gif/LiteInfer-LLM-`，仓库改名后需同步更新。
- `docs/`、`.github/` 不进镜像是设计；因此镜像内无法查看仓库文档（多文档请用 `docker cp` 或直接看本机）。

## 7. 踩过的坑

- 本机 venv 从未装 dev extras → `fastapi/matplotlib` 缺失，pytest 收集 3 文件崩：
  先 `pip list` 确诊再补 `pip install -e ".[dev]"`，勿猜测式修复。
- PShell 无 `&&`，环境变量赋值与命令要用 `;` 分行（`$env:...`）。
- PyYAML 1.1 把 CI 的 `on:` 解析为 bool，验证脚本别对 `d.keys()` 直接 `sorted()`。
- 部署资产测试在镜像内 FileNotFoundError → module 级 skipif 守卫（见 4-4）。
- 首次 `docker build` 的瓶颈是 torch CPU wheel 下载（~150kB/s × 196MB）；build 一次后层缓存秒级。

## 8. 本机验证命令与实测（Task 13）

```text
pytest -q                                        → 242 passed, 37 deselected
pytest tests/test_deployment_assets.py -q        → 6 passed
docker build -t liteinfer:cpu .                  → 构建成功
docker run --rm liteinfer:cpu python -m pytest -q → 236 passed, 6 skipped, 37 deselected
docker run --rm -v D:/LiteInfer/hf_cache:/model-cache liteinfer:cpu \
    python examples/async_engine_demo.py --max-tokens 8 → identical True + 零泄漏 OK
docker run --rm liteinfer:cpu python -m liteinfer.server.main --help → usage 正常
```

## 9. 验收命令

```bash
# 本机（仓库上下文）
set "HF_HOME=D:\LiteInfer\hf_cache"; set PYTHONPATH=d:\LiteInfer
python -m pytest -q                                   # 快速测试全绿（预期 242 passed）
python -m pytest tests/test_no_hardcoded_cuda.py -q   # A1 门禁 1 passed

# Docker（需本机 Docker 可用）
docker build -t liteinfer:cpu .
docker run --rm liteinfer:cpu python -m pytest -q     # 镜像内自检
docker run --rm -v D:/LiteInfer/hf_cache:/model-cache liteinfer:cpu \
    python examples/async_engine_demo.py --max-tokens 8   # 真模型冒烟

# CI：push 到 GitHub 后查看 Actions 面板 tests job 绿；
# 手动触发 docker-build job（Actions → Run workflow）。
```