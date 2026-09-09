# LiteInfer 镜像
#
# 默认构建 CPU 版本（补充条款 A：开发机无 GPU，一切必须能在 CPU 跑通）。
# 镜像内不写死任何设备，运行设备由 EngineConfig / 服务入口 --device 决定。
#
# 构建：docker build -t liteinfer:cpu .
# 运行：docker run --rm -p 8000:8000 -v D:/LiteInfer/hf_cache:/model-cache liteinfer:cpu
#
# GPU 机器构建（TORCH_INDEX_URL 指到对应 wheel index，例如 CUDA 12.1）：
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
#       -t liteinfer:gpu .
# 运行时同样靠 --device 从配置读取设备，镜像本身与 CPU 版共用同一套代码。

FROM python:3.12-slim

# --torch wheel 来源（缺省 CPU 版，避免 pip 默认从 PyPI 拉一个 CUDA 版 torch 到容器里）
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# --容器内模型缓存路径；运行时用 -v 把本机 hf_cache 挂到这里，避免重复下载
ENV HF_HOME=/model-cache \
    HF_HUB_DISABLE_SYMLINKS=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# --先单独装 torch：它在本项目是重依赖，单独一层缓存，改业务代码时不用重下
RUN python -m pip install torch --index-url "${TORCH_INDEX_URL}"

# --sentencepiece：Qwen 的 fast tokenizer 依赖它（本机 Task 05 踩过的坑），
#   loader 在缺失时会回退 use_fast=False，镜像里装好保证开箱即用
RUN python -m pip install sentencepiece

COPY pyproject.toml README.md ./
COPY liteinfer ./liteinfer
# --editable 安装 + dev 依赖（pytest/ruff/FastAPI/uvicorn/matplotlib 都在 dev extras 里）
RUN python -m pip install -e ".[dev]"

# --测试与示例也进镜像：容器内可直接跑验收命令（pytest -q / demo）
COPY tests ./tests
COPY examples ./examples
COPY benchmark ./benchmark

EXPOSE 8000

CMD ["python", "-m", "liteinfer.server.main", "--host", "0.0.0.0", "--port", "8000"]