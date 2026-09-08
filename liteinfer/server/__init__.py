"""Task 09：LiteInfer 服务层（OpenAI 兼容 HTTP API）。

这个包**不在** ``liteinfer/__init__.py`` 里导出：导入它会拉起 FastAPI，
而快速测试（不依赖模型、也不依赖 Web 栈）不该为此付出代价。
"""

from liteinfer.server.app import create_app

__all__ = ["create_app"]
