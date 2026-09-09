"""Task 10：可观测层（Metrics + Tracing）。

docs/02 §1 架构图中的"旁路：Metrics / Tracing"落点。本包只做**纯计算**：
从 ``RequestState`` 上的打点字段还原请求级指标（TTFT/TPOT/ITL/E2E）、
时间线（trace）与全局快照（waiting/running、KV utilization、吞吐），
不反向依赖也不驱动引擎。

导出面刻意收窄：``RequestMetrics``（请求级指标）、``MetricsRegistry``
（全局注册表）、``RequestTrace`` / ``build_trace``（时间线）。
"""

from liteinfer.observability.metrics import (
    MetricsRegistry,
    RequestMetrics,
    inter_token_latencies,
    tpot_of,
)
from liteinfer.observability.trace import RequestTrace, TraceEvent, build_trace

__all__ = [
    "MetricsRegistry",
    "RequestMetrics",
    "RequestTrace",
    "TraceEvent",
    "build_trace",
    "inter_token_latencies",
    "tpot_of",
]
