"""Task 10：请求级时间线（Tracing）。

验收标准（docs/07 §四 Task 10）：**一次请求可输出完整时间线**。本模块把
``RequestState`` 上的打点字段还原成一条按时间排序的事件序列：

    enqueue -> prefill_start -> prefill_end -> token(0..n-1) -> finished/cancelled

为什么不采用"引擎逐事件回调记录"的通用 tracer：通用方案需要在 core 的每个
关键位置插入 ``observer.on_event(...)`` 调用，等于把观测协议反向耦合进引擎
热路径；而引擎的单写者状态机（``RequestState``）本身已经天然携带了全部
关键时刻（Task 05 的 wall_start/wall_end、Task 04 的两段计时、Task 10 新增
的 token 时间戳），从中**派生**时间线即可，core 无需感知 trace 的存在。
代价是事件粒度固定为上述 5 类——对延迟归因（排队/prefill/逐 token）而言
已经完备，更细粒度（逐层 forward）属于 profiler 的职责，不属于 serving 指标。

渲染使用纯 ASCII（Task 04/09 的教训：Windows GBK 控制台打非 ASCII 符号会
乱码甚至抛 UnicodeEncodeError）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from liteinfer.observability.metrics import inter_token_latencies, tpot_of

if TYPE_CHECKING:  # 只作类型标注：不在运行时依赖 engine，避免导入环
    from liteinfer.engine.request import RequestState


@dataclass(frozen=True)
class TraceEvent:
    """时间线上的单个事件。``at_s`` 是 ``perf_counter`` 原始时刻，
    相对偏移在渲染/to_dict 时才换算（基准 = enqueue 时刻）。"""

    name: str
    at_s: float
    detail: str = ""


@dataclass(frozen=True)
class RequestTrace:
    """单请求完整时间线 + 由同一份时间戳导出的关键延迟（保证自洽）。"""

    request_id: str
    status: str
    finish_reason: Optional[str]
    events: tuple[TraceEvent, ...]
    ttft_s: float
    tpot_s: Optional[float]
    e2e_s: float
    output_tokens: int
    itls: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON 安全的结构化输出（服务层 ``/v1/requests/{id}/trace`` 直接返回它）。"""
        base = self.events[0].at_s if self.events else 0.0
        return {
            "request_id": self.request_id,
            "status": self.status,
            "finish_reason": self.finish_reason,
            "ttft_s": self.ttft_s,
            "tpot_s": self.tpot_s,
            "e2e_s": self.e2e_s,
            "output_tokens": self.output_tokens,
            "events": [
                {
                    "name": e.name,
                    "offset_s": e.at_s - base,
                    "at_s": e.at_s,
                    "detail": e.detail,
                }
                for e in self.events
            ],
        }

    def render(self) -> str:
        """人类可读的 ASCII 时间线，每行 ``+<偏移>s  <事件>  <细节>``。"""
        base = self.events[0].at_s if self.events else 0.0
        lines = [f"trace of {self.request_id} [{self.status}]"]
        for e in self.events:
            suffix = f"  {e.detail}" if e.detail else ""
            lines.append(f"  +{e.at_s - base:9.6f}s  {e.name}{suffix}")
        # 汇总行：时间线的"结论"，字段缺失（如单 token 无 TPOT）时显示 N/A
        tpot = f"{self.tpot_s * 1000:.2f}ms" if self.tpot_s is not None else "N/A"
        lines.append(
            f"  [summary] tokens={self.output_tokens} ttft={self.ttft_s * 1000:.2f}ms "
            f"tpot={tpot} e2e={self.e2e_s:.4f}s"
        )
        return "\n".join(lines)


def build_trace(st: "RequestState") -> RequestTrace:
    """从引擎内部态还原时间线。任何非终态请求也可调用（时间线截至当前）。"""
    req = st.request
    events: list[TraceEvent] = [
        TraceEvent("enqueue", st.wall_start, f"prompt_tokens={req.prompt_tokens}")
    ]
    # prefill_start_s == 0 表示还没被调度准入（还在 waiting 排队）
    if st.prefill_start_s > 0:
        events.append(
            TraceEvent("prefill_start", st.prefill_start_s, f"prompt_len={req.prompt_tokens}")
        )
    if st.prefill_end_s > 0:
        events.append(
            TraceEvent("prefill_end", st.prefill_end_s, f"latency={st.prefill_latency_s:.6f}s")
        )
    # token_times 与 req.generated 同步 append（同一处代码），按下标一一对应
    for i, ts in enumerate(st.token_times):
        token_id = req.generated[i] if i < len(req.generated) else None
        events.append(TraceEvent("token", ts, f"index={i} token_id={token_id}"))
    if st.wall_end > 0:
        events.append(
            TraceEvent(req.status.value, st.wall_end, f"finish_reason={req.finish_reason}")
        )

    e2e = (st.wall_end - st.wall_start) if st.wall_end > 0 else 0.0
    itls = tuple(inter_token_latencies(st.token_times))
    return RequestTrace(
        request_id=req.request_id,
        status=req.status.value,
        finish_reason=req.finish_reason,
        events=tuple(events),
        ttft_s=(st.token_times[0] - st.wall_start) if st.token_times else 0.0,
        tpot_s=tpot_of(st.token_times),
        e2e_s=e2e,
        output_tokens=req.output_tokens,
        itls=itls,
    )
