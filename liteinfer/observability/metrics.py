"""Task 10：请求级指标 + 全局指标注册表。

设计立场：**观测是旁路的纯计算层**。本模块不持有任何引擎内部引用、不反向
驱动引擎；所有指标都从 ``RequestState`` 上的打点字段（Task 10 在 core 中
新增的 ``token_times`` / ``prefill_start_s`` / ``prefill_end_s``）以纯函数
方式还原。这带来两个直接好处：

1. 指标计算可以脱离引擎、用合成时间戳做确定性单测（不靠 sleep，测试不会
   因为 CPU 调度抖动而假阴性）；
2. 同步 ``EngineCore.run()`` 与异步 ``AsyncEngine`` 两条执行路径天然共享
   同一套打点，不存在"流式路径漏记"的问题。

指标定义（与 vLLM/OpenAI 语义对齐）：
- ``TTFT`` = 首 token 时刻 - enqueue 时刻（覆盖排队 + prefill，这正是
  prefill/decode 两段分开计时的意义所在）；
- ``ITL_i`` = token_i 时刻 - token_{i-1} 时刻（i>=2）；
- ``TPOT`` = mean(ITL) = (末 token - 首 token) / (n-1)，单 token 请求无定义
  （返回 None，与 OpenAI usage 语义一致——不要用 0 冒充）；
- ``E2E`` = 终态时刻 - enqueue 时刻。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from liteinfer.device import peak_memory_mb

if TYPE_CHECKING:  # 只作类型标注：observability 不在运行时依赖 engine，避免导入环
    from liteinfer.engine.request import RequestState


# --------------------------------------------------------------------------- #
# 纯函数：从时间戳序列还原指标（可独立单测）
# --------------------------------------------------------------------------- #


def ttft_of(wall_start: float, token_times: Any) -> float:
    """首 token 延迟（s）。尚无 token 时返回 0.0（等待中的请求还没有 TTFT）。"""
    if not token_times:
        return 0.0
    return max(0.0, float(token_times[0]) - wall_start)


def inter_token_latencies(token_times: Any) -> list[float]:
    """相邻 token 间隔序列（s）。少于 2 个 token 时空列表。"""
    times = [float(t) for t in token_times]
    return [times[i] - times[i - 1] for i in range(1, len(times))]


def tpot_of(token_times: Any) -> Optional[float]:
    """平均每输出 token 耗时（s）。单 token 请求没有"每个 token"可言，返回 None。"""
    itls = inter_token_latencies(token_times)
    if not itls:
        return None
    return sum(itls) / len(itls)


def _percentile(sorted_values: list[float], q: float) -> Optional[float]:
    """线性插值分位数。``sorted_values`` 必须已升序；空序列返回 None。"""
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


# --------------------------------------------------------------------------- #
# 请求级指标
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RequestMetrics:
    """单个请求的完整延迟画像（在终态时一次性汇总，不可变）。

    ``token_times`` 保留每个 token 的产出时刻：全局吞吐的滚动窗口统计
    （"最近 N 秒输出了多少 token"）必须基于 token 级时间戳，仅有请求级的
    tokens_per_s 聚合值算不出真实窗口吞吐。
    """

    request_id: str
    status: str
    finish_reason: str
    prompt_tokens: int
    output_tokens: int
    ttft_s: float
    tpot_s: Optional[float]
    e2e_s: float
    tokens_per_s: float
    itl_p50_s: Optional[float]
    itl_p95_s: Optional[float]
    itl_max_s: Optional[float]
    prefill_latency_s: float
    decode_latency_s: float
    cache_bytes: int
    end_s: float
    token_times: tuple[float, ...] = ()

    @classmethod
    def from_state(cls, st: "RequestState") -> "RequestMetrics":
        """从引擎内部态一次性汇总。只在终态调用（wall_end 已回填）。"""
        req = st.request
        e2e = (st.wall_end - st.wall_start) if st.wall_end > 0 else 0.0
        itls = sorted(inter_token_latencies(st.token_times))
        tokens = len(st.token_times)
        # tokens_per_s 用 E2E（含排队+prefill）口径：这是用户感知的端到端吞吐，
        # 与 GenerationOutput.tokens_per_s 的既有语义保持一致
        return cls(
            request_id=req.request_id,
            status=req.status.value,
            finish_reason=req.finish_reason or "",
            prompt_tokens=req.prompt_tokens,
            output_tokens=req.output_tokens,
            ttft_s=ttft_of(st.wall_start, st.token_times),
            tpot_s=tpot_of(st.token_times),
            e2e_s=e2e,
            tokens_per_s=(tokens / e2e) if e2e > 0 else 0.0,
            itl_p50_s=_percentile(itls, 0.50),
            itl_p95_s=_percentile(itls, 0.95),
            itl_max_s=itls[-1] if itls else None,
            prefill_latency_s=st.prefill_latency_s,
            decode_latency_s=st.decode_latency_s,
            cache_bytes=st.cache_bytes,
            end_s=st.wall_end,
            token_times=tuple(float(t) for t in st.token_times),
        )


# --------------------------------------------------------------------------- #
# 全局指标注册表
# --------------------------------------------------------------------------- #


class MetricsRegistry:
    """进程内全局指标：按 request_id 记录终态请求，并输出可序列化的快照。

    为什么用 dict 而不是 list：``_release`` 理论上可能被对同一请求重复触达
    （防御式编程），dict 的覆盖语义天然幂等；同时支持 ``get(rid)`` 供
    服务层做单请求查询。

    线程安全说明：``record`` 只发生在引擎循环线程（core 的单写者纪律内），
    ``snapshot`` 可能被 HTTP 线程调用——dict 的读侧在 CPython 下是原子的，
    快照不是强一致（少记一个刚结束的请求），对观测指标而言可接受，
    不为此引入锁拖累引擎热路径。
    """

    def __init__(self, throughput_window_s: float = 10.0) -> None:
        self.throughput_window_s = throughput_window_s
        self._by_id: dict[str, RequestMetrics] = {}

    # ---- 记录 ----

    def record(self, metrics: RequestMetrics) -> None:
        self._by_id[metrics.request_id] = metrics

    def get(self, request_id: str) -> Optional[RequestMetrics]:
        return self._by_id.get(request_id)

    def has(self, request_id: str) -> bool:
        return request_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    # ---- 快照 ----

    def snapshot(
        self,
        *,
        num_waiting: Optional[int] = None,
        num_running: Optional[int] = None,
        kv_blocks_used: Optional[int] = None,
        kv_blocks_total: Optional[int] = None,
        now: Optional[float] = None,
    ) -> dict[str, Any]:
        """输出 JSON 安全的全局快照。

        ``now`` 参数化的原因：滚动窗口吞吐必须可注入时钟才能做确定性单测，
        生产路径传 None 即取当前 ``perf_counter``。

        显存指标遵循补充条款 A3：无 GPU 时 ``gpu_memory_mb`` 为 ``None``，
        展示层渲染成 ``N/A (no GPU)``，绝不填 0。
        """
        current = time.perf_counter() if now is None else now
        finished = sum(1 for m in self._by_id.values() if m.status == "finished")
        cancelled = sum(1 for m in self._by_id.values() if m.status == "cancelled")
        total_tokens = sum(m.output_tokens for m in self._by_id.values())

        # 滚动窗口吞吐：数窗口内产出的 token（基于 token 级时间戳），除以窗宽。
        # 分母恒为窗宽而非窗口时长：窗口刚启动时数字偏低是吞吐定义的自然结果，
        # 用"窗宽"作分母才能和稳态值直接比较。
        window_start = current - self.throughput_window_s
        window_tokens = sum(
            1
            for m in self._by_id.values()
            for t in m.token_times
            if t >= window_start
        )

        ttfts = [m.ttft_s for m in self._by_id.values() if m.output_tokens > 0]
        tpots = [m.tpot_s for m in self._by_id.values() if m.tpot_s is not None]
        kv_util = (
            (kv_blocks_used / kv_blocks_total)
            if kv_blocks_used is not None
            and kv_blocks_total not in (None, 0)
            else None
        )
        gpu_mb = peak_memory_mb()
        return {
            "requests_total": len(self._by_id),
            "requests_finished": finished,
            "requests_cancelled": cancelled,
            "num_waiting": num_waiting,
            "num_running": num_running,
            "kv_blocks_used": kv_blocks_used,
            "kv_blocks_total": kv_blocks_total,
            "kv_utilization": kv_util,
            "output_tokens_total": total_tokens,
            "output_tokens_per_s": window_tokens / self.throughput_window_s,
            "throughput_window_s": self.throughput_window_s,
            "ttft_s_mean": (sum(ttfts) / len(ttfts)) if ttfts else None,
            "tpot_s_mean": (sum(tpots) / len(tpots)) if tpots else None,
            "gpu_memory_mb": gpu_mb,
            "gpu_memory_mb_display": "N/A (no GPU)" if gpu_mb is None else f"{gpu_mb:.1f}",
        }
