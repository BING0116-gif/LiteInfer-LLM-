"""Task 10 测试：Metrics + Tracing。

分两层：

1. 快测（默认 `pytest -q`，不下载模型）：
   - 纯函数层：用**合成时间戳**驱动 TTFT/TPOT/ITL/E2E 的还原计算。为什么
     不用真实引擎计时再断言数值：CPU 调度抖动会让"TPOT ≈ 50ms"这类断言
     假阴性；时间戳合成后所有关系式都是确定性的。
   - 引擎集成：FakeLM 驱动真实 EngineCore，验证打点真的发生（Task 08 的
     教训：断言"度量被记录"在打点根本没发生时也会假绿，所以要断言具体
     的事件序列与数量关系）。

2. 真模型测试（marker=model）：一次真实请求的完整时间线 + 指标自洽。
"""

from __future__ import annotations

import pytest
import torch

from liteinfer.engine.request import Request, RequestState, RequestStatus
from liteinfer.observability import (
    MetricsRegistry,
    RequestMetrics,
    build_trace,
    inter_token_latencies,
    tpot_of,
)
from liteinfer.sampling.params import SamplingParams

from _fakes import make_core, simulate


# --------------------------------------------------------------------------- #
# 合成 RequestState 构造器：时间戳全部手工指定，断言完全确定
# --------------------------------------------------------------------------- #


_SEQ = iter(range(1000))


def _make_state(
    *,
    output_tokens: int = 5,
    wall_start: float = 100.0,
    prefill_start: float = 100.1,
    prefill_end: float = 100.5,
    first_token: float = 100.6,
    step: float = 0.1,
    wall_end: float | None = None,
    status: RequestStatus = RequestStatus.FINISHED,
    finish_reason: str = "length",
) -> RequestState:
    """时间线：enqueue=100.0, prefill 100.1~100.5, token 从 100.6 起每 step 一个。

    request_id 必须唯一：MetricsRegistry 按 id 覆盖记录，重名会静默吞掉
    上一条（本文件的第一次运行就栽在这里）。
    """
    token_times = [first_token + i * step for i in range(output_tokens)]
    if wall_end is None:
        wall_end = token_times[-1] + 0.1 if token_times else wall_start + 1.0
    req = Request(
        request_id=f"r{next(_SEQ)}",
        prompt="12345",
        # SamplingParams 要求 max_tokens>=1（构造期校验），合成态不违反它
        params=SamplingParams(max_tokens=max(1, output_tokens), temperature=0.0),
        status=status,
        prompt_tokens=5,
        generated=list(range(10, 10 + output_tokens)),
        finish_reason=finish_reason,
    )
    st = RequestState(
        request=req,
        wall_start=wall_start,
        wall_end=wall_end,
        prefill_latency_s=prefill_end - prefill_start,
        prefill_start_s=prefill_start,
        prefill_end_s=prefill_end,
        token_times=token_times,
        cache_bytes=1234,
    )
    return st


# --------------------------------------------------------------------------- #
# 纯函数：TTFT / TPOT / ITL / E2E 关系式
# --------------------------------------------------------------------------- #


class TestPureMetrics:
    def test_ttft_covers_queue_and_prefill(self) -> None:
        st = _make_state()
        m = RequestMetrics.from_state(st)
        # 首 token 在 100.6，enqueue 在 100.0：TTFT 必须包含排队(0.1)+prefill(0.4)+采样
        assert m.ttft_s == pytest.approx(0.6)

    def test_itl_sequence(self) -> None:
        st = _make_state(output_tokens=5, step=0.1)
        assert inter_token_latencies(st.token_times) == pytest.approx(
            [0.1, 0.1, 0.1, 0.1]
        )

    def test_tpot_is_mean_of_itls(self) -> None:
        # 不等间隔：token_times = 100.6, 100.7, 100.9, 101.2, 101.6
        st = _make_state(output_tokens=5, step=0.1, first_token=100.6)
        # 手工改成不等间隔验证 TPOT = (last-first)/(n-1)，而非简单平均假设
        st.token_times = [100.6, 100.7, 100.9, 101.2, 101.6]
        assert tpot_of(st.token_times) == pytest.approx(1.0 / 4)
        m = RequestMetrics.from_state(st)
        assert m.tpot_s == pytest.approx(0.25)

    def test_e2e_equals_ttft_plus_tpot_times_n_minus_1_plus_tail(self) -> None:
        """三者的自洽关系：E2E = TTFT + TPOT*(n-1) + (末token→终态的尾巴)。"""
        st = _make_state(output_tokens=5)
        m = RequestMetrics.from_state(st)
        assert m.output_tokens == 5
        tail = st.wall_end - st.token_times[-1]
        assert m.e2e_s == pytest.approx(m.ttft_s + m.tpot_s * 4 + tail)

    def test_single_token_has_no_tpot_and_no_itl(self) -> None:
        """单 token 请求：TPOT 无定义必须是 None（不能拿 0 冒充）。"""
        st = _make_state(output_tokens=1)
        m = RequestMetrics.from_state(st)
        assert m.tpot_s is None
        assert m.itl_p50_s is None and m.itl_p95_s is None and m.itl_max_s is None
        assert inter_token_latencies(st.token_times) == []

    def test_waiting_request_has_zero_ttft(self) -> None:
        """还没产出 token：TTFT=0、TPOT=None，不能抛异常。"""
        st = _make_state(output_tokens=0, status=RequestStatus.WAITING, finish_reason="")
        st.prefill_start_s = 0.0
        st.prefill_end_s = 0.0
        st.wall_end = 0.0
        m = RequestMetrics.from_state(st)
        assert m.ttft_s == 0.0
        assert m.tpot_s is None
        assert m.output_tokens == 0

    def test_itl_percentiles(self) -> None:
        """ITL 分位数：不等间隔序列上的 p50/p95/max 必须可复算。"""
        st = _make_state(output_tokens=0)
        st.token_times = [100.6, 100.7, 100.9, 101.2, 101.6]  # itl: .1 .2 .3 .4
        m = RequestMetrics.from_state(st)
        assert m.itl_p50_s == pytest.approx(0.25)  # 0.2 与 0.3 的插值中点
        assert m.itl_p95_s == pytest.approx(0.385)  # 0.3 + 0.95*3*(0.1)... 线性插值
        assert m.itl_max_s == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
# 全局注册表
# --------------------------------------------------------------------------- #


class TestMetricsRegistry:
    def test_record_is_idempotent_per_request(self) -> None:
        reg = MetricsRegistry()
        m = RequestMetrics.from_state(_make_state())
        reg.record(m)
        reg.record(m)  # 防御式：同请求重复触达不该翻倍
        assert len(reg) == 1

    def test_snapshot_counts_and_window_throughput(self) -> None:
        reg = MetricsRegistry(throughput_window_s=10.0)
        reg.record(RequestMetrics.from_state(_make_state(output_tokens=5)))
        reg.record(
            RequestMetrics.from_state(
                _make_state(
                    output_tokens=2,
                    status=RequestStatus.CANCELLED,
                    finish_reason="cancelled",
                )
            )
        )
        snap = reg.snapshot(
            num_waiting=1,
            num_running=2,
            kv_blocks_used=3,
            kv_blocks_total=8,
            # now=最后事件之后：窗口内 token = 5+2 = 7，吞吐 = 7/10
            now=102.0,
        )
        assert snap["requests_total"] == 2
        assert snap["requests_finished"] == 1
        assert snap["requests_cancelled"] == 1
        assert snap["num_waiting"] == 1 and snap["num_running"] == 2
        assert snap["kv_blocks_used"] == 3 and snap["kv_blocks_total"] == 8
        assert snap["kv_utilization"] == pytest.approx(3 / 8)
        assert snap["output_tokens_total"] == 7
        assert snap["output_tokens_per_s"] == pytest.approx(7 / 10)
    def test_snapshot_window_excludes_old_tokens(self) -> None:
        reg = MetricsRegistry(throughput_window_s=1.0)
        reg.record(RequestMetrics.from_state(_make_state(output_tokens=4)))
        # now=111（token 全部落在 ~100.6~101.6，窗口只覆盖 [110,111]）→ 0 token
        snap = reg.snapshot(now=111.0)
        assert snap["output_tokens_per_s"] == 0.0

    def test_snapshot_kv_utilization_none_when_no_blocks(self) -> None:
        reg = MetricsRegistry()
        snap = reg.snapshot(kv_blocks_used=None, kv_blocks_total=0)
        assert snap["kv_utilization"] is None

    def test_gpu_memory_none_without_cuda(self) -> None:
        """补充条款 A3：无 GPU 时显存指标是 None（展示层渲染 N/A），绝不填 0。"""
        if torch.cuda.is_available():  # 云端跑时本断言不适用
            pytest.skip("GPU 环境")
        reg = MetricsRegistry()
        snap = reg.snapshot()
        assert snap["gpu_memory_mb"] is None
        assert snap["gpu_memory_mb_display"] == "N/A (no GPU)"

    def test_snapshot_means_skip_single_token_requests(self) -> None:
        reg = MetricsRegistry()
        reg.record(RequestMetrics.from_state(_make_state(output_tokens=1)))
        snap = reg.snapshot()
        assert snap["ttft_s_mean"] is not None  # 有 TTFT（>0 才计入）
        assert snap["tpot_s_mean"] is None  # 单 token 无 TPOT，均值也不该是 0


# --------------------------------------------------------------------------- #
# Trace：事件序列还原
# --------------------------------------------------------------------------- #


class TestTrace:
    def test_event_order_and_offsets(self) -> None:
        st = _make_state(output_tokens=3)
        trace = build_trace(st)
        names = [e.name for e in trace.events]
        assert names == [
            "enqueue",
            "prefill_start",
            "prefill_end",
            "token",
            "token",
            "token",
            "finished",
        ]
        d = trace.to_dict()
        offsets = [e["offset_s"] for e in d["events"]]
        # 时间线严格单调不减，且起点（enqueue）偏移为 0
        assert offsets[0] == pytest.approx(0.0)
        assert offsets == sorted(offsets)
        assert offsets[-1] == pytest.approx(st.wall_end - st.wall_start)

    def test_token_events_carry_token_ids(self) -> None:
        st = _make_state(output_tokens=3)
        trace = build_trace(st)
        token_events = [e for e in trace.events if e.name == "token"]
        assert [e.detail for e in token_events] == [
            "index=0 token_id=10",
            "index=1 token_id=11",
            "index=2 token_id=12",
        ]

    def test_waiting_request_trace_has_only_enqueue(self) -> None:
        st = _make_state(output_tokens=0, status=RequestStatus.WAITING, finish_reason="")
        st.prefill_start_s = st.prefill_end_s = st.wall_end = 0.0
        trace = build_trace(st)
        assert [e.name for e in trace.events] == ["enqueue"]

    def test_cancelled_status_names_last_event(self) -> None:
        st = _make_state(
            output_tokens=2,
            status=RequestStatus.CANCELLED,
            finish_reason="cancelled",
        )
        trace = build_trace(st)
        assert trace.events[-1].name == "cancelled"
        assert trace.finish_reason == "cancelled"

    def test_render_is_ascii_and_has_summary(self) -> None:
        st = _make_state(output_tokens=3)
        text = build_trace(st).render()
        text.encode("gbk")  # Windows 控制台可直接打印（Task 04/09 的教训）
        assert "[summary]" in text
        assert "enqueue" in text and "prefill_start" in text


# --------------------------------------------------------------------------- #
# 引擎集成（FakeLM 驱动真实 EngineCore）：打点必须真的发生
# --------------------------------------------------------------------------- #


class TestEngineIntegration:
    def test_finished_request_metrics_and_trace(self) -> None:
        core = make_core(max_new_tokens=4)
        params = SamplingParams(max_tokens=4, temperature=0.0)
        rid = core.submit("2", params)
        out = core.run()[rid]

        assert core.metrics.has(rid), "终态请求没有被记入指标注册表"
        m = core.metrics.get(rid)
        assert m is not None
        assert m.status == "finished" and m.finish_reason == "length"
        assert m.output_tokens == 4 == out.output_tokens
        assert m.prompt_tokens == 1
        assert m.ttft_s > 0.0, "TTFT 必须为正（打点未发生时会假绿）"
        assert m.tpot_s is not None and m.tpot_s >= 0.0
        assert m.e2e_s > 0.0

        trace = core.trace_of(rid)
        names = [e.name for e in trace.events]
        assert names[0] == "enqueue" and names[-1] == "finished"
        assert names.count("token") == 4, "token 事件数应等于输出 token 数"
        assert "prefill_start" in names and "prefill_end" in names

    def test_metrics_snapshot_wired_to_scheduler_and_pool(self) -> None:
        core = make_core(max_new_tokens=4)
        core.submit("2", SamplingParams(max_tokens=4, temperature=0.0))
        core.run()
        snap = core.metrics_snapshot()
        assert snap["kv_blocks_used"] == 0  # 全部结束后块应归还
        assert snap["kv_blocks_total"] == core.runner.paged.num_blocks_total
        assert snap["requests_total"] == 1
        assert snap["output_tokens_total"] == 4
        # num_waiting 必为 0；num_running 不断言：调度器的 running 集合是惰性
        # 清理（下一次 schedule 才剔除已终态请求，Task 06 既有行为，非本 Task 范围）
        assert snap["num_waiting"] == 0

    def test_cancelled_request_recorded_and_blocks_freed(self) -> None:
        core = make_core(max_new_tokens=64)
        rid = core.submit("2", SamplingParams(max_tokens=64, temperature=0.0))
        core.step()  # prefill + 1 token，确保真的占用了块
        assert core.runner.paged.num_blocks_used > 0
        core.cancel(rid)

        m = core.metrics.get(rid)
        assert m is not None and m.status == "cancelled"
        assert m.finish_reason == "cancelled"
        assert core.trace_of(rid).events[-1].name == "cancelled"
        assert core.runner.paged.num_blocks_used == 0

    def test_multi_requests_all_recorded(self) -> None:
        core = make_core(max_new_tokens=3)
        rids = [core.submit(str(s), SamplingParams(max_tokens=3, temperature=0.0)) for s in (1, 2)]
        core.run()
        assert len(core.metrics) == 2
        # token_times 与 generated 长度必须一致（打点与 append 同步）
        for rid in rids:
            st = core._states[rid]
            assert len(st.token_times) == len(st.request.generated) == 3

    def test_trace_of_unknown_raises_keyerror(self) -> None:
        core = make_core()
        with pytest.raises(KeyError):
            core.trace_of("no-such-request")


# --------------------------------------------------------------------------- #
# 真模型：一次请求的完整时间线（Task 10 验收主体）
# --------------------------------------------------------------------------- #


@pytest.mark.model
class TestRealModelTimeline:
    def test_full_timeline_and_metrics_consistency(self) -> None:
        from liteinfer import EngineConfig
        from liteinfer.engine import EngineCore
        from liteinfer.model.eos import resolve_eos_ids
        from liteinfer.model.minimal.weights import load_minimal_from_hf

        cfg = EngineConfig(device="cpu", dtype=torch.float32, max_new_tokens=8)
        loaded = load_minimal_from_hf(cfg)
        eos = resolve_eos_ids(loaded.hf_model, loaded.tokenizer)
        core = EngineCore(loaded.minimal, loaded.tokenizer, cfg, eos_ids=eos)

        rid = core.submit("1+1等于几？只回答数字。", SamplingParams(max_tokens=8, temperature=0.0))
        out = core.run()[rid]

        m = core.metrics.get(rid)
        assert m is not None and m.output_tokens == out.output_tokens
        assert m.ttft_s > 0.0 and m.e2e_s >= m.ttft_s

        trace = core.trace_of(rid)
        names = [e.name for e in trace.events]
        assert names[0] == "enqueue" and names[-1] in ("finished", "cancelled")
        assert names.count("token") == out.output_tokens
        # 时间线自洽：TTFT/TPOT/E2E 三者可互相验算
        assert trace.e2e_s == pytest.approx(out.latency_s, abs=1e-6)
        timeline = trace.render()
        timeline.encode("gbk")  # demo 输出可在 GBK 控制台直接打印
        assert "[summary]" in timeline
