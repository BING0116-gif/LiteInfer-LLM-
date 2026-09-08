"""Task 06 测试：Scheduler 纯逻辑（无模型，不下载权重）。

覆盖：
- FCFS 顺序准入；
- sequence budget（并发序列上限）截断；
- token budget（单步 token 总量上限）截断 prefill；
- 在飞请求终态后 slot 即时释放、waiting 队首 FCFS 补位（连续批处理核心）；
- 取消从队列移除；
- 小预算下不饿死（所有请求最终都被准入）；
- 已在 running 的请求每步必被调度（decode）。

调度器只依赖 ``SchedulerRequestInfo`` 快照，不触碰张量/缓存，因此可完全脱离模型测试。
"""

from __future__ import annotations

from liteinfer.engine.request import RequestStatus
from liteinfer.scheduler.config import SchedulerConfig
from liteinfer.scheduler.scheduler import Scheduler, SchedulerRequestInfo


def _info(rid: str, prompt_len: int, status: RequestStatus = RequestStatus.WAITING, output_len: int = 0):
    return SchedulerRequestInfo(
        request_id=rid, prompt_len=prompt_len, output_len=output_len, status=status
    )


class TestSchedulerFCFS:
    def test_fcfs_admits_in_order(self) -> None:
        s = Scheduler(SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=64))
        for rid in ("r1", "r2", "r3", "r4"):
            s.enqueue(rid)
        snap = {f"r{i}": _info(f"r{i}", 1) for i in range(1, 5)}
        batch = s.schedule(snap)
        # 队首 3 个准入，第 4 个因 seq budget 排队
        assert batch.prefill_ids == ["r1", "r2", "r3"]
        assert s.num_running == 3
        assert s.num_waiting == 1

    def test_all_ids_prefill_then_decode_order(self) -> None:
        s = Scheduler(SchedulerConfig(max_num_seqs=2))
        s.enqueue("a")
        s.enqueue("b")
        snap = {"a": _info("a", 1), "b": _info("b", 1)}
        batch = s.schedule(snap)
        # prefill 在前、decode 在后（本步无 decode），保证新准入请求先出首 token
        assert batch.prefill_ids == ["a", "b"]
        assert batch.decode_ids == []
        assert batch.all_ids == ["a", "b"]


class TestSequenceBudget:
    def test_sequence_budget_caps_running(self) -> None:
        s = Scheduler(SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=1024))
        for rid in ("a", "b", "c", "d"):
            s.enqueue(rid)
        snap = {rid: _info(rid, 1) for rid in ("a", "b", "c", "d")}
        batch = s.schedule(snap)
        assert len(batch.prefill_ids) == 2
        assert s.num_running == 2

        # a,b 终态 -> 释放 slot -> FCFS 补位 c,d
        for rid in ("a", "b"):
            snap[rid] = _info(rid, 1, RequestStatus.FINISHED)
        batch2 = s.schedule(snap)
        assert batch2.prefill_ids == ["c", "d"]
        assert s.num_running == 2  # a,b 已移除，c,d 在跑
        assert "a" not in s._running and "b" not in s._running

    def test_small_budget_no_starvation(self) -> None:
        # seq budget=1：每步只允许 1 个在飞，但所有请求最终都该被准入（不饿死）
        s = Scheduler(SchedulerConfig(max_num_seqs=1, max_num_batched_tokens=1024))
        for rid in ("a", "b", "c", "d"):
            s.enqueue(rid)
        snap = {rid: _info(rid, 1, RequestStatus.WAITING) for rid in ("a", "b", "c", "d")}
        admitted: list[str] = []
        for _ in range(20):
            if not s.num_waiting and not s.num_running:
                break
            batch = s.schedule(snap)
            if not batch.all_ids:
                break
            rid = batch.all_ids[0]
            admitted.append(rid)
            snap[rid] = _info(rid, 1, RequestStatus.FINISHED)  # 标记完成以释放 slot
        assert admitted == ["a", "b", "c", "d"]


class TestTokenBudget:
    def test_token_budget_blocks_prefill(self) -> None:
        # token budget=5，每个 prompt 长度 3 => 放得下 1 个(3)，第 2 个 6>5 被挡
        s = Scheduler(SchedulerConfig(max_num_seqs=10, max_num_batched_tokens=5))
        s.enqueue("x")
        s.enqueue("y")
        snap = {"x": _info("x", 3), "y": _info("y", 3)}
        batch = s.schedule(snap)
        assert batch.prefill_ids == ["x"]
        assert s.num_waiting == 1

        # x 完成后，y 在下一步被准入
        snap["x"] = _info("x", 3, RequestStatus.FINISHED)
        batch2 = s.schedule(snap)
        assert batch2.prefill_ids == ["y"]

    def test_token_budget_allows_multiple_small(self) -> None:
        s = Scheduler(SchedulerConfig(max_num_seqs=10, max_num_batched_tokens=10))
        for rid in ("a", "b", "c", "d"):
            s.enqueue(rid)  # 各自长度 2 => 4*2=8 <= 10
        snap = {rid: _info(rid, 2) for rid in ("a", "b", "c", "d")}
        batch = s.schedule(snap)
        assert batch.prefill_ids == ["a", "b", "c", "d"]

    def test_running_decode_always_scheduled_under_budget(self) -> None:
        # 先准入（WAITING -> prefill），再模拟引擎 prefill 后转 DECODE：
        # 此时它们已在 running 集合，每步必被调度为 decode（cost=1），不受 waiting 准入影响
        s = Scheduler(SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=1024))
        s.enqueue("a")
        s.enqueue("b")
        snap_wait = {"a": _info("a", 1), "b": _info("b", 1)}
        b0 = s.schedule(snap_wait)  # 准入，作为 prefill
        assert set(b0.prefill_ids) == {"a", "b"}

        snap_dec = {
            "a": _info("a", 1, RequestStatus.DECODE),
            "b": _info("b", 1, RequestStatus.DECODE),
        }
        batch = s.schedule(snap_dec)
        assert batch.decode_ids == ["a", "b"]
        assert batch.prefill_ids == []


class TestCancelAndTerminal:
    def test_cancel_before_admission_removes_from_queue(self) -> None:
        s = Scheduler(SchedulerConfig())
        s.enqueue("a")
        s.enqueue("b")
        s.remove("a")
        snap = {"a": _info("a", 1), "b": _info("b", 1)}
        batch = s.schedule(snap)
        assert batch.prefill_ids == ["b"]
        assert "a" not in batch.all_ids

    def test_cancel_running_then_reschedule(self) -> None:
        s = Scheduler(SchedulerConfig(max_num_seqs=2))
        s.enqueue("b")
        s.enqueue("c")
        snap = {
            "b": _info("b", 1, RequestStatus.DECODE),
            "c": _info("c", 1, RequestStatus.WAITING),
        }
        s.schedule(snap)  # b decode, c prefill
        s.remove("b")  # b 被取消
        snap = {
            "b": _info("b", 1, RequestStatus.CANCELLED),
            "c": _info("c", 1, RequestStatus.DECODE),
        }
        batch = s.schedule(snap)
        assert "b" not in batch.all_ids
        assert batch.decode_ids == ["c"]

    def test_terminal_removed_from_running_each_step(self) -> None:
        s = Scheduler(SchedulerConfig(max_num_seqs=2))
        s.enqueue("a")
        s.enqueue("b")
        snap = {"a": _info("a", 1), "b": _info("b", 1)}
        s.schedule(snap)
        assert s.num_running == 2
        snap["a"] = _info("a", 1, RequestStatus.FINISHED)
        s.schedule(snap)
        assert "a" not in s._running
        assert s.num_running == 1


class TestSchedulerConfig:
    def test_defaults(self) -> None:
        cfg = SchedulerConfig()
        assert cfg.max_num_seqs == 16
        assert cfg.max_num_batched_tokens == 2048

    def test_frozen(self) -> None:
        import pytest

        cfg = SchedulerConfig()
        with pytest.raises(Exception):
            cfg.max_num_seqs = 99  # type: ignore[misc]
