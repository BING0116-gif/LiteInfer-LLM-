"""Task 06：Continuous Batching Scheduler。

调度器是**纯逻辑**组件，只持有两个队列：

- ``waiting``：FCFS 先进先出队列（deque），存放已 submit 但尚未被准入的请求；
- ``running``：集合，存放已被准入、正在 prefill/decode 的请求。

它**不持有任何张量、缓存或模型句柄**——这些仍由 ``RequestState.cache`` 持有
（Task 07 会把它换成 paged block）。调度器只回答一个问题：*这一 step 该跑哪些请求，
各自做 prefill 还是 decode*。具体怎么 forward、怎么写 KV 是 EngineCore 的事，二者解耦。

与 EngineCore 的契约：每步 EngineCore 调一次 ``schedule(snapshot)``，传入一个只读
快照 ``dict[request_id, SchedulerRequestInfo]``（含 prompt_len / output_len / status）。
调度器据此返回 ``ScheduledBatch``，之后 EngineCore 只对 ``batch.all_ids`` 里的请求调
``_advance``。这样调度器完全不反向依赖引擎内部状态，符合 docs/07 的「Scheduler/Cache/
ModelRunner 解耦」要求。

调度语义（vLLM 风格简化版，无抢占）：
1. 任何已终态（FINISHED/CANCELLED）的 running 请求，本步从 running 中移除，释放 slot；
2. running 中的请求**每步必被调度**（decode 计 1 token）——已在飞的不应被挂起；
3. 在 token/sequence 预算允许范围内，按 FCFS 从 waiting 队首补位准入（prefill 计
   prompt_len token）；放不下的继续排队，等后续 slot 释放后再试——这就是连续批处理
   相对静态批「一个走完、整批干等」的本质区别。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from liteinfer.scheduler.config import SchedulerConfig


@dataclass
class SchedulerRequestInfo:
    """调度快照：EngineCore 在每个 step 传给 Scheduler 的只读视图。

    只暴露调度决策所需的三个量，绝不暴露张量/缓存，避免调度器与引擎内部强耦合。
    """

    request_id: str
    prompt_len: int
    output_len: int
    status: RequestStatus


@dataclass
class ScheduledBatch:
    """本轮调度结果：哪些请求 prefill、哪些 decode。

    ``prefill_ids`` 里的请求状态仍是 WAITING（尚未做过 prefill），EngineCore 会对其
    执行 prefill 并产出首个 token；``decode_ids`` 已是 DECODE，执行单 token decode。
    """

    prefill_ids: list[str]
    decode_ids: list[str]

    @property
    def all_ids(self) -> list[str]:
        """本轮实际要推进的请求（prefill 在前，decode 在后，保证新准入请求先出首 token）。"""
        return self.prefill_ids + self.decode_ids

    def __len__(self) -> int:
        """本批请求总数（prefill + decode）。"""
        return len(self.prefill_ids) + len(self.decode_ids)


class Scheduler:
    """连续批处理调度器：waiting/running 双队列 + FCFS 准入 + 双预算闸门。

    线程不安全（CPU 单线程推理足够）；所有状态在 ``schedule`` 内一次性推进，无中间态。
    """

    def __init__(self, cfg: SchedulerConfig) -> None:
        self._cfg = cfg
        self._waiting: deque[str] = deque()
        self._running: set[str] = set()

    # ---- 对外查询（供测试 / Task 10 指标）----

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    @property
    def max_num_seqs(self) -> int:
        return self._cfg.max_num_seqs

    @property
    def max_num_batched_tokens(self) -> int:
        return self._cfg.max_num_batched_tokens

    # ---- 队列维护 ----

    def enqueue(self, request_id: str) -> None:
        """提交一个新请求：进 waiting 队尾（FCFS）。"""
        self._waiting.append(request_id)

    def remove(self, request_id: str) -> None:
        """请求被取消或外部回收时，从两个队列中彻底移除。"""
        if request_id in self._waiting:
            self._waiting.remove(request_id)
        self._running.discard(request_id)

    # ---- 核心：每步调度 ----

    def schedule(self, requests: dict[str, SchedulerRequestInfo]) -> ScheduledBatch:
        """根据当前快照决定本轮批次。

        ``requests`` 必须包含本引擎中所有已知请求（waiting + running + 已终态），
        key 为 request_id。调度器据此：(1) 清掉已终态的 running；(2) 必调度在飞
        decode；(3) 在预算内 FCFS 补位 waiting。
        """
        # 局部导入：避免模块加载期的循环依赖（scheduler <-> engine.request）。
        # 运行时调用 schedule 时 engine 已完整初始化，此处导入安全。
        from liteinfer.engine.request import RequestStatus

        # 1. 清理 running 中的终态请求，释放并发 slot（连续批处理的关键：完成即让位）
        self._running = {
            rid for rid in self._running
            if rid in requests and not requests[rid].status.is_terminal
        }

        prefill_ids: list[str] = []
        decode_ids: list[str] = []
        used_tokens = 0

        # 2. 已在 running 的请求必被调度：WAITING(尚未 prefill) 计 prompt_len，
        #    DECODE 计 1。running 请求总是放行（单步 decode 只占 1 token，必放得下）。
        for rid in list(self._running):
            info = requests[rid]
            cost = info.prompt_len if info.status == RequestStatus.WAITING else 1
            used_tokens += cost
            (prefill_ids if info.status == RequestStatus.WAITING else decode_ids).append(rid)

        # 3. FCFS 补位：在 sequence / token 预算内，从 waiting 队首逐步准入。
        #    prompt 超 token budget 的放不下就留队（等后续 slot 释放），不无限阻塞——
        #    因为 running 总会随 max_tokens 结束而释放预算。
        while self._waiting and len(self._running) < self._cfg.max_num_seqs:
            rid = self._waiting[0]
            info = requests[rid]
            cost = info.prompt_len  # 首次 prefill 的 token 量
            if used_tokens + cost > self._cfg.max_num_batched_tokens:
                break
            used_tokens += cost
            self._waiting.popleft()
            self._running.add(rid)
            prefill_ids.append(rid)

        return ScheduledBatch(prefill_ids=prefill_ids, decode_ids=decode_ids)
