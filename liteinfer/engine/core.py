"""Task 05：Engine Core——多请求的总控层。

定位（docs/02 §3）：Engine Core 是"总控层"，负责接收请求、推进状态机、
调用模型 forward、回收终态请求。本阶段它直接驱动 MinimalQwen 做"逐请求、
每个 step 推进一个 token"的时间片交错（不是真正的 batch 合并前向——那是
Task 08 的 ModelRunner）。调度（waiting/running 队列、FCFS、token/sequence
budget）也尚未引入，本阶段对所有已 submit 的请求"全部同时推进"，这正是
验收"可同时维护多个请求"的含义。

与 Task 04 的关系：复用同一套「模型只返回 logits + 调用方持有 KV 缓存」
的契约。Task 08 起每个请求不再预分配连续缓存，而是在首次 prefill 时从
``ModelRunner`` 的共享块池领一张 ``BlockTable``，物理块按需 lazy 分配；
请求进入终态或被取消时由本模块把块还回池里。step 循环的形状不变。
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

import torch

from liteinfer.config import EngineConfig
from liteinfer.device import resolve_dtype
from liteinfer.model.runner import ModelRunner, infer_kv_dims
from liteinfer.scheduler.config import SchedulerConfig
from liteinfer.scheduler.scheduler import (
    Scheduler,
    SchedulerRequestInfo,
)
from liteinfer.engine.request import (
    Request,
    RequestOutput,
    RequestState,
    RequestStatus,
    RequestStepResult,
    RequestRegistry,
)
from liteinfer.model.eos import resolve_eos_ids
from liteinfer.model.minimal.weights import load_minimal_from_hf
from liteinfer.observability.metrics import MetricsRegistry, RequestMetrics
from liteinfer.observability.trace import RequestTrace, build_trace
from liteinfer.sampling.params import SamplingParams
from liteinfer.sampling.sampler import Sampler

logger = logging.getLogger("liteinfer.engine.core")


# KV 维度推断统一由 ModelRunner 提供（infer_kv_dims），此处不再保留第二份实现：
# 缓存形状是"模型 + 配置"共同决定的，只有一份真相才能避免两处算出的块大小不一致。


class EngineCore:
    """多请求推理引擎核心。

    职责边界（docs/07 §五：Scheduler/Cache/ModelRunner 解耦）：
    - 本类只做"请求状态机 + 调用模型"的总控，不实现调度策略（Task 06）、
      不直接操作物理块（Task 08 起由 ``ModelRunner`` 持有共享块池）；
    - 模型通过构造函数注入（与 CachedGenerator 同风格），便于测试用假模型替换；
    - 设备/dtype 全部取自 ``EngineConfig``，本模块不出现任何设备字面量。

    Task 08 的变化：forward 不再直接调 ``self.model``，而是经 ``self.runner``
    （prefill/decode），KV 落在共享的 PagedKVCache 里；请求进入终态 / 被取消时
    由本类负责把块还回池里（docs/02 §2 生命周期的最后一步 "free KV blocks"）。
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        cfg: EngineConfig,
        eos_ids: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.sampler = Sampler()
        # EOS 真相共用 resolve_eos_ids（Task 02/04 同口径），避免两套链终止符不一致
        self._eos_ids = (
            frozenset(int(x) for x in eos_ids)
            if eos_ids is not None
            else resolve_eos_ids(model, tokenizer)
        )
        # Task 08：ModelRunner 持有共享块池 + 执行 prefill/decode。
        # 引擎因此不再自己分配缓存，只在终态负责把块还回池里。
        self.runner = ModelRunner(model, cfg)
        # 设备/dtype 一律从 EngineConfig 进入（补充条款 A1/A2），不依赖模型参数
        # 当前所在的设备——避免"模型忘了 .to(device)"这类隐患被悄悄吞掉
        self._device = torch.device(self.cfg.device)
        self._dtype = resolve_dtype(self.cfg.dtype, self.cfg.device)
        self.registry = RequestRegistry()
        # Task 10：全局指标注册表。record 只发生在本类（单写者）内，
        # snapshot 可被 HTTP 线程并发读取——dict 读侧原子，无需锁（见其 docstring）
        self.metrics = MetricsRegistry()
        # 内部态表：request_id -> RequestState（含缓存/张量，不对外暴露）
        self._states: dict[str, RequestState] = {}
        # Task 06：调度器持有 waiting/running 双队列与预算闸门，决定每步本批跑谁。
        # 配置来自 EngineConfig.scheduler（集中管理），默认 SchedulerConfig()。
        self.scheduler = Scheduler(cfg.scheduler if isinstance(cfg.scheduler, SchedulerConfig) else SchedulerConfig())

    @classmethod
    def from_config(cls, cfg: EngineConfig) -> "EngineCore":
        """加载 MinimalQwen 并构造引擎（EOS 从 HF 模型解析，见 eos.py 说明）。"""
        loaded = load_minimal_from_hf(cfg)
        eos_ids = resolve_eos_ids(loaded.hf_model, loaded.tokenizer)
        return cls(loaded.minimal, loaded.tokenizer, cfg, eos_ids)

    # ---- 块表：Task 08 起由 ModelRunner 的共享池按需分配，引擎不再预分配容量 ----

    def new_block_table(self):
        """为请求开一张空块表（物理块在 prefill/decode 写入时按需分配）。"""
        return self.runner.new_block_table()

    # ---- 请求接入 ----

    def submit(self, prompt: str, params: Optional[SamplingParams] = None) -> str:
        """提交一个请求，返回 request_id。

        在 inference_mode 之外分配缓存与张量：模式内新建的张量是 inference
        tensor，离开上下文后再原地 ``copy_`` 会抛 RuntimeError（Task 04 踩坑）。
        """
        if params is None:
            params = SamplingParams(
                max_tokens=self.cfg.max_new_tokens, temperature=0.0
            )

        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self._device)
        prompt_len = int(input_ids.shape[1])

        # Token budget 闸门：单步可调度 token 上限若连一个 prompt 都放不下，说明
        # 配置过小（或 prompt 异常），fail fast 比静默饿死更易排查。
        if prompt_len > self.scheduler.max_num_batched_tokens:
            raise ValueError(
                f"prompt 长度 {prompt_len} 超过 max_num_batched_tokens="
                f"{self.scheduler.max_num_batched_tokens}，无法准入"
            )

        request_id = uuid.uuid4().hex

        generator: Optional[torch.Generator] = None
        if params.seed is not None and not params.is_greedy:
            # 随机源按请求注入（不共享），并发下互不污染（同 Sampler 的设计动机）
            generator = torch.Generator(device=self._device)
            generator.manual_seed(params.seed)

        req = Request(
            request_id=request_id,
            prompt=prompt,
            params=params,
            status=RequestStatus.WAITING,
            prompt_tokens=prompt_len,
        )
        st = RequestState(
            request=req,
            # 块表留空，等真正被准入做 prefill 时才分配：waiting 队列里的请求
            # 不该提前占住物理块（这正是分页相对 Task 04 预分配的价值）
            cache=None,
            prompt_ids=input_ids,
            # Task 11：prompt 的 token id 列表，前缀哈希与 decode 期块注册共用
            prompt_token_ids=[int(t) for t in input_ids[0].tolist()],
            generator=generator,
            wall_start=time.perf_counter(),
        )
        self.registry.add(req)
        self._states[request_id] = st
        # Task 06：提交即入 waiting 队尾（FCFS）。是否真正开始 prefill 由调度器在
        # 后续 step 按序列/ token 预算准入决定，而非提交即"在飞"。
        self.scheduler.enqueue(request_id)
        logger.debug("提交请求 %s: prompt_len=%d max_tokens=%d", request_id, prompt_len, params.max_tokens)
        return request_id

    def get_request(self, request_id: str) -> Request:
        """读取某请求的当前状态（返回注册表中的活对象，仅供查询）。"""
        return self.registry.get(request_id)

    def trace_of(self, request_id: str) -> RequestTrace:
        """某请求的完整事件时间线（Task 10 验收：一次请求可输出完整时间线）。

        非终态请求也可调用（时间线截至当前时刻）；未知 id 抛 KeyError，
        由服务层翻译成 404。
        """
        st = self._states.get(request_id)
        if st is None:
            raise KeyError(f"未知 request_id: {request_id!r}")
        return build_trace(st)

    def metrics_snapshot(self) -> dict[str, Any]:
        """全局指标快照：把注册表与调度器/块池的实时观测项拼在一起。

        放在 core 而不是服务层：waiting/running/blocks 的真相都在这里，
        服务层只做协议翻译（docs/02 §1 的分层），不该自己拼观测数据。
        """
        return self.metrics.snapshot(
            num_waiting=self.scheduler.num_waiting,
            num_running=self.scheduler.num_running,
            kv_blocks_used=self.runner.paged.num_blocks_used,
            kv_blocks_total=self.runner.paged.num_blocks_total,
            # Task 11：前缀缓存观测。关闭时为 None（键仍存在，值为 null），
            # 与"显存指标无 GPU 时为 None 不填 0"同一口径
            prefix_cached_blocks=(
                self.runner.prefix.num_evictable_blocks
                if self.runner.prefix is not None
                else None
            ),
        )

    def cancel(self, request_id: str) -> None:
        """取消在飞请求：标记为 CANCELLED，停止继续推进，回填已生成文本。"""
        st = self._states.get(request_id)
        if st is None:
            raise KeyError(f"未知 request_id: {request_id!r}")
        req = st.request
        if req.status.is_terminal:
            return
        req.status = RequestStatus.CANCELLED
        req.finish_reason = "cancelled"
        st.wall_end = time.perf_counter()
        req.output_text = (
            self.tokenizer.decode(req.generated, skip_special_tokens=True)
            if req.generated
            else ""
        )
        # Task 06：从调度器队列彻底移除，避免后续 step 仍尝试推进一个已取消的请求
        self.scheduler.remove(request_id)
        # Task 08：取消是 docs/02 §10 的显式路径，必须立刻归还物理块，
        # 否则块会一直被"已取消但还占着块"的请求持有，直到进程退出
        self._release(st)
        logger.debug("取消请求 %s", request_id)

    def active_requests(self) -> list[Request]:
        return self.registry.active()

    # ---- 主循环：每个 step 由 Scheduler 准入出本批，逐个推进 ----

    def _snapshot(self) -> dict[str, SchedulerRequestInfo]:
        """构造调度快照：request_id -> 只读的 (prompt_len, output_len, status)。

        调度器只依赖这份快照做决策，不反向持有引擎内部状态，保证二者解耦。
        """
        snap: dict[str, SchedulerRequestInfo] = {}
        for rid, st in self._states.items():
            req = st.request
            snap[rid] = SchedulerRequestInfo(
                request_id=rid,
                prompt_len=req.prompt_tokens,
                output_len=len(req.generated),
                status=req.status,
            )
        return snap

    def step(self) -> list[RequestStepResult]:
        """推进本步由 Scheduler 准入出的批次各一个 token，返回逐步结果（供 Task 09 流式消费）。

        与 Task 05「所有 active 都步进」不同：本步只跑调度器放行的请求；waiting 中
        尚未准入、或已终态的请求不会推进。调度的「准入 + 双预算闸门」逻辑全在
        Scheduler.schedule 内，引擎主循环形状不变。
        """
        results: list[RequestStepResult] = []
        batch = self.scheduler.schedule(self._snapshot())
        # 用 list 快照：_advance 内部可能改变状态，但不在循环里增删表
        for rid in batch.all_ids:
            st = self._states.get(rid)
            if st is None or st.request.status.is_terminal:
                continue
            results.append(self._advance(st))
        return results

    def run(self, max_steps: Optional[int] = None) -> dict[str, RequestOutput]:
        """跑完所有在飞请求直到全部终态，返回 ``request_id -> RequestOutput``。

        ``max_steps`` 为 None 时按"所有 active 请求的 max_tokens 之和 + 余量"
        自动估算安全上限，避免任何 bug 导致死循环（例如 EOS 永不命中）。
        """
        if max_steps is None:
            active = self.registry.active()
            planned = sum(r.params.max_tokens for r in active)
            max_steps = planned + len(active) * 2 + 16
        steps = 0
        while self.registry.active() and steps < max_steps:
            self.step()
            steps += 1
        if self.registry.active():
            # 安全兜底：达到步数上限仍有未结束请求，强制以 length 终止，
            # 避免 run() 返回悬空请求。正常路径不会走到这里。
            logger.warning("达到 max_steps=%d 仍有 %d 个请求未结束，强制终止", max_steps, len(self.registry.active()))
            for st in list(self._states.values()):
                if not st.request.status.is_terminal:
                    self._mark_finished(st, "length")
        return {r.request_id: self._to_output(self._states[r.request_id]) for r in self.registry.all()}

    # ---- 内部：单请求推进 ----

    def _advance(self, st: RequestState) -> RequestStepResult:
        req = st.request
        if req.status == RequestStatus.WAITING:
            return self._prefill(st)
        return self._decode(st)

    def _prefill(self, st: RequestState) -> RequestStepResult:
        req = st.request
        req.status = RequestStatus.PREFILL
        t0 = time.perf_counter()
        # Task 10 打点：prefill 起点即"排队结束"的时刻，TTFT 的排队段由此可归因
        st.prefill_start_s = t0
        # 块表在被准入的这一刻才分配（waiting 期间不占物理块）；池在
        # ModelRunner 构造时于 inference_mode 之外建好，此处只写入不新建张量
        if st.cache is None:
            st.cache = self.new_block_table()
        # Task 11：前缀命中则收养已有物理块（引用计数在 lookup 内 +1），
        # forward 只跑未命中的后缀——write_pos=hit_len 让 KV 从命中块末尾
        # 续写，历史 KV 由收养的物理块直接提供（gather 读路径零改动）
        prefix = self.runner.prefix
        write_pos = 0
        input_ids = st.prompt_ids
        if prefix is not None:
            hit_ids, hit_len = prefix.lookup(st.prompt_token_ids)
            if hit_len > 0:
                st.cache.adopt(hit_ids, hit_len)
                st.prefix_hit_tokens = hit_len
                input_ids = st.prompt_ids[:, hit_len:]
                write_pos = hit_len
                logger.debug(
                    "请求 %s 前缀命中 %d tokens（%d 块）", req.request_id, hit_len, len(hit_ids)
                )
        logits = self.runner.prefill(input_ids, st.cache, write_pos=write_pos)
        t1 = time.perf_counter()
        st.prefill_latency_s += t1 - t0
        st.prefill_end_s = t1
        st.cached_len = req.prompt_tokens
        # Task 11：把本次（新算出的）完整块注册进哈希表。已注册过的块
        # （收养来的）会因哈希已存在而跳过；EOS 提前结束时 prompt 块也已
        # 注册完毕，释放后留在 LRU 里等后续请求白捡
        if prefix is not None:
            prefix.register(st.prompt_token_ids, st.cache)
        candidate = self.sampler.sample(logits[0, -1], req.params, st.generator)
        req.status = RequestStatus.DECODE

        if candidate in self._eos_ids:
            # 首 token 即 EOS：不产出任何 token，直接结束
            self._mark_finished(st, "eos")
            return RequestStepResult(req.request_id, None, "", True, "eos")

        req.generated.append(candidate)
        st.next_id = candidate
        return self._after_emit(st, candidate)

    def _decode(self, st: RequestState) -> RequestStepResult:
        req = st.request
        position = st.cached_len  # 绝对位置，RoPE 依赖它（Task 04 踩坑）
        t0 = time.perf_counter()
        logits = self.runner.decode(st.next_id, position, st.cache)
        st.decode_latency_s += time.perf_counter() - t0
        st.cached_len += 1
        # Task 11：本步写入后若恰好写满一个完整块，立即注册。必须放在
        # forward 之后（KV 已完整落块）——若提前注册，同批后 prefill 的请求
        # 可能收养到一个"最后槽位还是零"的半成品块。此刻 generated 里的
        # token 都已落块（各自在之前的 step 写入），拼接即精确等于已写序列
        prefix = self.runner.prefix
        if prefix is not None and st.cached_len % self.runner.block_size == 0:
            prefix.register(st.prompt_token_ids + req.generated, st.cache)
        candidate = self.sampler.sample(logits[0, -1], req.params, st.generator)

        if candidate in self._eos_ids:
            # 命中终止符：上一步的 token 已经产出并上报，本步不再吐新 token
            self._mark_finished(st, "eos")
            return RequestStepResult(req.request_id, None, "", True, "eos")

        req.generated.append(candidate)
        st.next_id = candidate
        return self._after_emit(st, candidate)

    def _after_emit(self, st: RequestState, token_id: int) -> RequestStepResult:
        """某 token 已 append 到 generated 后，判断是否达到 max_tokens。"""
        req = st.request
        # Task 10 打点：token 产出时刻。与 req.generated 在同一处同步 append，
        # 保证 token_times[i] 与 generated[i] 的下标对应关系永远成立
        st.token_times.append(time.perf_counter())
        token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
        if len(req.generated) >= req.params.max_tokens:
            self._mark_finished(st, "length")
            return RequestStepResult(req.request_id, token_id, token_text, True, "length")
        return RequestStepResult(req.request_id, token_id, token_text, False, None)

    def _release(self, st: RequestState) -> None:
        """把请求的物理块还回共享池，并在还回去之前记下它占用的字节数。

        顺序很重要：块 ``free`` 之后 ``BlockTable.num_tokens`` 归零，
        字节数就再也问不出来了，而 ``RequestOutput.cache_bytes`` 要在请求
        结束后仍可读（Task 10 指标会消费它），所以先记账再释放。
        """
        if st.cache is None:
            # 没碰过缓存的请求（如 waiting 中即被取消）也必须记账指标，
            # 否则全局请求数会漏掉"从未 prefill"的失败路径
            self.metrics.record(RequestMetrics.from_state(st))
            return
        st.cache_bytes = self.runner.bytes_for(st.cache)
        self.runner.free_table(st.cache)
        st.cache = None
        # Task 10：终态唯一汇聚点（finish 与 cancel 都走 _release），
        # 在这里一次性汇总请求级指标并进入全局注册表
        self.metrics.record(RequestMetrics.from_state(st))

    def _mark_finished(self, st: RequestState, reason: str) -> None:
        req = st.request
        req.status = RequestStatus.FINISHED
        req.finish_reason = reason
        st.wall_end = time.perf_counter()
        req.output_text = (
            self.tokenizer.decode(req.generated, skip_special_tokens=True)
            if req.generated
            else ""
        )
        # Task 08：终态即回收块（docs/02 §2 生命周期最后一步）
        self._release(st)

    def _to_output(self, st: RequestState) -> RequestOutput:
        req = st.request
        latency = (st.wall_end - st.wall_start) if st.wall_end > 0 else 0.0
        out_tokens = req.output_tokens
        return RequestOutput(
            request_id=req.request_id,
            text=req.output_text,
            prompt_tokens=req.prompt_tokens,
            output_tokens=out_tokens,
            finish_reason=req.finish_reason or "length",
            latency_s=latency,
            tokens_per_s=(out_tokens / latency) if latency > 0 else 0.0,
            device=str(self._device),
            dtype=str(self._dtype).replace("torch.", ""),
            prefill_latency_s=st.prefill_latency_s,
            decode_latency_s=st.decode_latency_s,
            cached_tokens=st.cached_len,
            cache_bytes=st.cache_bytes,
            prefix_hit_tokens=st.prefix_hit_tokens,
        )
