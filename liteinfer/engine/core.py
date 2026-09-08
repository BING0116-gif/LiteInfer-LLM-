"""Task 05：Engine Core——多请求的总控层。

定位（docs/02 §3）：Engine Core 是"总控层"，负责接收请求、推进状态机、
调用模型 forward、回收终态请求。本阶段它直接驱动 MinimalQwen 做"逐请求、
每个 step 推进一个 token"的时间片交错（不是真正的 batch 合并前向——那是
Task 08 的 ModelRunner）。调度（waiting/running 队列、FCFS、token/sequence
budget）也尚未引入，本阶段对所有已 submit 的请求"全部同时推进"，这正是
验收"可同时维护多个请求"的含义。

与 Task 04 的关系：复用同一套「模型只返回 logits + 调用方持有 KV 缓存」
的契约。每个请求持有一块专属 ``ContiguousKVCache``（Task 04 已把缓存生命
周期外部化），Task 07 仅把这块缓存换成 paged block，本文件的 step 循环不动。
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional

import torch

from liteinfer.cache.contiguous import ContiguousKVCache, KVCacheConfig
from liteinfer.config import EngineConfig
from liteinfer.device import resolve_dtype
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
from liteinfer.sampling.params import SamplingParams
from liteinfer.sampling.sampler import Sampler

logger = logging.getLogger("liteinfer.engine.core")


def _infer_kv_dims(model: Any) -> tuple:
    """从模型读出缓存形状三元组 ``(num_layers, num_kv_heads, head_dim)``。

    与 CachedGenerator 的 ``_model_kv_dims`` 同义，但更鲁棒：既支持
    ``model.model.layers``（MinimalQwenForCausalLM 的真实结构），也支持测试用
    的假模型把 layers 直接挂在顶层。按 KV 头数而非 Q 头数：GQA 下二者不等，
    按 Q 头数存会把缓存放大 7 倍。
    """
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None) or getattr(model, "layers", None)
    if layers is None:
        raise TypeError(
            f"EngineCore 需要可推断 KV 维度的模型（有 model.layers 或 layers），收到 {type(model).__name__}"
        )
    attn = layers[0].self_attn
    return len(layers), attn.num_kv_heads, attn.head_dim


class EngineCore:
    """多请求推理引擎核心。

    职责边界（docs/07 §五：Scheduler/Cache/ModelRunner 解耦）：
    - 本类只做"请求状态机 + 调用模型"的总控，不实现调度策略（Task 06）、
      不实现 batch 合并（Task 08）、不实现分页（Task 07）；
    - 模型通过构造函数注入（与 CachedGenerator 同风格），便于测试用假模型替换；
    - 设备/dtype 全部取自 ``EngineConfig``，本模块不出现任何设备字面量。
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
        self._kv_dims = _infer_kv_dims(model)
        # 设备/dtype 一律从 EngineConfig 进入（补充条款 A1/A2），不依赖模型参数
        # 当前所在的设备——避免"模型忘了 .to(device)"这类隐患被悄悄吞掉
        self._device = torch.device(self.cfg.device)
        self._dtype = resolve_dtype(self.cfg.dtype, self.cfg.device)
        self.registry = RequestRegistry()
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

    # ---- 缓存分配：与 CachedGenerator.new_cache 同形，按请求专属分配 ----

    def new_cache(self, max_seq_len: int) -> ContiguousKVCache:
        num_layers, num_kv_heads, head_dim = self._kv_dims
        cache_cfg = KVCacheConfig(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            dtype=self._dtype,
            device=self._device,
        )
        return ContiguousKVCache(cache_cfg)

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
        # 容量 = prompt + max_tokens：最后一个 token 只需被 forward 一次就结束，
        # 不需要再给它腾位置（与 CachedGenerator 口径一致）
        capacity = prompt_len + params.max_tokens

        # Token budget 闸门：单步可调度 token 上限若连一个 prompt 都放不下，说明
        # 配置过小（或 prompt 异常），fail fast 比静默饿死更易排查。
        if prompt_len > self.scheduler.max_num_batched_tokens:
            raise ValueError(
                f"prompt 长度 {prompt_len} 超过 max_num_batched_tokens="
                f"{self.scheduler.max_num_batched_tokens}，无法准入"
            )

        cache = self.new_cache(capacity)
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
            cache=cache,
            prompt_ids=input_ids,
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
        # 缓存已在 submit 时（inference_mode 外）分配；此处仅读取视图并写入
        with torch.inference_mode():
            logits = self.model(
                st.prompt_ids, kv_caches=st.cache.layer_caches, write_pos=0
            )
        st.prefill_latency_s += time.perf_counter() - t0
        st.cached_len = req.prompt_tokens
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
        step_input = torch.tensor([[st.next_id]], device=self._device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(
                step_input,
                position_ids=torch.tensor([[position]], device=self._device),
                kv_caches=st.cache.layer_caches,
                write_pos=position,
            )
        st.decode_latency_s += time.perf_counter() - t0
        st.cached_len += 1
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
        token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
        if len(req.generated) >= req.params.max_tokens:
            self._mark_finished(st, "length")
            return RequestStepResult(req.request_id, token_id, token_text, True, "length")
        return RequestStepResult(req.request_id, token_id, token_text, False, None)

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
            cache_bytes=st.cache.nbytes,
        )
