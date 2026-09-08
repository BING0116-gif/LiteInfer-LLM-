"""Task 05：请求模型与状态机。

把"一个请求的一生"拆成两层：

- ``Request``：用户态数据模型。引擎对外暴露的全部信息都在这里（id、prompt、
  采样参数、当前状态、已生成 token、终止原因、输出文本）。它不含任何张量/
  缓存句柄——这些是引擎内部资源，混进数据模型会让状态难以追踪，也会在
   Task 09 流式序列化时把 GPU/CPU 张量泄露出去。
- ``RequestState``：引擎内部可变态。持有 ``Request`` 外加"推进生成所需的一切
  可变资源"：专属 KV 缓存、prompt 的张量、已写入缓存的长度、下一步要 forward
  的 token、可选的随机源。每一 step 只动 RequestState，Request 作为它的
  ``request`` 字段被同步更新。

状态枚举 ``RequestStatus`` 与 ``docs/02 §2`` 的生命周期一一对应
（WAITING → PREFILL → DECODE → FINISHED / CANCELLED），方便下游 Scheduler
（Task 06）按状态分组、也方便测试断言。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from liteinfer.sampling.params import SamplingParams


class RequestStatus(str, Enum):
    """请求生命周期状态，字符串枚举便于日志/序列化直接打印。

    PREFILL/DECODE 在 Task 05 的单个 step 内会连续发生（prefill 完成即转
    DECODE），但保留二者是为了让 Task 06 调度器能区分"正在吃 prompt"和
    "正在逐 token 解码"两个阶段，分别计入 TTFT 与 TPOT 预算。
    """

    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """终态：不再参与 step。"""
        return self in (RequestStatus.FINISHED, RequestStatus.CANCELLED)


@dataclass
class Request:
    """用户态请求数据模型（不含任何张量/缓存句柄）。

    ``generated`` 是已产出的 token id 列表；``output_text`` 在结束时由引擎
    解码填入。``finish_reason`` 为 ``"length"``（达到 max_tokens）或
    ``"eos"``（命中终止符）或 ``"cancelled"``（被取消）。
    """

    request_id: str
    prompt: str
    params: SamplingParams
    status: RequestStatus = RequestStatus.WAITING
    prompt_tokens: int = 0
    generated: list[int] = field(default_factory=list)
    finish_reason: Optional[str] = None
    output_text: str = ""

    @property
    def output_tokens(self) -> int:
        return len(self.generated)

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal


@dataclass
class RequestOutput:
    """单请求最终结果的对外结构体（复刻 GenerationOutput 的字段语义）。

    额外带 ``request_id`` 与缓存相关观测，便于 Task 10 指标直接消费；
    prefill/decode 两段计时拆开（同 CachedGenerator 的设计动机：两段瓶颈
    不同，合在一起会掩盖 TTFT/TPOT 差异）。
    """

    request_id: str
    text: str
    prompt_tokens: int
    output_tokens: int
    finish_reason: str
    latency_s: float
    tokens_per_s: float
    device: str
    dtype: str
    prefill_latency_s: float = 0.0
    decode_latency_s: float = 0.0
    cached_tokens: int = 0
    cache_bytes: int = 0


@dataclass
class RequestStepResult:
    """单个 step 内某请求的逐步结果，供 Task 09 流式逐个吐 token。

    ``token_id`` 为 None 表示该 step 未产出新 token（例如 prefill 命中 EOS、
    或该请求本步被跳过）。``finished`` 标记本步是否使请求进入终态。
    """

    request_id: str
    token_id: Optional[int]
    token_text: str
    finished: bool
    finish_reason: Optional[str]


@dataclass
class RequestState:
    """引擎内部可变态：Request + 推进生成所需的全部可变资源。

    缓存 ``cache`` 由本状态持有（Task 04 已把缓存生命周期外部化），每个请求
    一块，互不干扰；Task 07 把它换成 paged block 时只改这一字段的构造方式，
    step 循环本身不动。
    """

    request: Request
    # Task 08：分页 KV。首次 prefill 时才由 ModelRunner 分配（惰性），
    # 终态/取消时由引擎 free_table 归还物理块并置回 None。
    cache: Any = None  # BlockTable | None
    prompt_ids: Any = None  # torch.Tensor [1, P]
    cached_len: int = 0
    next_id: int = 0
    generator: Any = None  # 可选 torch.Generator，greedy 时为 None
    prefill_latency_s: float = 0.0
    decode_latency_s: float = 0.0
    wall_start: float = 0.0
    wall_end: float = 0.0
    # 释放块之前记录的"本请求实际占用 KV 字节数"：块还回去之后就查不到了，
    # 而 RequestOutput.cache_bytes 需要在请求结束后仍能读
    cache_bytes: int = 0


class RequestRegistry:
    """请求注册表：id -> Request 的集中管理。

    只存 Request（用户态），不存 RequestState（内部态）——后者由 EngineCore
    另表持有。这样注册表可以安全地暴露给外部查询，而不会把缓存/张量泄露出去。
    """

    def __init__(self) -> None:
        self._requests: dict[str, Request] = {}

    def add(self, request: Request) -> None:
        if request.request_id in self._requests:
            raise ValueError(f"request_id 重复: {request.request_id!r}")
        self._requests[request.request_id] = request

    def get(self, request_id: str) -> Request:
        if request_id not in self._requests:
            raise KeyError(f"未知 request_id: {request_id!r}")
        return self._requests[request_id]

    def remove(self, request_id: str) -> None:
        self._requests.pop(request_id, None)

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._requests

    def __len__(self) -> int:
        return len(self._requests)

    def all(self) -> list[Request]:
        return list(self._requests.values())

    def active(self) -> list[Request]:
        """仍在运行（非终态）的请求。"""
        return [r for r in self._requests.values() if not r.is_terminal]

    def count_by_status(self) -> dict[RequestStatus, int]:
        counts = {s: 0 for s in RequestStatus}
        for r in self._requests.values():
            counts[r.status] += 1
        return counts
