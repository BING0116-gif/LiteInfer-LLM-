"""Task 06：Scheduler 配置。

集中存放调度预算参数，便于 Task 05 约定的「Scheduler 参数集中到 EngineConfig」。
本文件是纯 dataclass，**不依赖 torch 也不依赖引擎其他模块**，避免与 config.py 形成
循环导入（config.py 会 import 本文件来给 EngineConfig 提供默认值）。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerConfig:
    """连续批处理调度器的预算闸门。

    - ``max_num_seqs``：并发在飞序列数上限（sequence budget）。
      典型取 16~256；太小吞吐受限，太大显存/调度开销上升。
    - ``max_num_batched_tokens``：单步（一次 schedule）可调度 token 总量上限
      （token budget）。prefill 一个请求按其 ``prompt_len`` 计费，decode 一步只
      计 1（每请求每步只新产 1 个 token）。这是 vLLM 风格调度的核心闸门：
      它限制了「单步内最多 prefill 多少 token」，从而把长 prompt 与在飞 decode
      解耦，避免一个超长 prompt 占满整步、饿死其它请求。
    """

    max_num_seqs: int = 16
    max_num_batched_tokens: int = 2048
