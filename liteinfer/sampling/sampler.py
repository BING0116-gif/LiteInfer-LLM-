"""Sampler：从 logits 中选出 next token（Task 02）。

职责边界（docs/03 模块 3）：上游负责产生 logits，Sampler 只负责
"给定一行 logits，选出一个 token id"。因此本模块不 import 任何模型栈，
输入输出是纯 tensor / int，可以脱离真实模型做秒级单元测试。

流水线顺序与 HF 对齐：temperature -> top-k -> top-p -> softmax -> multinomial。
顺序有讲究：top-k / top-p 的过滤语义都定义在"缩放后的分布"上，先做
temperature 再过滤，才能保证"过滤掉的候选"与"采样分布"是同一个分布。
"""

from __future__ import annotations

from typing import Optional

import torch

from liteinfer.sampling.params import SamplingParams

_NEG_INF = float("-inf")


class Sampler:
    """无状态采样器。

    不持有任何随机源：所有随机性通过外部传入的 ``generator`` 注入。
    这样做的原因：seed 属于"请求级"属性而非"采样器级"属性——并发服务里
    多个请求各自带 seed，共享一个有状态采样器会互相污染随机序列。
    """

    def sample(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        generator: Optional[torch.Generator] = None,
    ) -> int:
        """从一行 logits 中采样一个 token id。

        Args:
            logits: 形状 ``[vocab]`` 的 1D 张量（通常是 forward 输出的最后一步）。
            params: 采样参数。
            generator: 可选随机源；greedy 路径不使用。
        """
        if logits.dim() != 1:
            raise ValueError(
                f"Sampler 期望 1D logits [vocab]，收到 shape={tuple(logits.shape)}；"
                "请在上游先取 logits[batch, -1, :]"
            )
        if params.temperature == 0.0:
            return int(torch.argmax(logits).item())

        # temperature 缩放：T 越小分布越尖。torch.softmax 内部做了 max 减法，
        # T 很小时不会上溢，无需在此手动稳定化
        scaled = logits / params.temperature

        # top-k：只保留分数最高的 k 个候选，其余置 -inf。
        # k >= vocab 时等价于不过滤，直接跳过（避免 topk 的 k 越界报错）
        if params.top_k != -1:
            k = min(params.top_k, logits.shape[-1])
            if k < logits.shape[-1]:
                kth_value = torch.topk(scaled, k).values[-1]
                scaled = scaled.masked_fill(scaled < kth_value, _NEG_INF)

        # top-p（nucleus）：保留累计概率达到 p 的最小 token 集合。
        # 用 cum - probs（即"排在它前面的概率之和"）>= p 来判淘汰，
        # 天然保留恰好跨越阈值的那个 token，不需要 HF 实现里右移一位的修正
        if params.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cum_exclusive = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
            sorted_logits = sorted_logits.masked_fill(
                cum_exclusive >= params.top_p, _NEG_INF
            )
            # scatter 回原 token 顺序，保证返回的 id 与词表对齐
            scaled = torch.full_like(scaled, _NEG_INF).scatter_(
                -1, sorted_idx, sorted_logits
            )

        probs = torch.softmax(scaled, dim=-1)
        # top-p > 0 时首个 token 的 cum_exclusive 恒为 0、top-k 至少留 1 个，
        # 因此走到这里至少有一个非零概率候选，multinomial 不会全零报错
        next_id = torch.multinomial(probs, num_samples=1, generator=generator)
        return int(next_id.item())
