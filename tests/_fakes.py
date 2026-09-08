"""Task 09 测试共用的假模型 / 假 tokenizer（不下载任何权重）。

为什么单独成文件：``test_async_engine.py`` 与 ``test_server_api.py`` 都需要
"会真的往 KV 缓存里写"的假模型（Task 08 的教训：忽略 kv_caches 的 FakeLM 测不出块生命周期），
复制两份迟早会漂移。

文件名以 ``_`` 开头，不会被 pytest 当成测试模块收集。
"""

from __future__ import annotations

import time

import torch
from torch import nn

from liteinfer import EngineConfig
from liteinfer.engine import EngineCore
from liteinfer.scheduler.config import SchedulerConfig

VOCAB = 10
NUM_LAYERS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 8


class _FakeAttn:
    """只需让 ``infer_kv_dims`` 能读出 KV 头数与 head_dim。"""

    def __init__(self, num_kv_heads: int, head_dim: int) -> None:
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim


class _FakeLayer(nn.Module):
    def __init__(self, num_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.self_attn = _FakeAttn(num_kv_heads, head_dim)


class _FakeBody(nn.Module):
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [_FakeLayer(num_kv_heads, head_dim) for _ in range(num_layers)]
        )


class FakeLM(nn.Module):
    """forward 会把 K/V 写进 ``kv_caches``，并按输入最后一位产出下一个 token。

    ``nxt = (last + 1) % vocab`` 是**输入 token 的纯函数**：多请求共享同一个模型时
    互不干扰，因此能真正验证并发维护（Task 05 的教训）。
    ``sleep_s`` 用于模拟真实前向的耗时，验证事件循环没有被阻塞。
    """

    def __init__(self, vocab: int = VOCAB, sleep_s: float = 0.0) -> None:
        super().__init__()
        self.vocab = vocab
        self.sleep_s = sleep_s
        self.model = _FakeBody(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM)

    def forward(self, input_ids, position_ids=None, kv_caches=None, write_pos=0):
        if self.sleep_s:
            time.sleep(self.sleep_s)
        batch, seq_len = input_ids.shape
        if kv_caches is not None:
            k = torch.zeros(1, seq_len, NUM_KV_HEADS, HEAD_DIM)
            v = torch.zeros(1, seq_len, NUM_KV_HEADS, HEAD_DIM)
            for cache in kv_caches:
                cache.append(k, v, write_pos)
        logits = torch.full((batch, seq_len, self.vocab), -1e9)
        last = int(input_ids[0, -1].item())
        logits[:, -1, (last + 1) % self.vocab] = 1.0
        return logits


class FakeTokenizer:
    """prompt 是若干数字字符，一个字符 = 一个 token；decode 把 id 拼回数字串。

    没有 ``apply_chat_template``：正好覆盖服务层"无聊天模板时降级拼接"的分支。
    """

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[int(c) for c in text]], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(str(int(i)) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        """只取最后一条消息的内容。

        真实的 Qwen tokenizer 会拼出完整的 `<|im_start|>...` 模板；这里退化成
        "取最后一条 user 内容"，是为了让 chat 链路仍然落在"数字 token"这套可预测的
        语义上（本 tokenizer 只能编码数字字符）。无模板的降级分支由
        ``test_server_api.test_chat_prompt_fallback_without_template`` 单独覆盖。
        """
        return messages[-1]["content"]


def simulate(start: int, max_tokens: int, vocab: int = VOCAB) -> list[int]:
    """假模型在 greedy 下会产出的 token 序列。"""
    seq: list[int] = []
    cur = start
    for _ in range(max_tokens):
        nxt = (cur + 1) % vocab
        seq.append(nxt)
        cur = nxt
    return seq


def make_core(
    max_new_tokens: int = 4,
    *,
    sleep_s: float = 0.0,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 1024,
    block_size: int = 16,
    eos: frozenset[int] | set[int] | None = None,
) -> EngineCore:
    """用假模型拼一个真实的 ``EngineCore``（含 Scheduler + PagedKVCache）。"""
    cfg = EngineConfig(
        device="cpu",
        dtype=torch.float32,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        scheduler=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
    )
    return EngineCore(
        FakeLM(sleep_s=sleep_s), FakeTokenizer(), cfg, eos_ids=frozenset(eos or ())
    )
