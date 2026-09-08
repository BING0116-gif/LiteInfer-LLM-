"""Task 06 最小运行示例：在调度器限制下观察逐步准入（连续批处理）。

用「假模型」驱动（不下载权重，CPU 上即时跑完），清晰展示：
1. 一次性提交 12 个请求，但序列预算只放行 3 个并发；
2. 每个 step 打印本轮被准入 / 在飞 / 排队的数量，直观看到 waiting -> running -> 终态
   的动态流转；
3. 在飞请求按 max_tokens 结束后立即释放 slot，waiting 队首 FCFS 补位——这就是
   continuous batching 相对静态批「一个走完整批干等」的本质区别。

运行方式（仓库根目录）：

    set PYTHONPATH=d:\\LiteInfer
    set "HF_HOME=D:\\LiteInfer\\hf_cache"
    python examples/scheduler_demo.py

注意：本例用 FakeLM（纯函数，无网络/无权重）只为即时演示调度行为；真实模型路径见
tests 中 marker=model 的逐字一致性测试。
"""

from __future__ import annotations

import torch

from liteinfer import EngineConfig, EngineCore
from liteinfer.sampling.params import SamplingParams
from liteinfer.scheduler.config import SchedulerConfig


class FakeLM(torch.nn.Module):
    """next = (last + 1) % 10 的纯函数模型，便于无权重即时演示。"""

    def __init__(self, vocab: int = 10) -> None:
        super().__init__()
        self.vocab = vocab
        layer = torch.nn.Module()
        layer.self_attn = type("A", (), {"num_kv_heads": 2, "head_dim": 8})()
        body = torch.nn.Module()
        body.layers = [layer]
        self.model = body

    def forward(self, input_ids, position_ids=None, kv_caches=None, write_pos=0):
        b, s = input_ids.shape
        logits = torch.full((b, s, self.vocab), -1e9)
        last = int(input_ids[0, -1].item())
        logits[:, -1, (last + 1) % self.vocab] = 1.0
        return logits


class FakeTokenizer:
    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[int(text)]], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(str(int(i)) for i in ids)


def main() -> None:
    # 序列预算=3：最多 3 个并发在飞；token budget 足够大（不限制 prefill）
    sched = SchedulerConfig(max_num_seqs=3, max_num_batched_tokens=4096)
    cfg = EngineConfig(device="cpu", dtype=torch.float32, max_new_tokens=4, scheduler=sched)

    engine = EngineCore(FakeLM(), FakeTokenizer(), cfg, eos_ids=frozenset())
    params = SamplingParams(max_tokens=4, temperature=0.0)

    n = 12
    starts = [str(i) for i in range(n)]
    rids = [engine.submit(s, params) for s in starts]
    print(f"[submit] 提交 {n} 个请求；waiting={engine.scheduler.num_waiting}, running={engine.scheduler.num_running}")

    print("\nstep | admitted | running | waiting | finished")
    print("-----+----------+---------+---------+---------")
    step = 0
    while engine.active_requests():
        before = engine.scheduler.num_running
        engine.step()
        step += 1
        finished = n - len(engine.active_requests())
        print(
            f"{step:>4} | {before:>8} | {engine.scheduler.num_running:>7} | "
            f"{engine.scheduler.num_waiting:>7} | {finished:>8}"
        )

    print("\n[check] 所有请求均正确结束：")
    ok = all(
        engine.get_request(rid).generated == [(int(s) + 1 + k) % 10 for k in range(4)]
        for rid, s in zip(rids, starts)
    )
    print(f"  正确性: {ok}")
    print("[note] running 列始终 <= max_num_seqs=3，证明序列预算真正生效；")
    print("        waiting 随 running 释放 slot 而逐步清空，即连续批处理动态补位。")


if __name__ == "__main__":
    main()
