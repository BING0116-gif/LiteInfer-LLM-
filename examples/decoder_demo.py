"""Task 03 最小运行示例：MinimalQwen 加载真实权重并做 greedy 生成。

运行方式（仓库根目录）：
    set PYTHONPATH=d:\\LiteInfer
    python examples/decoder_demo.py

展示三件事：
1. 从 HF checkpoint 权重构建 MinimalQwen（strict 加载）；
2. last-token logits 与 HF 对齐（allclose + top-1 agreement）；
3. 用 MinimalQwen 的 forward 做一个玩具 greedy 循环（无 KV Cache，
   每步全序列 forward——生产生成链仍是 Task 02 的 ManualGenerator）。
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer

from liteinfer import EngineConfig
from liteinfer.model.alignment import alignment_tolerances, top1_agreement
from liteinfer.model.minimal import load_minimal_from_hf

PROMPT = "The capital of France is"
N_STEPS = 8


def main() -> None:
    # CPU + FP32：本机环境硬性约束，dtype/device 只从 EngineConfig 进入
    cfg = EngineConfig(device="cpu", dtype=torch.float32, max_new_tokens=N_STEPS)
    loaded = load_minimal_from_hf(cfg)
    minimal, hf_model = loaded.minimal, loaded.hf_model
    atol, rtol = alignment_tolerances(cfg.device)

    cache = str(cfg.resolved_hf_cache_dir() / "hub")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id, cache_dir=cache, local_files_only=cfg.local_files_only
    )

    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    with torch.inference_mode():
        mine = minimal(input_ids)
        hf = hf_model(input_ids).logits
    max_diff = (mine - hf).abs().max().item()
    print(f"[logits] shape={tuple(mine.shape)} max|diff|={max_diff:.3e} "
          f"(atol={atol}, rtol={rtol})")
    print(f"[logits] top-1 agreement = {top1_agreement(mine, hf):.4f}")

    # 玩具 greedy 循环：EOS 出现即停（与 Task 02 的 EOS 语义一致）
    eos_ids = {tokenizer.eos_token_id, 151645}  # <|im_end|> 是 Qwen2.5 的生成终止符
    ids = input_ids.clone()
    generated: list[int] = []
    with torch.inference_mode():
        for _ in range(N_STEPS):
            next_id = minimal.greedy_next_token(ids)
            if next_id in eos_ids:
                break
            generated.append(next_id)
            ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"[greedy] prompt = {PROMPT!r}")
    print(f"[greedy] output = {text!r}  ({len(generated)} tokens)")


if __name__ == "__main__":
    main()
