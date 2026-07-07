"""End-to-end equivalence: _greedy_decode with vs without KV cache must produce identical tokens.

Loads a checkpoint, builds a synthetic prefix in the right ARC-AGI format
([SEP_IN, in_900, SEP_OUT]), greedy-decodes 64 tokens via both code paths, and asserts
the token sequences are identical. Also reports per-step wall-time so we can see the speedup.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors_file

PAD_ID = 0
EOS_ID = 1
DIGIT_OFFSET = 2
SEP_IN_ID = 12
SEP_OUT_ID = 13
GRID_SIZE = 30
GRID_AREA = GRID_SIZE * GRID_SIZE  # 900


def load_model(ckpt_dir: Path, device: torch.device):
    config_dict = json.loads((ckpt_dir / "config.json").read_text())
    model_type = config_dict.get("nanogpt_next_model_type", "gpt-mha-rope-loop")
    mod = importlib.import_module(f"model.{model_type}")
    gpt_config = mod.GPTConfig(**config_dict)
    model = mod.GPT(gpt_config)
    state_dict = load_safetensors_file(str(ckpt_dir / "model.safetensors"), device="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.tie_weights()
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_decode", type=int, default=64, help="Decode this many tokens.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from scripts.eval.run_arc_agi_eval import _greedy_decode  # noqa: E402

    device = torch.device(args.device)
    ckpt_dir = Path(args.checkpoint).resolve()
    if (ckpt_dir / "checkpoint-best").is_dir():
        ckpt_dir = ckpt_dir / "checkpoint-best"
    model = load_model(ckpt_dir, device)

    # Build a synthetic prefix in ARC-AGI format: [SEP_IN, in_900_random, SEP_OUT].
    rng = np.random.default_rng(args.seed)
    in_grid = rng.integers(low=DIGIT_OFFSET, high=DIGIT_OFFSET + 10, size=(args.batch_size, GRID_AREA), dtype=np.int64)
    rows = np.empty((args.batch_size, 1 + GRID_AREA + 1), dtype=np.int64)
    rows[:, 0] = SEP_IN_ID
    rows[:, 1 : 1 + GRID_AREA] = in_grid
    rows[:, 1 + GRID_AREA] = SEP_OUT_ID
    prefix = torch.from_numpy(rows).to(device)
    print(f"prefix shape: {tuple(prefix.shape)}, decode {args.n_decode} tokens, batch={args.batch_size}")

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    out_no_cache = _greedy_decode(model, prefix, n_tokens=args.n_decode, use_kv_cache=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_no = time.time() - t0
    print(f"no-cache decode: {t_no:.2f}s ({args.n_decode / t_no:.2f} tok/s)")

    t0 = time.time()
    out_cache = _greedy_decode(model, prefix, n_tokens=args.n_decode, use_kv_cache=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_kv = time.time() - t0
    print(f"kv-cache decode: {t_kv:.2f}s ({args.n_decode / t_kv:.2f} tok/s)")
    print(f"speedup: {t_no / t_kv:.2f}×")

    matches = (out_no_cache == out_cache).all().item()
    n_diff = (out_no_cache != out_cache).sum().item()
    print(f"token-by-token match: {matches}  (diff count: {n_diff} / {out_no_cache.numel()})")
    if not matches:
        # Find first divergence per row.
        for b in range(out_no_cache.shape[0]):
            row_a = out_no_cache[b].tolist()
            row_b = out_cache[b].tolist()
            for i, (a, c) in enumerate(zip(row_a, row_b)):
                if a != c:
                    print(f"  row {b}: first divergence at step {i}: no-cache={a}, kv-cache={c}")
                    break
        sys.exit(1)
    print("PASS: cached and non-cached decode produce identical token sequences.")


if __name__ == "__main__":
    main()
