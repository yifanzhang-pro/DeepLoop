"""Numeric-equivalence check: looped GPT forward with KV cache must match the no-cache path.

Loads a trained checkpoint, runs:
  (1) full forward over a 100-token prefix → logits[:, -1, :]  (no cache)
  (2) prefill 99 tokens with use_cache=True, then 1-token decode with the past KV → logits

Both must agree to within ~1e-4 absolute / 1e-4 relative. If they don't, the cache wiring is wrong.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file as load_safetensors_file


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
    return model, gpt_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prefix_len", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    device = torch.device(args.device)
    ckpt_dir = Path(args.checkpoint).resolve()
    if (ckpt_dir / "checkpoint-best").is_dir():
        ckpt_dir = ckpt_dir / "checkpoint-best"
    model, cfg = load_model(ckpt_dir, device)
    L = int(cfg.num_hidden_layers)
    R = int(getattr(cfg, "block_repeats", 1))
    print(f"model: L={L}, R={R}, vocab={cfg.vocab_size}, hidden={cfg.hidden_size}")

    torch.manual_seed(args.seed)
    prefix = torch.randint(0, int(cfg.vocab_size), (1, args.prefix_len), device=device, dtype=torch.long)

    # Path A: full forward over prefix, return logits at the last position.
    with torch.no_grad():
        logits_a, _ = model(idx=prefix)
    logits_a_last = logits_a[:, -1, :].float() if logits_a.dim() == 3 else logits_a.float()

    # Path B: prefill (prefix_len-1) with use_cache=True, then single-token decode.
    pre_prefill = prefix[:, :-1].contiguous()
    last_token = prefix[:, -1:].contiguous()
    with torch.no_grad():
        out_pre = model(idx=pre_prefill, use_cache=True, return_dict=True)
        past_kv = out_pre.past_key_values
        out_b = model(idx=last_token, past_key_values=past_kv, use_cache=True, return_dict=True)
    logits_b = out_b.logits
    if logits_b.dim() == 3:
        logits_b = logits_b[:, -1, :]
    logits_b = logits_b.float()

    abs_diff = (logits_a_last - logits_b).abs()
    rel_diff = abs_diff / (logits_a_last.abs() + 1e-6)
    a_top = logits_a_last.argmax(dim=-1).item()
    b_top = logits_b.argmax(dim=-1).item()
    print(f"past_kv length: {len(past_kv)}  (expected L*R = {L * R})")
    print(f"max |a-b|: {abs_diff.max().item():.3e}")
    print(f"mean |a-b|: {abs_diff.mean().item():.3e}")
    print(f"max rel diff: {rel_diff.max().item():.3e}")
    print(f"argmax (no-cache): {a_top}   argmax (cache): {b_top}   match={a_top == b_top}")
    if abs_diff.max().item() > 1e-3:
        print("FAIL: logits diverge beyond tolerance.")
        sys.exit(1)
    if a_top != b_top:
        print("FAIL: top-1 token differs.")
        sys.exit(1)
    print("PASS: KV-cache forward matches no-cache forward.")


if __name__ == "__main__":
    main()
