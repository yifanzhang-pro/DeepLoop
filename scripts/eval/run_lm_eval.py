"""Run lm-evaluation-harness against a DeepLoop checkpoint.

Small bridge script: load the custom GPT class defined in
`model/gpt-mha-rope-loop.py`, hand the resulting `PreTrainedModel` to
`lm_eval.models.huggingface.HFLM`, and run `simple_evaluate` over the
arxiv 2601.00417 benchmark suite (arc_challenge, arc_easy, hellaswag,
openbookqa, piqa, sciq, social_iqa, winogrande; both 0-shot and 1-shot).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS = (
    "arc_challenge",
    "arc_easy",
    "hellaswag",
    "openbookqa",
    "piqa",
    "sciq",
    "social_iqa",
    "winogrande",
)


def _load_gpt_class():
    """Load model.GPT from `gpt-mha-rope-loop.py` (dash in filename blocks a plain import)."""
    src = REPO_ROOT / "model" / "gpt-mha-rope-loop.py"
    spec = importlib.util.spec_from_file_location("model.gpt_mha_rope_loop", str(src))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.GPT


def _tf_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to a checkpoint-best directory.")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                    help="Comma-separated lm-eval task names.")
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="Debug: cap #examples per task.")
    ap.add_argument("--output_dir", required=True, help="Directory to write results.json into.")
    ap.add_argument("--run_name", default=None, help="Subdirectory label under output_dir.")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(REPO_ROOT))
    # `lm_eval` must be installed separately — see scripts/eval/SETUP.md.

    GPT = _load_gpt_class()
    print(f"[run_lm_eval] loading ckpt: {args.checkpoint}", flush=True)
    model = GPT.from_pretrained(args.checkpoint, dtype=_tf_dtype(args.dtype))
    model.eval()
    # HFLM inherits `device` from an already-initialized model, so we must
    # move the model onto the CUDA device here.
    if args.device != "cpu":
        model = model.to(args.device)
    print(f"[run_lm_eval] model on device: {next(model.parameters()).device}", flush=True)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[run_lm_eval] model loaded: {n_params:.2f}M params, "
          f"block_repeats={getattr(model.config, 'block_repeats', 1)}, "
          f"use_loop_deepnorm={getattr(model.config, 'use_loop_deepnorm', False)}, "
          f"use_input_embed_rmsnorm={getattr(model.config, 'use_input_embed_rmsnorm', False)}",
          flush=True)

    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    # Silence HFLM's `get_model_sha` path: it tries to call HfApi with the
    # model object when we pass a pre-initialized model. Harmless but the
    # exception's `repr(model)` pollutes the log with hundreds of lines.
    def _no_sha(self) -> dict:  # type: ignore[no-untyped-def]
        return {
            "model_num_parameters": sum(p.numel() for p in self._model.parameters()),
            "model_dtype": str(self._model.dtype),
            "model_revision": self.revision,
            "model_sha": "",
        }
    HFLM.get_model_info = _no_sha  # type: ignore[assignment]

    # HFLM wraps a transformers.PreTrainedModel. We pre-moved it to the right
    # device above; HFLM picks up `tokenizer` separately.
    lm = HFLM(
        pretrained=model,
        tokenizer="gpt2",
        batch_size=args.batch_size,
        max_length=int(getattr(model.config, "block_size", 1024)),
        device=args.device,
    )

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"[run_lm_eval] tasks={task_list}  num_fewshot={args.num_fewshot}  bs={args.batch_size}",
          flush=True)

    results = simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    run_name = args.run_name or Path(args.checkpoint).parent.name
    out_dir = Path(args.output_dir) / run_name / f"nfs{args.num_fewshot}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compose a lean table of per-task acc (+ acc_norm when present)
    flat = {}
    for task, metrics in results["results"].items():
        flat[task] = {
            k: v for k, v in metrics.items() if k.startswith(("acc", "acc_norm", "word_perplexity", "byte_perplexity"))
        }
    flat_path = out_dir / "summary.json"
    flat_path.write_text(json.dumps(flat, indent=2))
    print(f"[run_lm_eval] summary -> {flat_path}", flush=True)

    # Also dump the full results (minus non-JSON-serializable config objects)
    full_path = out_dir / "full_results.json"
    serializable = {
        "results": results["results"],
        "configs": {t: results["configs"][t].__dict__ if hasattr(results["configs"].get(t, None), "__dict__") else str(results["configs"].get(t, None)) for t in task_list},
        "versions": results.get("versions", {}),
        "n-shot": {t: args.num_fewshot for t in task_list},
        "checkpoint": str(args.checkpoint),
        "run_name": run_name,
    }
    try:
        full_path.write_text(json.dumps(serializable, indent=2, default=str))
        print(f"[run_lm_eval] full results -> {full_path}", flush=True)
    except Exception as e:
        print(f"[run_lm_eval] (skipping full_results.json: {e})", flush=True)


if __name__ == "__main__":
    main()
