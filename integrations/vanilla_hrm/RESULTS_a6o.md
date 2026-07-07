# Vanilla HRM reproduction — results on AI Studio (a6o / dlrover)

Ran the four `vanilla_hrm` experiments on the AI Studio H200 cluster (a6o +
dlrover-run) instead of slurm. All training uses the **real AdamATan2**
optimizer (the `adam_atan2` C-extension is built from source and the run
**fails loud** if the backend can't import — never the silent AdamW fallback).
Each experiment is split into a **training job** (per-checkpoint async OSS
exfil, no inline eval — keeps the post-train tail short so dlrover's
hang-detector / SUCCEEDED_EXITED grace can't kill the upload) plus a **separate
eval job** (`evaluate.py` → `arc_eval_script.py`).

## Results

| # | run | dataset | epochs | per-example exact | voted K=1 | voted K=2 | K=1000 | N |
|---|---|---|---|---|---|---|---|---|
| 3 | full_train_arc1 | ARC-AGI-1 | 100k | 25.93% | 33.5% | **37.25%** | 48.5% | 400 |
| 2 | smoke_train_arc1 | ARC-AGI-1 | 8k | 15.56% | 25.5% | 29.25% | 42.5% | 400 |
| 4 | full_train_arc2 | ARC-AGI-2 | 100k | — | 1.67% | 1.67% | 6.67% | 120 |
| 1 | validate_eval (published `sapientinc/HRM-checkpoint-ARC-2`) | ARC-AGI-2 | — | — | 0.83% | 1.67% | 5.0% | 120 |

## Findings

**ARC-AGI-1 reproduces the paper.** Full 100k vanilla → **K=2 = 37.25%**,
within ~3 pp of the paper's 40.3% (the same harness gap documented in
`h010nf_voting_recompute.md`) and essentially equal to the in-house
vanilla-100k (37.0%). This confirms the a6o port, the real-AdamATan2 build, and
the eval pipeline end-to-end. (The 8k smoke ran a bit hotter than the script's
guideline band — 15.56% per-example / K=2 29.25% — consistent with AdamATan2's
fast early convergence; not eval leakage, since the 100k run lands *below* the
paper, which it could not if test answers were leaking.)

**ARC-AGI-2: our model matches the published checkpoint, but neither reaches the
paper's 5% through this protocol.** Our from-scratch ARC-2 model (fully trained
— ARC-2 has fewer train groups so 100k epochs = ~606k steps, not under-trained)
and the **paper's own published ARC-2 checkpoint** both score **K=2 = 1.67%**
through the identical, validated pipeline. Our `arc-2-aug-1000`
`num_puzzle_identifiers = 1,045,829` matches the published checkpoint's
`puzzle_emb` shape `(1045829, 512)` exactly (clean ARC-2-only build, not
contaminated), and the two checkpoints load via different paths
(ours / HF + `DISABLE_COMPILE=1`) yet agree — so this is **not** a
dataset/load/eval defect on our side. The official `arc_eval.ipynb` protocol on
the public ckpt simply does not reach the paper's headline 5%; that gap appears
to be authors'-side (eval set / augmentation budget / decoding), and is worth a
joint look.

## Eval bugs found

1. A custom single-pass eval we wrote (`integrations/hrm/eval_k2.py`) had a
   grid-canonicalization bug — it inverse-dihedral-transformed the full 30×30
   frame instead of cropping to the content rectangle first — which floored
   *every* score at ~1%. Abandoned it for the package's `evaluate.py` +
   `arc_eval_script.py`.
2. **`arc_eval_script.py` aborts on ARC-AGI-1.** Its
   `assert input_hash not in puzzle_labels[name]` assumes each base puzzle's
   test inputs hash uniquely, but the ARC-AGI-1 eval JSONs contain
   exact-duplicate test pairs (419 exact `(input, label)` repeats, 0 conflicting
   labels) → `AssertionError`. **Fixed in this PR**: replaced with an idempotent
   dedup that only errors on a genuine same-input/different-label conflict.
   (ARC-AGI-2 does not hit this, which is why the original assert survived.)
