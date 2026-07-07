"""Port of arc_eval.ipynb to a CLI script — the paper voting protocol.

This is the canonical post-processing step that turns evaluate.py's saved
predictions into the K=1 / K=2 / K=10 / K=100 / K=1000 numbers HRM reports
in the paper. Reproduces the notebook exactly:

  - Crop each prediction to the maximum-sized EOS-free rectangle and hash
    the cropped digit grid (NOT the full 30x30 token frame).
  - Group predictions by (puzzle_name, input_hash) — per test-pair, not per
    puzzle, because puzzles can have multiple test pairs.
  - Rank unique canonical predictions by mean sigmoid(q_halt_logits) per
    canonical (NOT raw vote count).
  - A puzzle is "correct under K=k" iff every test pair has the ground
    truth among its top-k ranked canonical answers.

Run from HRM_DIR. Assumes evaluate.py has already saved
`<checkpoint>_all_preds.<rank>` files.

Usage:
    python arc_eval_script.py \
        --data data/arc-aug-1000 \
        --checkpoint "checkpoints/.../step_<N>"
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob

import numpy as np
import torch
from numba import njit

from dataset.common import inverse_dihedral_transform

PAD_PUZZLE_IDENTIFIER = 0


def load_identifiers_and_preds(dataset_path: str, checkpoint_path: str):
    with open(os.path.join(dataset_path, "identifiers.json"), "r") as f:
        identifier_map = json.load(f)

    # evaluate.py writes preds to  {dirname}/step_{step}_all_preds.{rank},
    # where `step` is parsed from the checkpoint *filename* only when it
    # starts with "step_" (else step=0). The published checkpoint is named
    # plain "checkpoint", so its preds land as step_0_all_preds.* — NOT
    # checkpoint_all_preds.*. Reconstruct the exact path evaluate.py uses so
    # this works for both the published checkpoint and our own step_<N> ones.
    ckpt_dir = os.path.dirname(checkpoint_path)
    ckpt_name = os.path.basename(checkpoint_path)
    step = int(ckpt_name.removeprefix("step_")) if ckpt_name.startswith("step_") else 0

    all_preds: dict = {}
    pred_files = sorted(glob(os.path.join(ckpt_dir, f"step_{step}_all_preds.*")))
    if not pred_files:
        # Backwards-compat with the notebook convention / older runs.
        pred_files = sorted(glob(f"{checkpoint_path}_all_preds.*"))
    if not pred_files:
        available = sorted(glob(os.path.join(ckpt_dir, "*_all_preds.*")))
        raise FileNotFoundError(
            f"No prediction files for checkpoint '{checkpoint_path}'. Looked for "
            f"{os.path.join(ckpt_dir, f'step_{step}_all_preds.*')} and "
            f"{checkpoint_path}_all_preds.*. "
            f"Available *_all_preds.* in {ckpt_dir or '.'}: {available or 'none'}. "
            "Run evaluate.py first to produce them."
        )
    for filename in pred_files:
        preds = torch.load(filename, weights_only=False)
        for k, v in preds.items():
            all_preds.setdefault(k, [])
            all_preds[k].append(v)
        del preds

    all_preds = {k: torch.cat(v, dim=0) for k, v in all_preds.items()}

    mask = all_preds["puzzle_identifiers"] != PAD_PUZZLE_IDENTIFIER
    all_preds = {k: v[mask] for k, v in all_preds.items()}

    return identifier_map, all_preds


def inverse_aug(name: str, grid: np.ndarray) -> np.ndarray:
    if "_" not in name:
        return grid
    trans_id, perm = name.split("_")[-2:]
    trans_id = int(trans_id[1:])
    inv_perm = np.argsort(list(perm))
    return inv_perm[inverse_dihedral_transform(grid, trans_id)]


def grid_hash(grid: np.ndarray) -> int:
    return hash((grid.tobytes(), grid.shape))


@njit
def crop(grid: np.ndarray) -> np.ndarray:
    grid = grid.reshape(30, 30)
    max_area = 0
    max_size = (0, 0)
    nr, nc = grid.shape
    num_c = nc
    for num_r in range(1, nr + 1):
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) | (x > 11):
                num_c = c - 1
                break
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)
    return grid[: max_size[0], : max_size[1]] - 2


def test(data_path: str, checkpoint_path: str, Ks=(1, 2, 10, 100, 1000)) -> None:
    identifier_map, all_preds = load_identifiers_and_preds(data_path, checkpoint_path)

    puzzle_labels: dict = {}
    for identifier, input, label in zip(
        all_preds["puzzle_identifiers"], all_preds["inputs"], all_preds["labels"]
    ):
        name = identifier_map[int(identifier)]
        if "_" not in name:
            puzzle_labels.setdefault(name, {})
            input_c = crop(input.numpy())
            label_c = crop(label.numpy())
            input_hash = grid_hash(input_c)
            label_hash = grid_hash(label_c)
            # ARC-AGI-1 eval JSONs contain exact-duplicate test pairs (same
            # input AND label), so the original `assert input_hash not in ...`
            # is too strict for a1 and aborts the eval. Dedup idempotently;
            # only error on a GENUINE conflict (same input -> different label).
            if input_hash in puzzle_labels[name]:
                assert puzzle_labels[name][input_hash] == label_hash, (
                    f"conflicting label for {name}"
                )
                continue
            puzzle_labels[name][input_hash] = label_hash

    print(f"Number of puzzles: {len(puzzle_labels)}")

    preds = all_preds["logits"].argmax(-1)
    q_halts = all_preds["q_halt_logits"].float().sigmoid()

    pred_answers: dict = {}
    skipped_no_gt = 0
    skipped_no_input_hash = 0
    for identifier, input, pred, q in zip(
        all_preds["puzzle_identifiers"], all_preds["inputs"], preds, q_halts
    ):
        name = identifier_map[int(identifier)]
        orig_name = name.split("_")[0]
        if orig_name not in puzzle_labels:
            skipped_no_gt += 1
            continue
        input_np = input.numpy()
        input_hash = grid_hash(inverse_aug(name, crop(input_np)))
        if input_hash not in puzzle_labels[orig_name]:
            skipped_no_input_hash += 1
            continue
        pred_canon = inverse_aug(name, crop(pred.numpy()))
        pred_hash = grid_hash(pred_canon)
        pred_answers.setdefault(orig_name, {})
        pred_answers[orig_name].setdefault(input_hash, [])
        pred_answers[orig_name][input_hash].append((pred_hash, float(q.item())))

    if skipped_no_gt or skipped_no_input_hash:
        print(
            f"  (skipped {skipped_no_gt} aug preds with no base-puzzle GT, "
            f"{skipped_no_input_hash} aug preds with no input-hash match)"
        )

    correct = [0 for _ in Ks]
    for name, tests in puzzle_labels.items():
        num_test_correct = [0 for _ in Ks]
        for input_hash, label_hash in tests.items():
            if name not in pred_answers or input_hash not in pred_answers[name]:
                continue
            p = pred_answers[name][input_hash]
            p_map: dict = {}
            for h, q in p:
                p_map.setdefault(h, [0, 0.0])
                p_map[h][0] += 1
                p_map[h][1] += q
            for h, stats in p_map.items():
                stats[1] /= stats[0]
            # Rank by (vote_count, mean_q) descending — matches arc_eval.ipynb,
            # which sorts on the whole [count, mean_q] list: count first, mean
            # sigmoid(q_halt) only as a tiebreak. Ranking by q_halt ALONE (the
            # previous bug here) buries the majority answer and yields
            # K=1=K=2=0% on the published checkpoint vs the paper's ~5%.
            ranked = sorted(p_map.items(), key=lambda kv: (kv[1][0], kv[1][1]), reverse=True)
            for i, k in enumerate(Ks):
                ok = any(h == label_hash for h, _ in ranked[:k])
                num_test_correct[i] += ok
        for i in range(len(Ks)):
            correct[i] += num_test_correct[i] == len(tests)

    n_puzzles = len(puzzle_labels)
    print()
    print(f"{'K':>4}  {'accuracy':>10}  {'count':>10}")
    for i, k in enumerate(Ks):
        print(f"{k:>4}  {correct[i] / n_puzzles * 100:>9.2f}%  {correct[i]:>4}/{n_puzzles}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Dataset dir (contains identifiers.json)")
    p.add_argument("--checkpoint", required=True, help="Path to step_<N> checkpoint file")
    args = p.parse_args()
    test(args.data, args.checkpoint)
