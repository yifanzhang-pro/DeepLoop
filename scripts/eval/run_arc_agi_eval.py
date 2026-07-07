"""ARC-AGI evaluation for the DeepLoop causal-LM port of HRM (arXiv:2506.21734).

Loads a trained checkpoint, runs the ARC-AGI evaluation protocol described in
the HRM paper Section 3.2 ("Evaluation Details") on the official
``fchollet/ARC-AGI`` evaluation set, and reports K-shot puzzle accuracy:

  For each test input in the evaluation set:
    1. Generate K augmented variants (random color permutation + one of the
       eight dihedral symmetries).
    2. Greedy autoregressive decoding of the 900 output-grid tokens for each
       variant given the prefix ``[SEP_IN, in_900, SEP_OUT]``.
    3. Apply the inverse augmentation to map every prediction back to the
       original orientation.
    4. Crop each prediction to its EOS-bounded rectangle.
    5. Vote: bucket predictions by hash, sort by count (descending). Top-K
       buckets are the K candidate answers (ARC scoring uses top-2 by default).
  A test pair is "solved" if any of its top-K candidates equals the held-out
  ground-truth output. A puzzle is "solved" if every test pair in it is solved.

Usage:

    python scripts/eval/run_arc_agi_eval.py \\
        --checkpoint output/out_arc_*_jobid_7423564/checkpoint-best \\
        --raw_data_dir /path/to/ARC-AGI/data \\
        --num_puzzles 5 --num_augs 4 --out eval_results/arc_agi_smoke.json

The looped GPT model does NOT support a KV cache, so each generation step does
a full forward pass on the growing prefix. This is the dominant cost; the
script batches all K augmentations of one test pair into a single forward pass
to amortize.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors_file

logger = logging.getLogger(__name__)

# Token IDs must match data/arc-agi/prepare.py.
PAD_ID = 0
EOS_ID = 1
DIGIT_OFFSET = 2
SEP_IN_ID = 12
SEP_OUT_ID = 13
PUZZLE_TOKEN_OFFSET = 14
GRID_SIZE = 30
GRID_AREA = GRID_SIZE * GRID_SIZE  # 900


def _dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    if tid == 0:
        return arr
    if tid == 1:
        return np.rot90(arr, k=1)
    if tid == 2:
        return np.rot90(arr, k=2)
    if tid == 3:
        return np.rot90(arr, k=3)
    if tid == 4:
        return np.fliplr(arr)
    if tid == 5:
        return np.flipud(arr)
    if tid == 6:
        return arr.T
    if tid == 7:
        return np.fliplr(np.rot90(arr, k=1))
    raise ValueError(f"Unknown tid={tid}")


_DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


def _inverse_dihedral(arr: np.ndarray, tid: int) -> np.ndarray:
    return _dihedral_transform(arr, _DIHEDRAL_INVERSE[tid])


def _grid_to_tokens(grid: np.ndarray, pad_r: int = 0, pad_c: int = 0) -> np.ndarray:
    """30x30 token canvas with HRM-style EOS boundary markers (flattened)."""
    nrow, ncol = grid.shape
    canvas = np.pad(
        grid + DIGIT_OFFSET,
        ((pad_r, GRID_SIZE - pad_r - nrow), (pad_c, GRID_SIZE - pad_c - ncol)),
        constant_values=PAD_ID,
    )
    eos_row = pad_r + nrow
    eos_col = pad_c + ncol
    if eos_row < GRID_SIZE:
        canvas[eos_row, pad_c:eos_col] = EOS_ID
    if eos_col < GRID_SIZE:
        canvas[pad_r:eos_row, eos_col] = EOS_ID
    return canvas.flatten()


def _crop(tokens_900: np.ndarray) -> np.ndarray:
    """Find the largest top-left-anchored rectangle of digit cells; return as 0..9 grid.

    Mirrors HRM's eval-time ``crop`` with a pure-numpy implementation: scan
    rows top to bottom, shrinking the column window whenever a non-digit
    appears, and keep the rectangle of maximum area.
    """
    grid = tokens_900.reshape(GRID_SIZE, GRID_SIZE)
    max_area = 0
    max_size = (1, 1)
    num_c = GRID_SIZE
    for num_r in range(1, GRID_SIZE + 1):
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if x < DIGIT_OFFSET or x > DIGIT_OFFSET + 9:
                num_c = c - 1
                break
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)
    if max_area == 0:
        return np.zeros((1, 1), dtype=np.int64)
    return (grid[: max_size[0], : max_size[1]] - DIGIT_OFFSET).astype(np.int64)


@dataclass
class TestPair:
    input: np.ndarray  # uint8 grid of digits 0..9
    output: np.ndarray


@dataclass
class Puzzle:
    name: str
    train: list[TestPair] = field(default_factory=list)
    test: list[TestPair] = field(default_factory=list)


def _load_eval_puzzles(raw_data_dir: str) -> list[Puzzle]:
    """Load ``raw_data_dir/evaluation/*.json`` as Puzzle objects."""
    eval_dir = Path(raw_data_dir) / "evaluation"
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Missing {eval_dir}")
    puzzles: list[Puzzle] = []
    for path in sorted(eval_dir.glob("*.json")):
        data = json.loads(path.read_text())
        p = Puzzle(name=path.stem)
        for ex in data.get("train", []):
            p.train.append(TestPair(np.array(ex["input"], dtype=np.uint8), np.array(ex["output"], dtype=np.uint8)))
        for ex in data.get("test", []):
            p.test.append(TestPair(np.array(ex["input"], dtype=np.uint8), np.array(ex["output"], dtype=np.uint8)))
        puzzles.append(p)
    return puzzles


def _load_model(ckpt_dir: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    """Re-instantiate the GPT model from a checkpoint dir written by train_adam_arc_agi.py."""
    config_dict = json.loads((ckpt_dir / "config.json").read_text())
    model_type = config_dict.get("nanogpt_next_model_type", "gpt-mha-rope-loop")
    mod = importlib.import_module(f"model.{model_type}")
    gpt_config = mod.GPTConfig(**config_dict)
    model = mod.GPT(gpt_config)
    state_dict = load_safetensors_file(str(ckpt_dir / "model.safetensors"), device="cpu")
    load_result = model.load_state_dict(state_dict, strict=False)
    if any(k != "lm_head.weight" for k in load_result.missing_keys):
        raise RuntimeError(f"Unexpected missing keys: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        raise RuntimeError(f"Unexpected extra keys: {load_result.unexpected_keys}")
    model.tie_weights()
    model.to(device).eval()
    return model, config_dict


@torch.no_grad()
def _greedy_decode(
    model: torch.nn.Module,
    prefix: torch.Tensor,
    n_tokens: int,
    *,
    use_kv_cache: bool = True,
) -> torch.Tensor:
    """Greedy AR decoding.

    Two paths controlled by `use_kv_cache`:

      * KV-cache path (default): one prefill pass with `use_cache=True`, then `n_tokens-1`
        single-token decode steps that reuse the cached K/V for every (block, repeat).
        For looped models, the cache stores L*R past_key_values per token.
      * No-cache path: full forward of the growing prefix at every step. ~O(n_tokens²) attn
        memory traffic, kept for correctness sanity-checking against the cached path.

    Args:
        prefix: (B, L) int64 input tokens.
        n_tokens: number of tokens to generate.
        use_kv_cache: if True, use the prefill+decode KV-cache path.

    Returns:
        (B, n_tokens) int64 generated tokens.
    """
    out = torch.empty((prefix.shape[0], n_tokens), dtype=torch.int64, device=prefix.device)
    if not use_kv_cache:
        cur = prefix
        for i in range(n_tokens):
            logits, _ = model(cur)
            if logits.dim() == 3:
                last = logits[:, -1, :]
            else:
                last = logits
            next_tok = last.argmax(dim=-1)
            out[:, i] = next_tok
            cur = torch.cat([cur, next_tok.unsqueeze(-1)], dim=-1)
        return out

    # Prefill — one full forward over the prefix; ask for the cache.
    pre_out = model(idx=prefix, use_cache=True, return_dict=True)
    past_kv = pre_out.past_key_values
    pre_logits = pre_out.logits  # (B, L, V) when use_cache=True (the model returns full-seq logits).
    if pre_logits.dim() == 3:
        last = pre_logits[:, -1, :]
    else:
        last = pre_logits
    next_tok = last.argmax(dim=-1)
    out[:, 0] = next_tok

    # Decode — single-token forwards reusing the past KV.
    for i in range(1, n_tokens):
        step_out = model(
            idx=next_tok.unsqueeze(-1),
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )
        past_kv = step_out.past_key_values
        step_logits = step_out.logits
        if step_logits.dim() == 3:
            step_last = step_logits[:, -1, :]
        else:
            step_last = step_logits
        next_tok = step_last.argmax(dim=-1)
        out[:, i] = next_tok
    return out


def _grid_hash(grid: np.ndarray) -> tuple:
    """Hashable identity of a 2D grid (shape, content)."""
    return (grid.shape, grid.tobytes())


def _sample_augmentation(rng: np.random.Generator) -> tuple[int, np.ndarray]:
    """Draw a (dihedral_id, color_perm 0..9 -> 0..9) augmentation."""
    tid = int(rng.integers(0, 8))
    color_map = np.concatenate(
        [np.zeros(1, dtype=np.uint8), rng.permutation(np.arange(1, 10, dtype=np.uint8))]
    )
    return tid, color_map


def _apply_augmentation(grid: np.ndarray, tid: int, color_map: np.ndarray) -> np.ndarray:
    return _dihedral_transform(color_map[grid], tid)


def _inverse_augmentation(grid: np.ndarray, tid: int, color_map: np.ndarray) -> np.ndarray:
    inv_perm = np.argsort(color_map)
    return inv_perm[_inverse_dihedral(grid, tid)]


def _build_prefix_row(
    *,
    inp: np.ndarray,
    tid: int,
    cmap: np.ndarray,
    puzzle_id_token: int | None,
) -> np.ndarray:
    """Build one ARC-AGI prefix row [maybe puzzle-id, SEP_IN, in_900, SEP_OUT]."""
    aug_inp = _apply_augmentation(inp, tid, cmap)
    tok_in = _grid_to_tokens(aug_inp, pad_r=0, pad_c=0)
    prefix_extra = 1 if puzzle_id_token is not None else 0
    row = np.empty(prefix_extra + 1 + GRID_AREA + 1, dtype=np.int64)
    if puzzle_id_token is not None:
        row[0] = puzzle_id_token
    row[prefix_extra] = SEP_IN_ID
    row[prefix_extra + 1 : prefix_extra + 1 + GRID_AREA] = tok_in
    row[prefix_extra + 1 + GRID_AREA] = SEP_OUT_ID
    return row


def evaluate(
    *,
    checkpoint: str,
    raw_data_dir: str,
    num_puzzles: int | None,
    num_augs: int,
    Ks: list[int],
    out_path: str,
    device_str: str,
    seed: int,
    puzzle_offset: int = 0,
    snapshot_every: int = 10,
    identifier_map_path: str | None = None,
    max_examples_per_test_pair: int | None = None,  # Future hook.
    use_kv_cache: bool = True,
    batch_size: int = 32,
) -> dict:
    """Two-phase eval: (1) batch decode across puzzles, (2) per-puzzle vote/score.

    Phase 1 builds a deterministic queue of (puzzle, test_pair, aug) work items
    (RNG draws happen in the same per-test-pair order as the legacy single-puzzle
    path, so eval results are byte-identical for the same seed). It then chunks
    the queue into groups of `batch_size` rows and runs `_greedy_decode` once
    per chunk — this is the win over the legacy "one decode call per test pair"
    layout, because the per-step decode kernels stop being launch-bound at large
    `batch_size`.

    Phase 2 walks the same (puzzle, test_pair) order as before, looks up the
    decoded outputs from the dict produced by phase 1, and runs the same
    crop/inverse-aug/vote/score logic.

    Multi-GPU sharding via torchrun: when WORLD_SIZE>1, every rank constructs
    the *same* aug-spec sequence (so the seed-determined RNG state is
    bit-identical across ranks) but only decodes the puzzles assigned to its
    rank (`puzzles[RANK::WORLD_SIZE]`). Each rank writes its own JSON to
    `{out_path}.rank{R}.json`; merge them with
    scripts/eval/merge_arc_eval_shards.py to produce the final result.
    """
    del max_examples_per_test_pair

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        # Override device with rank-local GPU; ignore device_str's GPU index.
        device_str = f"cuda:{local_rank}"

    identifier_map: dict[str, int] | None = None
    if identifier_map_path is not None:
        with open(identifier_map_path) as f:
            identifier_map = json.load(f)
        if rank == 0:
            logger.info("Loaded identifier_map (%d puzzles) from %s", len(identifier_map), identifier_map_path)
    device = torch.device(device_str)
    ckpt_dir = Path(checkpoint).resolve()
    if (ckpt_dir / "checkpoint-best").is_dir():
        ckpt_dir = ckpt_dir / "checkpoint-best"
    if rank == 0:
        logger.info("Loading model from %s ...", ckpt_dir)
    model, _config = _load_model(ckpt_dir, device)

    all_puzzles = _load_eval_puzzles(raw_data_dir)
    start = max(0, int(puzzle_offset))
    end = len(all_puzzles) if num_puzzles is None else min(len(all_puzzles), start + int(num_puzzles))
    puzzles = all_puzzles[start:end]
    if world_size > 1:
        my_puzzle_indices = set(range(rank, len(puzzles), world_size))
    else:
        my_puzzle_indices = set(range(len(puzzles)))
    logger.info(
        "[rank %d/%d] Evaluating %d puzzles (offset %d, end %d, of %d total; my slice %d) "
        "with %d augs each, batch_size=%d",
        rank, world_size, len(puzzles), start, end, len(all_puzzles),
        len(my_puzzle_indices), num_augs, batch_size,
    )

    rng = np.random.default_rng(seed)

    # ---- Phase 1: build the work queue (deterministic across all ranks) ----
    # Every rank steps the RNG in the same canonical order so aug specs are
    # identical regardless of WORLD_SIZE. Only this rank's puzzles get materialized.
    work_rows: list[np.ndarray] = []  # prefix rows (each shape (prefix_len,))
    work_index: list[tuple[int, int, int]] = []  # (puz_i, test_idx, aug_idx)
    aug_specs_per_test: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = {}
    for puz_i, puzzle in enumerate(puzzles):
        is_mine = puz_i in my_puzzle_indices
        puzzle_id_token: int | None = None
        if identifier_map is not None:
            if puzzle.name not in identifier_map:
                raise KeyError(
                    f"Puzzle {puzzle.name!r} not in identifier_map; was the model "
                    "trained on a different ARC-AGI raw_data_dir?"
                )
            puzzle_id_token = PUZZLE_TOKEN_OFFSET + int(identifier_map[puzzle.name])
        for test_idx, test_pair in enumerate(puzzle.test):
            specs: list[tuple[int, np.ndarray]] = []
            for aug_idx in range(num_augs):
                tid, cmap = _sample_augmentation(rng)
                specs.append((tid, cmap))
                if is_mine:
                    row = _build_prefix_row(inp=test_pair.input, tid=tid, cmap=cmap, puzzle_id_token=puzzle_id_token)
                    work_rows.append(row)
                    work_index.append((puz_i, test_idx, aug_idx))
            if is_mine:
                aug_specs_per_test[(puz_i, test_idx)] = specs

    n_test_pairs_total = len(aug_specs_per_test)
    logger.info("[rank %d/%d] Built work queue: %d rows across %d test pairs",
                rank, world_size, len(work_rows), n_test_pairs_total)

    # ---- Phase 1b: batched decode ----
    decoded: dict[tuple[int, int, int], np.ndarray] = {}
    t_decode_start = time.time()
    decode_seconds_total = 0.0
    last_log = t_decode_start
    for batch_start in range(0, len(work_rows), batch_size):
        batch_end = min(batch_start + batch_size, len(work_rows))
        batch_rows = np.stack(work_rows[batch_start:batch_end], axis=0)
        prefix = torch.from_numpy(batch_rows).to(device)
        t0 = time.time()
        output = _greedy_decode(model, prefix, n_tokens=GRID_AREA, use_kv_cache=use_kv_cache)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        decode_seconds_total += dt
        out_np = output.cpu().numpy()
        for j, key in enumerate(work_index[batch_start:batch_end]):
            decoded[key] = out_np[j]
        now = time.time()
        if now - last_log > 30.0 or batch_end == len(work_rows):
            elapsed = now - t_decode_start
            done = batch_end
            tok_per_s = (done * GRID_AREA) / max(1e-3, elapsed)
            logger.info(
                "decode %d/%d rows  elapsed %.1fs  throughput %.1f tok/s",
                done, len(work_rows), elapsed, tok_per_s,
            )
            last_log = now

    # ---- Phase 2: per-puzzle scoring + voting ----
    results: dict[str, dict] = {}
    correct_at_k = {k: 0 for k in Ks}
    n_test_pairs = 0
    t_start = t_decode_start  # keep "wall_seconds" semantics: end-to-end including phase 1

    for puz_i, puzzle in enumerate(puzzles):
        if puz_i not in my_puzzle_indices:
            continue  # this rank didn't decode this puzzle
        per_test_results = []
        all_pairs_solved_at_k = {k: True for k in Ks}
        for test_idx, test_pair in enumerate(puzzle.test):
            n_test_pairs += 1
            label = test_pair.output
            specs = aug_specs_per_test[(puz_i, test_idx)]
            pred_grids = []
            for aug_idx, (tid, cmap) in enumerate(specs):
                tokens_900 = decoded[(puz_i, test_idx, aug_idx)]
                cropped = _crop(tokens_900)
                if cropped.size == 0:
                    pred_grids.append(np.zeros((1, 1), dtype=np.int64))
                    continue
                pred_grid = _inverse_augmentation(cropped.astype(np.uint8), tid, cmap)
                pred_grids.append(pred_grid.astype(np.int64))

            counts: dict[tuple, list] = {}
            grid_lookup: dict[tuple, np.ndarray] = {}
            for g in pred_grids:
                h = _grid_hash(g)
                counts.setdefault(h, [0])
                counts[h][0] += 1
                grid_lookup.setdefault(h, g)
            ranked = sorted(counts.items(), key=lambda kv: -kv[1][0])
            label_hash = _grid_hash(label.astype(np.int64))

            per_k = {}
            for k in Ks:
                top_k_hashes = {h for h, _ in ranked[:k]}
                solved = label_hash in top_k_hashes
                per_k[k] = solved
                if not solved:
                    all_pairs_solved_at_k[k] = False
            per_test_results.append(
                {
                    "test_idx": test_idx,
                    "label_shape": list(label.shape),
                    "n_unique_predictions": len(counts),
                    "top_pred_count": ranked[0][1][0] if ranked else 0,
                    "solved_at_k": per_k,
                }
            )

        for k in Ks:
            if all_pairs_solved_at_k[k] and puzzle.test:
                correct_at_k[k] += 1

        results[puzzle.name] = {
            "n_tests": len(puzzle.test),
            "tests": per_test_results,
            "puzzle_solved_at_k": all_pairs_solved_at_k,
        }
        if (puz_i + 1) % 5 == 0 or puz_i + 1 == len(puzzles):
            elapsed = time.time() - t_start
            sols = ", ".join(f"K={k}: {correct_at_k[k]}" for k in Ks)
            logger.info(
                "[%d/%d] elapsed %.0fs solved-so-far {%s} (n_test_pairs=%d)",
                puz_i + 1,
                len(puzzles),
                elapsed,
                sols,
                n_test_pairs,
            )
        if snapshot_every > 0 and (puz_i + 1) % int(snapshot_every) == 0:
            snapshot = {
                "summary": {
                    "checkpoint": str(ckpt_dir),
                    "raw_data_dir": str(raw_data_dir),
                    "num_puzzles_planned": len(puzzles),
                    "num_puzzles_done_this_rank": len(results),
                    "puzzle_offset": start,
                    "num_augs": num_augs,
                    "batch_size": batch_size,
                    "rank": rank,
                    "world_size": world_size,
                    "Ks": Ks,
                    "n_test_pairs_so_far": n_test_pairs,
                    "wall_seconds_so_far": time.time() - t_start,
                    "is_partial": True,
                },
                "puzzles": results,
            }
            shard_path = (
                f"{out_path}.rank{rank}.json" if world_size > 1 else out_path
            )
            Path(shard_path).parent.mkdir(parents=True, exist_ok=True)
            Path(shard_path).write_text(json.dumps(snapshot, indent=2))

    summary = {
        "checkpoint": str(ckpt_dir),
        "raw_data_dir": str(raw_data_dir),
        "num_puzzles_total": len(puzzles),
        "num_puzzles_this_rank": len(my_puzzle_indices),
        "puzzle_offset": start,
        "num_augs": num_augs,
        "batch_size": batch_size,
        "rank": rank,
        "world_size": world_size,
        "n_test_pairs": n_test_pairs,
        "Ks": Ks,
        # The accuracy in a shard is over this rank's slice; merge across ranks
        # via scripts/eval/merge_arc_eval_shards.py for the final number.
        "shard_correct_at_k": correct_at_k,
        "shard_puzzle_accuracy_at_k": {
            k: correct_at_k[k] / max(1, len(my_puzzle_indices)) for k in Ks
        },
        "wall_seconds": time.time() - t_start,
        "decode_seconds": decode_seconds_total,
        "use_kv_cache": use_kv_cache,
        "is_partial": False,
    }
    final = {"summary": summary, "puzzles": results}
    shard_path = f"{out_path}.rank{rank}.json" if world_size > 1 else out_path
    Path(shard_path).parent.mkdir(parents=True, exist_ok=True)
    Path(shard_path).write_text(json.dumps(final, indent=2))
    logger.info(
        "[rank %d/%d] Wrote %s. Shard accuracy (over %d puzzles): %s",
        rank, world_size, shard_path, len(my_puzzle_indices),
        ", ".join(
            f"K={k}: {summary['shard_puzzle_accuracy_at_k'][k] * 100:.2f}%" for k in Ks
        ),
    )
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint-best dir or its parent.")
    parser.add_argument(
        "--raw_data_dir",
        default=os.environ.get(
            "ARC_RAW_DIR",
            "/path/to/idea_den/arc_agi_ref/ARC-AGI/data",
        ),
    )
    parser.add_argument("--num_puzzles", type=int, default=None, help="Cap puzzle count for smokes / slicing.")
    parser.add_argument("--puzzle_offset", type=int, default=0, help="Skip the first N puzzles (for slicing across ranks).")
    parser.add_argument("--num_augs", type=int, default=8)
    parser.add_argument("--Ks", default="1,2", help="Comma-separated top-K values.")
    parser.add_argument("--out", default="eval_results/arc_agi/eval.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snapshot_every", type=int, default=10, help="Persist partial JSON every N puzzles (0 to disable).")
    parser.add_argument(
        "--identifier_map",
        default=None,
        help=(
            "Path to identifiers.json (output of data/arc-agi/prepare.py "
            "--with_puzzle_id). When provided, every eval row is prefixed with "
            "the puzzle's id token (PUZZLE_TOKEN_OFFSET + id). Required when "
            "evaluating a checkpoint trained on the puzzle-id-augmented split."
        ),
    )
    parser.add_argument(
        "--no_kv_cache",
        action="store_true",
        help=(
            "Disable the KV-cache decode path (slow O(n^2) full-prefix forward at every step). "
            "Used as a sanity check that the cached path matches the slow path."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help=(
            "Number of (puzzle, test_pair, aug) rows decoded together. The legacy single-puzzle "
            "layout uses batch_size = num_augs and processes one test pair at a time; this flag "
            "lets a single batch span multiple puzzles, which the per-step decode kernels need "
            "in order to stop being launch-bound on H100/H200."
        ),
    )
    args = parser.parse_args()

    Ks = [int(k) for k in args.Ks.split(",") if k.strip()]
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root for `model.*`
    evaluate(
        checkpoint=args.checkpoint,
        raw_data_dir=args.raw_data_dir,
        num_puzzles=args.num_puzzles,
        num_augs=args.num_augs,
        Ks=Ks,
        out_path=args.out,
        device_str=args.device,
        seed=args.seed,
        puzzle_offset=args.puzzle_offset,
        snapshot_every=args.snapshot_every,
        identifier_map_path=args.identifier_map,
        use_kv_cache=not args.no_kv_cache,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
