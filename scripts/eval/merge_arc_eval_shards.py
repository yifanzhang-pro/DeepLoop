"""Merge per-rank ARC-AGI eval JSON shards into a single combined result.

Each rank of the multi-GPU eval (run_arc_agi_eval.py) writes
`{out}.rank{R}.json` containing its slice of the puzzles. This tool reads all
shards, validates that puzzle assignments are disjoint, sums solve counts
across shards, and writes a combined `{out}` JSON with the same schema as a
single-GPU run.

Usage:

    python scripts/eval/merge_arc_eval_shards.py \\
        --out eval_results/arc_agi/arc1c_aug1000.json \\
        --world_size 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Final combined JSON path (the same path you passed to run_arc_agi_eval.py).")
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--keep_shards", action="store_true", help="Keep the per-rank .rank{R}.json files (default: delete after successful merge).")
    args = parser.parse_args()

    base = Path(args.out)
    shards = []
    for r in range(args.world_size):
        p = Path(f"{args.out}.rank{r}.json")
        if not p.exists():
            print(f"FATAL: missing shard {p}", file=sys.stderr)
            sys.exit(1)
        shards.append((p, json.loads(p.read_text())))

    # Validate: same checkpoint, same num_augs, same Ks, disjoint puzzle sets.
    ref_summary = shards[0][1]["summary"]
    keys_to_match = ["checkpoint", "raw_data_dir", "num_puzzles_total", "num_augs", "Ks", "use_kv_cache"]
    for p, data in shards[1:]:
        for k in keys_to_match:
            if data["summary"][k] != ref_summary[k]:
                print(f"FATAL: shard {p} disagrees on {k!r}: {data['summary'][k]} vs {ref_summary[k]}", file=sys.stderr)
                sys.exit(1)

    seen_puzzles: set[str] = set()
    combined_puzzles: dict[str, dict] = {}
    total_correct_at_k = {k: 0 for k in ref_summary["Ks"]}
    total_n_test_pairs = 0
    total_decode_seconds = 0.0
    max_wall_seconds = 0.0
    for p, data in shards:
        for name, pz in data["puzzles"].items():
            if name in seen_puzzles:
                print(f"FATAL: puzzle {name!r} present in multiple shards", file=sys.stderr)
                sys.exit(1)
            seen_puzzles.add(name)
            combined_puzzles[name] = pz
        s = data["summary"]
        for k in s["Ks"]:
            total_correct_at_k[k] += int(s["shard_correct_at_k"][str(k)] if str(k) in s["shard_correct_at_k"] else s["shard_correct_at_k"][k])
        total_n_test_pairs += int(s["n_test_pairs"])
        total_decode_seconds += float(s["decode_seconds"])
        max_wall_seconds = max(max_wall_seconds, float(s["wall_seconds"]))

    if len(combined_puzzles) != int(ref_summary["num_puzzles_total"]):
        print(
            f"WARNING: combined puzzles ({len(combined_puzzles)}) != "
            f"num_puzzles_total ({ref_summary['num_puzzles_total']}) — some ranks may have produced no puzzles",
            file=sys.stderr,
        )

    Ks = ref_summary["Ks"]
    final_summary = {
        "checkpoint": ref_summary["checkpoint"],
        "raw_data_dir": ref_summary["raw_data_dir"],
        "num_puzzles": int(ref_summary["num_puzzles_total"]),
        "num_puzzles_combined": len(combined_puzzles),
        "puzzle_offset": ref_summary.get("puzzle_offset", 0),
        "num_augs": ref_summary["num_augs"],
        "batch_size": ref_summary.get("batch_size"),
        "world_size": int(args.world_size),
        "n_test_pairs": total_n_test_pairs,
        "Ks": Ks,
        "puzzle_accuracy_at_k": {
            k: total_correct_at_k[k] / max(1, len(combined_puzzles)) for k in Ks
        },
        "wall_seconds": max_wall_seconds,  # max across ranks — i.e., the slowest rank
        "decode_seconds_total_across_ranks": total_decode_seconds,
        "use_kv_cache": ref_summary["use_kv_cache"],
        "is_partial": False,
    }
    final = {"summary": final_summary, "puzzles": combined_puzzles}

    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text(json.dumps(final, indent=2))
    print(f"Wrote combined: {base}")
    print(f"Puzzle accuracy: {', '.join(f'K={k}: {final_summary['puzzle_accuracy_at_k'][k] * 100:.2f}%' for k in Ks)}")
    print(f"({len(combined_puzzles)} puzzles, {total_n_test_pairs} test pairs, max wall {max_wall_seconds:.0f}s)")

    if not args.keep_shards:
        for p, _ in shards:
            p.unlink()
            print(f"  removed {p}")


if __name__ == "__main__":
    main()
