"""Aggregate per-run summary.json files under eval_results/ into a paper
table matching arxiv 2601.00417 (the Deep Delta Learning paper) - same
eight tasks, acc metric, unweighted arithmetic mean across tasks, both
0-shot and 1-shot. Emits Markdown to stdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = (
    ("ARC-C", "arc_challenge"),
    ("ARC-E", "arc_easy"),
    ("HellaSwag", "hellaswag"),
    ("OBQA", "openbookqa"),
    ("PIQA", "piqa"),
    ("SciQ", "sciq"),
    ("SIQA", "social_iqa"),
    ("WG", "winogrande"),
)

MODELS = (
    "base_l1s",
    "base_l3s",
    "base_l5s",
    "dlv1_l1s",
    "dlv1_l3s",
    "dlv1_l5s",
)


def _acc(summary: dict, task: str) -> float | None:
    """Return the harness-default `acc` value for a task, scaled to %."""
    metrics = summary.get(task)
    if not metrics:
        return None
    for k in ("acc,none", "acc"):
        if k in metrics:
            return float(metrics[k]) * 100.0
    return None


def _row_label(name: str) -> str:
    return {
        "base_l1s": "base (R=1)",
        "base_l3s": "base (R=3)",
        "base_l5s": "base (R=5)",
        "dlv1_l1s": "DeepLoopv1 (R=1)",
        "dlv1_l3s": "DeepLoopv1 (R=3)",
        "dlv1_l5s": "DeepLoopv1 (R=5)",
    }.get(name, name)


def format_table(root: Path, num_fewshot: int) -> str:
    head = ["Model"] + [n for n, _ in TASKS] + ["Avg"]
    lines = [
        "| " + " | ".join(head) + " |",
        "| " + " | ".join(["---"] + ["---:"] * len(head[1:])) + " |",
    ]

    for m in MODELS:
        summary_path = root / m / f"nfs{num_fewshot}" / "summary.json"
        if not summary_path.is_file():
            lines.append("| " + _row_label(m) + " | " + " | ".join(["—"] * (len(head) - 1)) + " |")
            continue
        summary = json.loads(summary_path.read_text())
        vals = [_acc(summary, t) for _, t in TASKS]
        defined = [v for v in vals if v is not None]
        avg = sum(defined) / len(defined) if defined else None
        cells = [f"{v:.2f}" if v is not None else "—" for v in vals]
        cells.append(f"{avg:.2f}" if avg is not None else "—")
        lines.append("| " + _row_label(m) + " | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="eval_results",
                    help="Directory containing <model>/nfs<k>/summary.json files.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    print(f"# Downstream evaluation\n")
    print("Per the arxiv 2601.00417 protocol: 8 tasks, metric = `acc` (not `acc_norm`), "
          "unweighted arithmetic mean across tasks, reported in percent.\n")
    for nfs in (0, 1):
        print(f"## {nfs}-shot\n")
        print(format_table(root, nfs))
        print()


if __name__ == "__main__":
    main()
