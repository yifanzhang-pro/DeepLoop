#!/bin/bash
# ============================================================================
# EXPERIMENT 2 — short vanilla training on ARC-AGI-1 to validate the
# training pipeline end-to-end BEFORE committing 22 h of compute.
#
# What it does:
#   1. Trains vanilla HRM (arch=hrm_v1) on ARC-AGI-1 for 8000 epochs at the
#      paper's default config — exactly the command from the HRM README,
#      with only `epochs` and `eval_interval` overridden for speed.
#   2. After training finishes, runs evaluate.py + arc_eval_script.py on the
#      final checkpoint.
#
# Hardware:  8 H100/H200 GPUs
# Wall time: ~2 hours of training + ~30 min eval
#
# Expected result:
#   At 8000 epochs the model is far from converged; the per-example
#   exact_accuracy (printed by HRM's eval) should be ~ 9-11 %, matching the
#   8k vanilla baseline reported elsewhere (K1 ≈ 9.44 %). The voted K=2
#   from arc_eval_script.py will be small (~5-10 %); that's expected at
#   this scale, the curve only takes off after 50k+ epochs.
#
#   If exact_accuracy is < 5 % or > 15 % at 8k epochs, something in the
#   training pipeline is off — fix before running experiment 3.
# ============================================================================

set -euo pipefail

HRM_DIR="${HRM_DIR:?set HRM_DIR}"
PYTHON="${PYTHON:?set PYTHON}"
DATA_PATH="data/arc-aug-1000"
source "$(dirname "$0")/_preflight.sh"

cd "$HRM_DIR"

# Train.
echo "[1/3] Training vanilla HRM (arch=hrm_v1) on ARC-AGI-1, 8000 epochs ..."
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    pretrain.py epochs=8000 eval_interval=1000

# Discover the run directory (HRM picks a random coolname per run).
RUN_DIR=$(ls -dt "checkpoints/Arc-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 "* 2>/dev/null | head -1)
if [[ -z "$RUN_DIR" ]]; then
    echo "ERROR: cannot find the new run directory under checkpoints/. Aborting."
    exit 1
fi
echo "Run dir: $RUN_DIR"

# Pick the final checkpoint.
LATEST_STEP=$(ls "$RUN_DIR" | grep -E "^step_[0-9]+$" | sed 's/step_//' | sort -n | tail -1)
CKPT_FILE="$RUN_DIR/step_$LATEST_STEP"
echo "Checkpoint: $CKPT_FILE"

# evaluate.py
echo "[2/3] Running evaluate.py ..."
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
    evaluate.py checkpoint="$CKPT_FILE"

# arc_eval_script.py
echo "[3/3] Running arc_eval_script.py ..."
cp "$(dirname "$0")/arc_eval_script.py" "$HRM_DIR/arc_eval_script.py"
"$PYTHON" arc_eval_script.py --data "$DATA_PATH" --checkpoint "$CKPT_FILE"

cat <<EOF

============================================================================
Expected at 8k epochs (ARC-AGI-1, vanilla):
  - HRM eval/exact_accuracy ~ 9-11 %  (training-time per-example)
  - voted K=1 ~ 4-8 %
  - voted K=2 ~ 6-12 %

If exact_accuracy is in [5, 15] % the training pipeline is producing the
expected 8k-baseline. Proceed to experiment 3 (100 k epoch full run).

Checkpoint: $CKPT_FILE
============================================================================
EOF
