#!/bin/bash
# ============================================================================
# EXPERIMENT 3 — full vanilla ARC-AGI-1 reproduction (the answer).
#
# This is byte-for-byte the HRM README's ARC-AGI-1 training command:
#
#   OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py
#
# i.e. arch=hrm_v1 (vanilla, no DL), data_path=data/arc-aug-1000 (default),
# epochs=100000 (default), eval_interval=10000 (default), with the full set
# of default optimization hyperparameters from config/cfg_pretrain.yaml.
#
# Hardware:  8 H100/H200 GPUs
# Wall time: ~22 h of training + ~30 min of eval
#
# Expected result (HRM paper headline on ARC-AGI-1):
#   - HRM eval/exact_accuracy (training-time, per-example) ~ 32 %
#   - voted K=2 (arc_eval_script.py) ~ 40.3 %
#
# Decision matrix:
#   K=2 in [38, 42] %  → full reproduction confirmed; we can trust paper-recipe
#                       comparisons and use this as the vanilla baseline.
#   K=2 in [30, 38] %  → close but not quite; possible seed variance or
#                       minor env delta. Re-run once for an independent
#                       confirmation.
#   K=2 < 30 %         → systematic gap. Inspect: was the run on real
#                       AdamATan2? did wandb show normal loss curves?
#                       different from paper at which eval step?
# ============================================================================

set -euo pipefail

HRM_DIR="${HRM_DIR:?set HRM_DIR}"
PYTHON="${PYTHON:?set PYTHON}"
DATA_PATH="data/arc-aug-1000"
source "$(dirname "$0")/_preflight.sh"

cd "$HRM_DIR"

# Train — paper recipe, NO overrides. The README's command, full stop.
echo "[1/3] Training vanilla HRM at full paper recipe (100000 epochs) ..."
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 pretrain.py

# Discover the run dir (coolname).
RUN_DIR=$(ls -dt "checkpoints/Arc-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 "* 2>/dev/null | head -1)
[[ -z "$RUN_DIR" ]] && { echo "ERROR: no run directory found"; exit 1; }
echo "Run dir: $RUN_DIR"

LATEST_STEP=$(ls "$RUN_DIR" | grep -E "^step_[0-9]+$" | sed 's/step_//' | sort -n | tail -1)
CKPT_FILE="$RUN_DIR/step_$LATEST_STEP"
echo "Checkpoint: $CKPT_FILE"

echo "[2/3] Running evaluate.py ..."
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
    evaluate.py checkpoint="$CKPT_FILE"

echo "[3/3] Running arc_eval_script.py ..."
cp "$(dirname "$0")/arc_eval_script.py" "$HRM_DIR/arc_eval_script.py"
"$PYTHON" arc_eval_script.py --data "$DATA_PATH" --checkpoint "$CKPT_FILE"

cat <<EOF

============================================================================
Expected at 100k epochs (ARC-AGI-1, vanilla, paper recipe):
  - HRM eval/exact_accuracy ~ 32 %
  - voted K=1 ~ 30-35 %
  - voted K=2 ~ 40.3 %

Checkpoint: $CKPT_FILE
============================================================================
EOF
