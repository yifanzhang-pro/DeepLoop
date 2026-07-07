#!/bin/bash
# ============================================================================
# EXPERIMENT 4 — independent cross-check: full vanilla ARC-AGI-2.
#
# Identical to experiment 3 except `data_path=data/arc-2-aug-1000`.
# This is the HRM README's exact ARC-AGI-2 command. Two reasons to run it:
#
#   (a) Independent confirmation of reproduction — paper reports a SECOND
#       number on ARC-AGI-2 (5.0 % K=2). Hitting both 40.3 % (ARC-1) and
#       5.0 % (ARC-2) is far stronger evidence that our setup matches the
#       paper than hitting either one alone.
#   (b) If experiment 3 falls short of paper, experiment 4 tells us whether
#       the gap is specific to ARC-1 (e.g. ConceptARC mixin trouble) or
#       general to our training pipeline.
#
# Hardware:  8 H100/H200 GPUs
# Wall time: ~22 h of training + ~30 min of eval
#
# Expected:
#   - HRM eval/exact_accuracy ~ 3-5 %  (per paper, ARC-AGI-2 is much harder)
#   - voted K=2 ~ 5.0 %
# ============================================================================

set -euo pipefail

HRM_DIR="${HRM_DIR:?set HRM_DIR}"
PYTHON="${PYTHON:?set PYTHON}"
DATA_PATH="data/arc-2-aug-1000"
source "$(dirname "$0")/_preflight.sh"

cd "$HRM_DIR"

echo "[1/3] Training vanilla HRM on ARC-AGI-2, 100000 epochs ..."
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    pretrain.py data_path="$DATA_PATH"

RUN_DIR=$(ls -dt "checkpoints/Arc-2-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 "* 2>/dev/null | head -1)
[[ -z "$RUN_DIR" ]] && { echo "ERROR: no run directory found"; exit 1; }
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
Expected at 100k epochs (ARC-AGI-2, vanilla, paper recipe):
  - HRM eval/exact_accuracy ~ 3-5 %
  - voted K=2 ~ 5.0 %

Checkpoint: $CKPT_FILE
============================================================================
EOF
