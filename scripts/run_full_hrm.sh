#!/usr/bin/env bash
# ============================================================================
# Full HRM ARC-AGI-1 reproduction (train + eval, 100k epoch, 8x GPU).
#
# Use:  bash scripts/run_full_hrm.sh
#
# Env-var overrides (all optional):
#   HRM_DIR      = path to HRM repo (default: ../HRM relative to this script)
#   DATA_DIR     = arc-aug-1000 dataset (default: $HRM_DIR/data/arc-aug-1000)
#   NPROC        = num GPUs (default: 8)
#   EPOCHS       = total epochs (default: 100000, HRM standard)
#   EVAL_INTERVAL= ckpt + eval frequency (default: 5000 — must divide EPOCHS)
#   PYTHON       = python interpreter (default: $VENV_DIR/bin/python or python3)
#
# Resilience: HRM saves a checkpoint every EVAL_INTERVAL epochs. If the server
# is interrupted, this script auto-detects the latest checkpoint on the next
# invocation and prints how to resume manually (HRM upstream doesn't support
# auto-resume; see RESUME section at the bottom).
#
# Expected wall time (A100/H100/H200, batch=768):
#   8 GPU: ~22-24h
#   4 GPU: ~45h
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPLOOP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

HRM_DIR="${HRM_DIR:-$(cd "$DEEPLOOP_DIR/.." && pwd)/HRM}"
NPROC="${NPROC:-8}"
EPOCHS="${EPOCHS:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
DATA_DIR="${DATA_DIR:-$HRM_DIR/data/arc-aug-1000}"
PYTHON="${PYTHON:-${VENV_DIR:-$DEEPLOOP_DIR/.venv}/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"

if (( EPOCHS % EVAL_INTERVAL != 0 )); then
    echo "FATAL: EVAL_INTERVAL ($EVAL_INTERVAL) must divide EPOCHS ($EPOCHS); HRM asserts this." >&2
    exit 2
fi

echo "[run_full_hrm] HRM_DIR        = $HRM_DIR"
echo "[run_full_hrm] DEEPLOOP_DIR   = $DEEPLOOP_DIR"
echo "[run_full_hrm] DATA_DIR       = $DATA_DIR"
echo "[run_full_hrm] NPROC          = $NPROC"
echo "[run_full_hrm] EPOCHS         = $EPOCHS"
echo "[run_full_hrm] EVAL_INTERVAL  = $EVAL_INTERVAL"
echo "[run_full_hrm] PYTHON         = $PYTHON"

[[ -d "$HRM_DIR" ]] || { echo "FATAL: HRM_DIR not found. Clone sapientinc/HRM and re-run." >&2; exit 2; }
[[ -d "$DATA_DIR" ]] || { echo "FATAL: DATA_DIR ($DATA_DIR) missing. See README dataset section." >&2; exit 2; }

cd "$HRM_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# ----------------------------------------------------------------------------
# RESUME: detect existing in-progress run
# ----------------------------------------------------------------------------
EXISTING_RUN_DIR=$(find checkpoints -maxdepth 3 -type d -name 'HierarchicalReasoningModel_ACTV1*' 2>/dev/null | head -1 || true)
if [[ -n "$EXISTING_RUN_DIR" ]]; then
    LATEST_STEP=$(ls "$EXISTING_RUN_DIR"/step_* 2>/dev/null | sort -V | tail -1 || true)
    if [[ -n "$LATEST_STEP" ]]; then
        STEP_NUM=$(basename "$LATEST_STEP" | sed 's/step_//')
        echo
        echo "[run_full_hrm] WARNING: existing run with checkpoint at step $STEP_NUM:"
        echo "  $LATEST_STEP"
        echo "HRM upstream does NOT auto-resume; this script will train a NEW run from"
        echo "scratch. To resume, manually edit pretrain.py to load this state, or just"
        echo "evaluate the existing ckpt by running:"
        echo "  $PYTHON evaluate.py checkpoint=\"$LATEST_STEP\""
        echo
    fi
fi

# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
echo "[run_full_hrm] starting train..."
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    pretrain.py \
    data_path="$DATA_DIR" \
    epochs="$EPOCHS" \
    eval_interval="$EVAL_INTERVAL" \
    checkpoint_every_eval=True

# ----------------------------------------------------------------------------
# Locate final checkpoint and evaluate
# ----------------------------------------------------------------------------
FINAL_RUN_DIR=$(find checkpoints -maxdepth 3 -type d -name 'HierarchicalReasoningModel_ACTV1*' 2>/dev/null | sort | tail -1)
FINAL_CKPT=$(ls "$FINAL_RUN_DIR"/step_* 2>/dev/null | sort -V | tail -1)
echo
echo "[run_full_hrm] training done. Final ckpt: $FINAL_CKPT"
echo "[run_full_hrm] starting eval..."

"$PYTHON" evaluate.py checkpoint="$FINAL_CKPT"

echo
echo "[run_full_hrm] all done. Eval result printed above (look for"
echo "  {'all': {'accuracy': ..., 'exact_accuracy': ..., ...}}"
echo "exact_accuracy is the K=1 single-shot ARC-AGI score."
