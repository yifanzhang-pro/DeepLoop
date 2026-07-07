#!/usr/bin/env bash
# ============================================================================
# Quick comparison run A: VANILLA HRM (sapientinc/HRM) on ARC-AGI-1+ConceptARC.
# 8 GPU x ~6h x 27000 epoch (~27% of the 100k full HRM recipe).
#
# Pair with run_quick_hrm_dl.sh (same compute, our LoopDeepNorm patch) for a
# direct A/B head-to-head. Both finish in ~6h on 8x H200.
#
# Use:  bash scripts/run_quick_hrm.sh
#
# Env-var overrides (all optional):
#   HRM_DIR        = clone of sapientinc/HRM (default: $DEEPLOOP_DIR/../HRM)
#   DATA_DIR       = arc-aug-1000 dataset (default: $HRM_DIR/data/arc-aug-1000)
#   DEEPLOOP_DIR   = this repo (default: parent of this script)
#   NPROC          = num GPUs (default: 8)
#   EPOCHS         = total epochs (default: 27000; must be divisible by EVAL_INTERVAL)
#   EVAL_INTERVAL  = eval + ckpt frequency (default: 3000 → 9 evals)
#   PYTHON         = python interpreter (default: $VENV_DIR/bin/python or python3)
#
# Expected wall on 8x H200: ~5.5-6h training + 1 eval = ~6h total.
# Resilience: HRM does NOT auto-resume; if interrupted, evaluate the latest
# saved step_<N> manually via:
#     $PY evaluate.py checkpoint=<HRM_DIR>/checkpoints/.../step_<N>
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPLOOP_DIR="${DEEPLOOP_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
HRM_DIR="${HRM_DIR:-$(cd "$DEEPLOOP_DIR/.." && pwd)/HRM}"

NPROC="${NPROC:-8}"
EPOCHS="${EPOCHS:-27000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-3000}"
DATA_DIR="${DATA_DIR:-$HRM_DIR/data/arc-aug-1000}"
PYTHON="${PYTHON:-${VENV_DIR:-$DEEPLOOP_DIR/.venv}/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"

if (( EPOCHS % EVAL_INTERVAL != 0 )); then
    echo "FATAL: EVAL_INTERVAL ($EVAL_INTERVAL) must divide EPOCHS ($EPOCHS); HRM asserts this." >&2
    exit 2
fi

echo "[run_quick_hrm] HRM_DIR        = $HRM_DIR"
echo "[run_quick_hrm] DATA_DIR       = $DATA_DIR"
echo "[run_quick_hrm] NPROC          = $NPROC"
echo "[run_quick_hrm] EPOCHS         = $EPOCHS"
echo "[run_quick_hrm] EVAL_INTERVAL  = $EVAL_INTERVAL"
echo "[run_quick_hrm] PYTHON         = $PYTHON"

[[ -d "$HRM_DIR" ]] || { echo "FATAL: HRM_DIR ($HRM_DIR) not found. See README setup." >&2; exit 2; }
[[ -d "$DATA_DIR" ]] || {
    echo "FATAL: DATA_DIR ($DATA_DIR) missing. Build it via:" >&2
    echo "  cd $HRM_DIR && python dataset/build_arc_dataset.py" >&2
    exit 2
}

cd "$HRM_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# arch defaults to hrm_v1 (vanilla HRM upstream). Hyperparams from cfg_pretrain.yaml.
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    pretrain.py \
    data_path="$DATA_DIR" \
    epochs="$EPOCHS" \
    eval_interval="$EVAL_INTERVAL" \
    checkpoint_every_eval=True

# Eval final ckpt
FINAL_RUN_DIR=$(find checkpoints -maxdepth 3 -type d -name 'HierarchicalReasoningModel_ACTV1*' 2>/dev/null | grep -iE 'arc-1|arc-aug-1000' | sort | tail -1)
FINAL_CKPT=$(ls "$FINAL_RUN_DIR"/step_* 2>/dev/null | sort -V | tail -1 || true)
echo
echo "[run_quick_hrm] training done. Final ckpt: $FINAL_CKPT"
echo "[run_quick_hrm] starting eval..."

if [[ -n "$FINAL_CKPT" ]]; then
    "$PYTHON" evaluate.py checkpoint="$FINAL_CKPT"
else
    echo "[run_quick_hrm] WARN: no checkpoint found; skipping eval. Resume manually." >&2
fi

echo
echo "[run_quick_hrm] all done. Look for the metrics dict above; exact_accuracy"
echo "is the K=1 single-shot ARC-AGI-1 score (compare against run_quick_hrm_dl.sh)."
