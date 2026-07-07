#!/usr/bin/env bash
# ============================================================================
# Quick comparison run B: HRM + DeepLoop core method (LoopDeepNorm + input
# embed RMSNorm; "Phase 1") on ARC-AGI-1+ConceptARC.
# 8 GPU x ~6h x 27000 epoch — same compute as run_quick_hrm.sh.
#
# Only difference from run_quick_hrm.sh: arch=hrm_v1_dl, which loads the
# patched model `models/hrm/hrm_act_v1_dl.py` (HRM upstream + α residual
# scaling + β output-projection init + input embed RMSNorm).
#
# Use:  bash scripts/run_quick_hrm_dl.sh
#
# Same env-var overrides as run_quick_hrm.sh. PRECONDITION: the patched arch
# files have been copied into HRM_DIR. Specifically:
#   $HRM_DIR/models/hrm/hrm_act_v1_dl.py    ← from this repo, see Setup.5 below
#   $HRM_DIR/config/arch/hrm_v1_dl.yaml     ← from this repo, see Setup.5 below
#
# Expected wall on 8x H200: ~5.5-6h training + 1 eval = ~6h total.
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

echo "[run_quick_hrm_dl] HRM_DIR        = $HRM_DIR"
echo "[run_quick_hrm_dl] DATA_DIR       = $DATA_DIR"
echo "[run_quick_hrm_dl] NPROC          = $NPROC"
echo "[run_quick_hrm_dl] EPOCHS         = $EPOCHS"
echo "[run_quick_hrm_dl] EVAL_INTERVAL  = $EVAL_INTERVAL"
echo "[run_quick_hrm_dl] PYTHON         = $PYTHON"

[[ -d "$HRM_DIR" ]] || { echo "FATAL: HRM_DIR ($HRM_DIR) not found. See README setup." >&2; exit 2; }
[[ -d "$DATA_DIR" ]] || {
    echo "FATAL: DATA_DIR ($DATA_DIR) missing. Build it via:" >&2
    echo "  cd $HRM_DIR && python dataset/build_arc_dataset.py" >&2
    exit 2
}
[[ -f "$HRM_DIR/models/hrm/hrm_act_v1_dl.py" ]] || {
    echo "FATAL: $HRM_DIR/models/hrm/hrm_act_v1_dl.py missing. Copy from DeepLoop:" >&2
    echo "  cp $DEEPLOOP_DIR/integrations/hrm/hrm_act_v1_dl.py $HRM_DIR/models/hrm/" >&2
    echo "  cp $DEEPLOOP_DIR/integrations/hrm/hrm_v1_dl.yaml   $HRM_DIR/config/arch/" >&2
    exit 2
}
[[ -f "$HRM_DIR/config/arch/hrm_v1_dl.yaml" ]] || {
    echo "FATAL: $HRM_DIR/config/arch/hrm_v1_dl.yaml missing. See message above." >&2
    exit 2
}

cd "$HRM_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# arch=hrm_v1_dl loads the patched model with use_loop_deepnorm=True + use_input_embed_rmsnorm=True.
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    pretrain.py \
    arch=hrm_v1_dl \
    data_path="$DATA_DIR" \
    epochs="$EPOCHS" \
    eval_interval="$EVAL_INTERVAL" \
    checkpoint_every_eval=True

# Eval final ckpt
FINAL_RUN_DIR=$(find checkpoints -maxdepth 3 -type d -name 'HierarchicalReasoningModel_ACTV1*' 2>/dev/null | grep -iE 'arc-1|arc-aug-1000' | sort | tail -1)
FINAL_CKPT=$(ls "$FINAL_RUN_DIR"/step_* 2>/dev/null | sort -V | tail -1 || true)
echo
echo "[run_quick_hrm_dl] training done. Final ckpt: $FINAL_CKPT"
echo "[run_quick_hrm_dl] starting eval..."

if [[ -n "$FINAL_CKPT" ]]; then
    "$PYTHON" evaluate.py checkpoint="$FINAL_CKPT"
else
    echo "[run_quick_hrm_dl] WARN: no checkpoint found; skipping eval. Resume manually." >&2
fi

echo
echo "[run_quick_hrm_dl] all done. Look for the metrics dict above; exact_accuracy"
echo "is the K=1 single-shot ARC-AGI-1 score (compare against run_quick_hrm.sh)."
