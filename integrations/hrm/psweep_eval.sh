#!/bin/bash
# ============================================================================
# p-sweep EVAL (generic, train-decoupled) — evaluate.py + paper-protocol
# count-first voting for ANY checkpoint of any sweep run.
#
# Select the run by ONE of:
#   ARCH=hrm_v1_dl_h013_noemb   -> reads the marker written by psweep_train.sh
#                                  (.exploration/psweep_${ARCH}_rundir)
#   RUN_DIR="checkpoints/.../HierarchicalReasoningModel_ACTV1 <coolname>"
# Optional:
#   STEP=259035        evaluate a specific checkpoint (default: latest step_N)
#   SKIP_EVALUATE=auto reuse existing step_N_all_preds.* if present (default);
#                =1    never run evaluate.py (vote only);  =0  always re-run
#
# 4 GPU, ~45 min (ARC-1 full test set). Run from HRM_DIR.
# Note: our own checkpoints are trained WITH torch.compile (keys have the
# _orig_mod. prefix), which matches evaluate.py's default load path — do NOT
# set DISABLE_COMPILE here (that is only for the published checkpoint).
# ============================================================================
set -euo pipefail

HRM_DIR="${HRM_DIR:?set HRM_DIR to your HRM clone}"
PYTHON="${PYTHON:?set PYTHON to the venv python}"
DATA_PATH="${DATA_PATH:-data/arc-aug-1000}"
SKIP_EVALUATE="${SKIP_EVALUATE:-auto}"
NPROC="${NPROC:-4}"
MPORT="${MPORT:-$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}"
ARC1_IDENTIFIERS_MD5="a8cafb9bdab646dab951c28d82700ffe"

export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export OMP_NUM_THREADS=8

cd "$HRM_DIR"

# ---- Resolve run dir (marker keyed by ARCH + optional SEED) ------------------
if [[ -z "${RUN_DIR:-}" ]]; then
    ARCH="${ARCH:?set ARCH (to use the train marker) or RUN_DIR explicitly}"
    SEED="${SEED:-}"
    MARKER=".exploration/psweep_${ARCH}${SEED:+_seed${SEED}}_rundir"
    [[ -f "$MARKER" ]] || { echo "FATAL: no marker $MARKER — did psweep_train.sh finish (with the same SEED)? Otherwise pass RUN_DIR explicitly."; exit 1; }
    RUN_DIR="$(cat "$MARKER")"
fi
[[ -d "$RUN_DIR" ]] || { echo "FATAL: RUN_DIR '$RUN_DIR' does not exist"; exit 1; }
echo "[psweval] run dir: $RUN_DIR"

# ---- Dataset content gate (same reference build as training/baselines) -------
GOT_MD5=$(md5sum "$DATA_PATH/identifiers.json" | cut -d' ' -f1)
[[ "$GOT_MD5" == "$ARC1_IDENTIFIERS_MD5" ]] || { echo "FATAL: $DATA_PATH is not the reference ARC-1 build ($GOT_MD5) — numbers would not be comparable."; exit 1; }
echo "[psweval] dataset content gate: PASS"

# ---- Resolve checkpoint -------------------------------------------------------
if [[ -z "${STEP:-}" ]]; then
    STEP=$(ls "$RUN_DIR" | grep -E '^step_[0-9]+$' | sed 's/step_//' | sort -n | tail -1)
    [[ -n "$STEP" ]] || { echo "FATAL: no step_N checkpoints in $RUN_DIR"; exit 1; }
fi
CKPT="$RUN_DIR/step_$STEP"
[[ -f "$CKPT" ]] || { echo "FATAL: $CKPT not found"; exit 1; }
echo "[psweval] checkpoint: $CKPT"

# ---- evaluate.py (preds), skippable when preds already exist ------------------
HAVE_PREDS=0
ls "$RUN_DIR/step_${STEP}_all_preds."* >/dev/null 2>&1 && HAVE_PREDS=1
RUN_EVAL=1
case "$SKIP_EVALUATE" in
    1)    RUN_EVAL=0 ;;
    auto) [[ "$HAVE_PREDS" == 1 ]] && { RUN_EVAL=0; echo "[psweval] reusing existing preds (SKIP_EVALUATE=auto)"; } ;;
esac
if [[ "$RUN_EVAL" == 1 ]]; then
    echo "[psweval] evaluate.py ($NPROC GPU, port $MPORT) ..."
    "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node="$NPROC" \
        --master_addr=127.0.0.1 --master_port="$MPORT" \
        evaluate.py checkpoint="$CKPT"
elif [[ "$HAVE_PREDS" == 0 ]]; then
    echo "FATAL: SKIP_EVALUATE=1 but no preds exist for step_$STEP"; exit 1
fi

# ---- Paper-protocol voting (count-first ranking + a1 dedup fix) ---------------
echo "[psweval] voting ..."
PYTHONPATH="$HRM_DIR" "$PYTHON" "$(dirname "$0")/arc_eval_script.py" \
    --data "$DATA_PATH" --checkpoint "$CKPT"

cat <<'EOF'

============================================================================
Reference points (ARC-1, fixed count-first voting, OUR harness):
  vanilla 100k (phenomenal-scallop, AdamATan2)      K=2 = 37.00%
  DeepLoop p=0.10 50k  (prudent-sturgeon, pre-fix)  K=2 = 35.75%
  DeepLoop p=0.10 100k (h010nf, pre-fix optimizer)  K=2 = 38.50%
  vanilla 50k  (beautiful-chamois, pre-fix)         K=2 = 32.75%
  HRM paper (their harness; ~3pp above ours)        K=2 = 40.3%
Checkpoint-to-checkpoint noise is ~2pp at this scale — judge p-points by
the curve, not single-point deltas.
============================================================================
EOF
