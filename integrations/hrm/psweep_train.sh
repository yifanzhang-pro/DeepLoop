#!/bin/bash
# ============================================================================
# p-sweep TRAIN (shared body) — one DeepLoop p-curve point at HRM-paper scale.
# Train ONLY; evaluation is a separate job (psweep_eval.sh / psweval.slurm).
#
# Invoked by the thin pst_*.slurm wrappers, which set ARCH:
#   ARCH=hrm_v1_dl_h013_noemb  -> DeepLoop H-only p=0.13, no embed RMSNorm
#   ARCH=hrm_v1                -> vanilla control
#
# This is the corrected redo of the original p-curve runs, whose results were
# biased by (a) the silent AdamW fallback (no preflight then) and (b) broken
# voting eval. Gates baked in here:
#   1. _preflight.sh           — AdamATan2 C-extension importable, pretrain.py
#                                unpatched, dataset present, CUDA up.
#   2. ARC-1 dataset CONTENT gate (md5 of identifiers.json) — presence is not
#                                enough; a divergent build is not comparable to
#                                the in-house baselines (vanilla-100k=37.00%).
#   3. checkpoint_every_eval=True passed EXPLICITLY + post-train verification
#                                that checkpoints were actually written and the
#                                final one is loadable, + an early background
#                                watcher that screams if no checkpoint has
#                                appeared after the first eval interval.
#
# 8 GPU x EPOCHS (default 100000) ~ 22 h. Run from HRM_DIR.
# ============================================================================
set -euo pipefail

HRM_DIR="${HRM_DIR:?set HRM_DIR to your HRM clone}"
PYTHON="${PYTHON:?set PYTHON to the venv python with adam_atan2_backend}"
ARCH="${ARCH:?set ARCH (e.g. hrm_v1_dl_h013_noemb, or hrm_v1 for vanilla)}"
EPOCHS="${EPOCHS:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
SEED="${SEED:-}"                 # optional: re-run a point under a different training seed
DATA_PATH="data/arc-aug-1000"

# Optional seed override (default = pretrain.py's config default, seed=0). When
# set, it ONLY changes training stochasticity (data shuffle + init), never the
# hash-verified dataset. Marker/run identity is keyed by SEED so multiple seeds
# of the same arch do not clobber each other's run-dir marker.
SEED_ARG=""; SEED_TAG=""
if [[ -n "$SEED" ]]; then SEED_ARG="seed=$SEED"; SEED_TAG="_seed${SEED}"; fi

# md5 of the reference ARC-1 build's identifiers.json — the build every
# in-house baseline was trained/evaluated on. See
# integrations/vanilla_hrm/reference_hashes.json (single source of truth).
ARC1_IDENTIFIERS_MD5="a8cafb9bdab646dab951c28d82700ffe"

# Compute nodes are offline.
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

# Gate 1: AdamATan2 + unpatched pretrain.py + dataset presence + CUDA.
source "$(dirname "$0")/_preflight.sh"

cd "$HRM_DIR"

# Gate 2: dataset CONTENT (not just presence).
GOT_MD5=$(md5sum "$DATA_PATH/identifiers.json" | cut -d' ' -f1)
if [[ "$GOT_MD5" != "$ARC1_IDENTIFIERS_MD5" ]]; then
    cat <<EOF 1>&2
FATAL: $DATA_PATH is NOT the reference ARC-1 build
       (identifiers.json md5 $GOT_MD5 != $ARC1_IDENTIFIERS_MD5).
Results on a divergent build are not comparable to the in-house baselines.
Rebuild deterministically:
    python integrations/vanilla_hrm/build_from_manifest.py \\
        --hrm-dir "$HRM_DIR" --dataset arc1
EOF
    exit 1
fi
echo "[psweep] dataset content gate: PASS"

# Gate 3 (pre): arch config exists; checkpoint flag supported by pretrain.py.
[[ -f "config/arch/$ARCH.yaml" ]] || { echo "FATAL: config/arch/$ARCH.yaml missing — deploy the arch yamls first"; exit 1; }
grep -q "checkpoint_every_eval" pretrain.py || { echo "FATAL: pretrain.py lacks checkpoint_every_eval — wrong/modified upstream?"; exit 1; }

PREFIX="checkpoints/Arc-aug-1000 ACT-torch/HierarchicalReasoningModel_ACTV1 "

# Snapshot pre-existing run dirs (coolnames are assigned at runtime).
SNAP="$(mktemp)"; ls -d "$PREFIX"* 2>/dev/null > "$SNAP" || true

# Early checkpoint watcher: first eval lands at ~EVAL_INTERVAL epochs (~2 h at
# full scale); if 4 h in, a new run dir exists but holds NO step_* checkpoint,
# scream — likely a disk/permission problem that would waste a 22 h run.
(
    sleep 14400
    mapfile -t NEWDIRS < <(ls -d "$PREFIX"* 2>/dev/null | grep -vxFf "$SNAP" || true)
    for d in "${NEWDIRS[@]}"; do
        if ! ls "$d"/step_* >/dev/null 2>&1; then
            touch "$d/CKPT_WARNING"
            echo "[psweep][WATCHDOG] !!! 4h elapsed and NO checkpoint in '$d' — check disk quota / permissions. Training continues, but if this persists the run will only have its final checkpoint (or none on crash)." 1>&2
        fi
    done
) & WATCHER_PID=$!
trap 'kill "$WATCHER_PID" 2>/dev/null || true; rm -f "$SNAP"' EXIT

echo "[psweep] training: arch=$ARCH epochs=$EPOCHS eval_interval=$EVAL_INTERVAL checkpoint_every_eval=True ${SEED_ARG:-(seed=default)}"
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=8 \
    pretrain.py arch="$ARCH" epochs="$EPOCHS" eval_interval="$EVAL_INTERVAL" \
    checkpoint_every_eval=True $SEED_ARG

# ---- Identify THIS run's dir (set difference vs snapshot) -------------------
mapfile -t NEW < <(ls -d "$PREFIX"* 2>/dev/null | grep -vxFf "$SNAP" || true)
if [[ ${#NEW[@]} -gt 1 ]]; then
    echo "ERROR: ${#NEW[@]} new run dirs appeared (concurrent runs?). Identify manually:"; printf '  %s\n' "${NEW[@]}"; exit 1
fi
RUN_DIR="${NEW[0]:-}"
[[ -z "$RUN_DIR" ]] && RUN_DIR="$(ls -dt "$PREFIX"* 2>/dev/null | head -1)"
[[ -n "$RUN_DIR" ]] || { echo "FATAL: no run directory found after training"; exit 1; }
echo "[psweep] run dir: $RUN_DIR"

# ---- Post-train checkpoint verification -------------------------------------
EXPECTED=$(( EPOCHS / EVAL_INTERVAL ))
NCKPT=$(ls "$RUN_DIR" | grep -cE '^step_[0-9]+$' || true)
echo "[psweep] checkpoints written: $NCKPT (expected ~$EXPECTED with checkpoint_every_eval=True)"
if [[ "$NCKPT" -eq 0 ]]; then
    echo "FATAL: training finished but NO checkpoint exists in $RUN_DIR — nothing to evaluate."; exit 1
fi
[[ "$NCKPT" -lt "$EXPECTED" ]] && echo "[psweep][WARN] fewer checkpoints than expected — intermediate evals may have failed to save; final checkpoint is still usable."

LATEST_STEP=$(ls "$RUN_DIR" | grep -E '^step_[0-9]+$' | sed 's/step_//' | sort -n | tail -1)
CKPT="$RUN_DIR/step_$LATEST_STEP"
SIZE=$(stat -c%s "$CKPT")
[[ "$SIZE" -gt 500000000 ]] || { echo "FATAL: final checkpoint $CKPT is suspiciously small ($SIZE bytes) — truncated write?"; exit 1; }
"$PYTHON" - "$CKPT" <<'PYEOF'
import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu")
assert isinstance(sd, dict) and len(sd) > 10, f"state dict malformed: {type(sd)} len={len(sd) if hasattr(sd,'__len__') else '?'}"
print(f"[psweep] final checkpoint loadable: {len(sd)} keys OK")
PYEOF

# ---- Handoff marker for the separate eval job (keyed by ARCH + SEED) --------
mkdir -p .exploration
echo "$RUN_DIR" > ".exploration/psweep_${ARCH}${SEED_TAG}_rundir"
cat <<EOF

============================================================================
TRAIN DONE  arch=$ARCH${SEED_TAG:+  seed=$SEED}
  run dir   : $RUN_DIR
  checkpoint: $CKPT  ($NCKPT checkpoints kept)
Evaluate (separate 4-GPU job, ~45 min):
    sbatch --export=ALL,ARCH=$ARCH${SEED:+,SEED=$SEED} psweval.slurm
============================================================================
EOF
