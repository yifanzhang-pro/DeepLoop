#!/usr/bin/env bash
# ============================================================================
# Full HRM + ours: our gpt-mha-rope-loop backbone with HRM training tricks
# (stablemax CE, AdamW, lr=1e-4 with HRM warmup, batch=768, 100k step).
# Train + eval on ARC-AGI-1 + ConceptARC HRM-split.
#
# Use:  bash scripts/run_full_hrm_ours.sh
#
# Env-var overrides:
#   DEEPLOOP_DIR = our repo path (default: parent of this script)
#   NPROC        = num GPUs (default: 8)
#   MAX_ITERS    = total training steps (default: 100000)
#   EVAL_INTERVAL= eval + save frequency (default: 5000)
#   PYTHON       = python interpreter (default: $VENV_DIR/bin/python)
#   ARC_RAW_DIR  = official ARC-AGI raw data (default below)
#
# Resilience: train_adam_arc_agi.py supports --init_from=resume natively.
# This script auto-detects the latest checkpoint dir matching the run_name
# and resumes from it. Server interruption loses at most ONE eval interval
# of work.
#
# Expected wall time:
#   8 GPU: ~28-30h
#   4 GPU: ~57h
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPLOOP_DIR="${DEEPLOOP_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

NPROC="${NPROC:-8}"
MAX_ITERS="${MAX_ITERS:-100000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-5000}"
PYTHON="${PYTHON:-${VENV_DIR:-$DEEPLOOP_DIR/.venv}/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"

ARC_RAW_DIR="${ARC_RAW_DIR:-/path/to/idea_den/arc_agi_ref/ARC-AGI/data}"
RUN_NAME="${RUN_NAME:-trn_arc1c_full_hrm_tricks}"
CONFIG_FILE="config/train_gpt_mha_rope_small_loop3_arc_agi1_conceptarc_hrmsplit_pid_full_hrm_tricks_80g8.py"

echo "[run_full_hrm_ours] DEEPLOOP_DIR  = $DEEPLOOP_DIR"
echo "[run_full_hrm_ours] NPROC         = $NPROC"
echo "[run_full_hrm_ours] MAX_ITERS     = $MAX_ITERS"
echo "[run_full_hrm_ours] EVAL_INTERVAL = $EVAL_INTERVAL"
echo "[run_full_hrm_ours] PYTHON        = $PYTHON"
echo "[run_full_hrm_ours] RUN_NAME      = $RUN_NAME"

cd "$DEEPLOOP_DIR"
[[ -f "$CONFIG_FILE" ]] || { echo "FATAL: $CONFIG_FILE not found." >&2; exit 2; }
[[ -d "data/arc-agi1-conceptarc-hrmsplit-puzzleid-aug1000" ]] || {
    echo "FATAL: dataset arc-agi1-conceptarc-hrmsplit-puzzleid-aug1000 not built." >&2
    echo "Run: $PYTHON data/arc-agi/prepare.py --hrm_split --with_puzzle_id \\" >&2
    echo "       --raw_data_dir $ARC_RAW_DIR --extra_train_dir <ConceptARC>/corpus \\" >&2
    echo "       --output_dir data/arc-agi1-conceptarc-hrmsplit-puzzleid-aug1000 \\" >&2
    echo "       --num_aug 1000" >&2
    exit 2
}

export HF_HOME="${HF_HOME:-$DEEPLOOP_DIR/hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

# ----------------------------------------------------------------------------
# RESUME: find latest in-progress checkpoint matching RUN_NAME
# ----------------------------------------------------------------------------
INIT_FROM_FLAG=""
RESUME_DIR_FLAG=""
LATEST_RUN=$(ls -td output/out_arc_*${RUN_NAME}* 2>/dev/null | head -1 || true)
if [[ -z "$LATEST_RUN" ]]; then
    LATEST_RUN=$(ls -td output/out_arc_*hrm_tricks* 2>/dev/null | head -1 || true)
fi
if [[ -n "$LATEST_RUN" ]] && [[ -d "$LATEST_RUN/checkpoint-best" ]]; then
    LAST_STEP=$(python3 -c "import json; print(json.load(open('$LATEST_RUN/checkpoint-best/trainer_state.json'))['global_step'])" 2>/dev/null || echo 0)
    if (( LAST_STEP > 0 )) && (( LAST_STEP < MAX_ITERS )); then
        echo "[run_full_hrm_ours] RESUMING from $LATEST_RUN @ step $LAST_STEP"
        INIT_FROM_FLAG="--init_from=resume"
        RESUME_DIR_FLAG="--resume_dir=$LATEST_RUN"
    fi
fi

# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
    train_adam_arc_agi.py \
    "$CONFIG_FILE" \
    --model_type=gpt-mha-rope-loop \
    --block_repeats=3 \
    --use_input_embed_rmsnorm=True \
    --use_loop_deepnorm=True \
    --loss_type=stablemax_ce \
    --optimizer_name=adam_atan2 \
    --max_iters="$MAX_ITERS" \
    --lr_decay_iters="$MAX_ITERS" \
    --eval_interval="$EVAL_INTERVAL" \
    --save_checkpoints=True \
    --keep_last_checkpoints=3 \
    --wandb_run_name="$RUN_NAME" \
    $INIT_FROM_FLAG $RESUME_DIR_FLAG

# ----------------------------------------------------------------------------
# Locate final ckpt and run small + full eval
# ----------------------------------------------------------------------------
FINAL_RUN=$(ls -td output/out_arc_*${RUN_NAME}* 2>/dev/null | head -1 || \
            ls -td output/out_arc_*hrm_tricks* 2>/dev/null | head -1)
FINAL_CKPT="$FINAL_RUN/checkpoint-best"
echo
echo "[run_full_hrm_ours] training done. Final ckpt: $FINAL_CKPT"

# Small eval (100 puzzles × 64 augs) — fast sanity check.
mkdir -p eval_results/arc_agi
EVAL_OUT_SMALL="eval_results/arc_agi/iter_full_hrm_tricks_aug64.json"
EVAL_OUT_FULL="eval_results/arc_agi/iter_full_hrm_tricks_aug1000_full.json"

echo "[run_full_hrm_ours] small eval: 100 puzzles × 64 augs"
"$PYTHON" scripts/eval/run_arc_agi_eval.py \
    --checkpoint "$FINAL_CKPT" \
    --raw_data_dir "$ARC_RAW_DIR" \
    --identifier_map data/arc-agi1-conceptarc-hrmsplit-puzzleid-aug1000/identifiers.json \
    --num_puzzles 100 --num_augs 64 --batch_size 32 --Ks 1,2 \
    --snapshot_every 10 \
    --out "$EVAL_OUT_SMALL" --seed 42

echo
echo "[run_full_hrm_ours] full eval: 400 puzzles × 1000 augs (HRM-equivalent K=2 scoring)"
echo "  multi-GPU recommended for this; using NPROC=$NPROC"
if (( NPROC > 1 )); then
    "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" \
        scripts/eval/run_arc_agi_eval.py \
        --checkpoint "$FINAL_CKPT" \
        --raw_data_dir "$ARC_RAW_DIR" \
        --identifier_map data/arc-agi1-conceptarc-hrmsplit-puzzleid-aug1000/identifiers.json \
        --num_augs 1000 --batch_size 32 --Ks 1,2 \
        --snapshot_every 5 \
        --out "$EVAL_OUT_FULL" --seed 42
    "$PYTHON" scripts/eval/merge_arc_eval_shards.py --out "$EVAL_OUT_FULL" --world_size "$NPROC"
else
    "$PYTHON" scripts/eval/run_arc_agi_eval.py \
        --checkpoint "$FINAL_CKPT" \
        --raw_data_dir "$ARC_RAW_DIR" \
        --identifier_map data/arc-agi1-conceptarc-hrmsplit-puzzleid-aug1000/identifiers.json \
        --num_augs 1000 --batch_size 32 --Ks 1,2 \
        --snapshot_every 5 \
        --out "$EVAL_OUT_FULL" --seed 42
fi

echo
echo "[run_full_hrm_ours] all done. Results:"
echo "  small (100 × 64 aug):       $EVAL_OUT_SMALL"
echo "  full  (400 × 1000 aug):     $EVAL_OUT_FULL"
"$PYTHON" -c "
import json
for path in ['$EVAL_OUT_SMALL', '$EVAL_OUT_FULL']:
    try:
        s = json.load(open(path))['summary']
        acc = s.get('puzzle_accuracy_at_k') or s.get('shard_puzzle_accuracy_at_k')
        print(path, '→ K=1=' + str(round(acc['1']*100, 2)) + '%, K=2=' + str(round(acc['2']*100, 2)) + '%')
    except Exception as e:
        print(path, '→ (failed to parse:', e, ')')
"
