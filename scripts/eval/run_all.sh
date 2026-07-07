#!/usr/bin/env bash
# Drive arxiv-2601.00417 evaluation for all six small-scale training records.
#
# Runs on della-gpu (login node with a single A100 40GB + network). Loops
# over six checkpoints x {0-shot, 1-shot} and writes per-run JSON under
# eval_results/. Does NOT parallelize (one GPU).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV="${VENV:-/path/to/GeoLoop/.venv_sl8808}"

cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/path/to/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/path/to/hf_datasets_cache}"
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_VERBOSITY=error
export PYTHONNOUSERSITE=1
# Login node; network access kept on for dataset fetches that are not yet
# cached. Set HF_HUB_OFFLINE=1 from the caller if you want to force offline.

TASKS="${TASKS:-arc_challenge,arc_easy,hellaswag,openbookqa,piqa,sciq,social_iqa,winogrande}"
BATCH="${BATCH:-16}"
DTYPE="${DTYPE:-bfloat16}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/eval_results}"
LIMIT_ARG=""
if [ -n "${LIMIT:-}" ]; then LIMIT_ARG="--limit $LIMIT"; fi

MODELS=(base_l1s base_l3s base_l5s dlv1_l1s dlv1_l3s dlv1_l5s)
SHOTS=(0 1)

mkdir -p "$OUT_ROOT" "$REPO_ROOT/eval_logs"

for name in "${MODELS[@]}"; do
  ckpt="$REPO_ROOT/training_records/$name/checkpoint-best"
  if [ ! -f "$ckpt/model.safetensors" ]; then
    echo "[run_all] SKIP $name (no ckpt at $ckpt)" >&2
    continue
  fi
  for nfs in "${SHOTS[@]}"; do
    sentinel="$OUT_ROOT/$name/nfs${nfs}/summary.json"
    if [ -f "$sentinel" ] && [ -z "${FORCE:-}" ]; then
      echo "[run_all] [$(date +%H:%M:%S)] skip $name nfs=$nfs (already done)"
      continue
    fi
    log="$REPO_ROOT/eval_logs/${name}_nfs${nfs}.log"
    echo "[run_all] [$(date +%H:%M:%S)] $name nfs=$nfs -> $log"
    "$VENV/bin/python" scripts/eval/run_lm_eval.py \
      --checkpoint "$ckpt" \
      --tasks "$TASKS" \
      --num_fewshot "$nfs" \
      --batch_size "$BATCH" \
      --dtype "$DTYPE" \
      --output_dir "$OUT_ROOT" \
      --run_name "$name" \
      $LIMIT_ARG >"$log" 2>&1
    echo "[run_all] [$(date +%H:%M:%S)]   done $name nfs=$nfs"
  done
done

echo "[run_all] all runs finished. summaries under $OUT_ROOT/"
