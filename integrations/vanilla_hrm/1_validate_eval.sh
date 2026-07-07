#!/bin/bash
# ============================================================================
# EXPERIMENT 1 — validate the evaluation pipeline against a paper-published
# checkpoint, BEFORE running any training. If this fails our pipeline is
# wrong and we must fix it before believing any training result.
#
# What it does:
#   1. Downloads sapientinc/HRM-checkpoint-ARC-2 from HuggingFace.
#   2. Builds data/arc-2-aug-1000 if not already built.
#   3. Runs evaluate.py to produce <ckpt>_all_preds.<rank> files.
#   4. Runs arc_eval_script.py (port of arc_eval.ipynb) to print K=1/K=2/...
#
# Hardware:  4 H100/H200 GPUs, ~1 GB GPU mem each
# Wall time: ~30-60 min
#
# Expected result:
#   The HRM paper reports 5.0% on ARC-AGI-2. The published checkpoint should
#   match within ±1 pp at K=2. If you see K=2 < 3% or > 8%, our pipeline is
#   producing different numbers than the authors did and that needs fixing
#   before any training reproduction is meaningful.
# ============================================================================

set -euo pipefail

# Fill these in for your environment, then run:
#   bash 1_validate_eval.sh
# OR via SLURM:
#   sbatch 1_validate_eval.slurm
HRM_DIR="${HRM_DIR:?set HRM_DIR to your HRM repo clone}"
PYTHON="${PYTHON:?set PYTHON to your venv python}"
DATA_PATH="data/arc-2-aug-1000"
CKPT_DIR="$HRM_DIR/published_checkpoints/HRM-checkpoint-ARC-2"
CKPT_FILE="$CKPT_DIR/checkpoint"   # adjust if HuggingFace uses a different filename

# Need ONLINE HuggingFace access for the download.
export HF_HUB_OFFLINE=0
SKIP_DATA_CHECK=1   # we may need to build the dataset below
source "$(dirname "$0")/_preflight.sh"

cd "$HRM_DIR"

# Step 1: Download the published checkpoint (if not already cached).
if [[ ! -f "$CKPT_FILE" ]]; then
    echo "[1/4] Downloading sapientinc/HRM-checkpoint-ARC-2 ..."
    "$PYTHON" -c "
from huggingface_hub import snapshot_download
import os
out = snapshot_download(repo_id='sapientinc/HRM-checkpoint-ARC-2', local_dir='$CKPT_DIR')
print('downloaded to', out)
print('contents:', os.listdir(out))
"
    # The checkpoint file's exact name depends on HF repo layout; list the
    # contents and let the user point CKPT_FILE at the right file if needed.
    if [[ ! -f "$CKPT_FILE" ]]; then
        echo
        echo "ERROR: $CKPT_FILE not found after download. List the downloaded files:"
        find "$CKPT_DIR" -maxdepth 3 -type f
        echo
        echo "Set CKPT_FILE to the actual checkpoint file (likely 'all_config.yaml's"
        echo "sibling 'step_<N>'), then re-run."
        exit 1
    fi
fi

# Step 2: Build ARC-AGI-2 dataset if missing.
#
# DO NOT pass --dataset-dirs to build_arc_dataset.py: argdantic *appends*
# the CLI value to the pydantic default ["ARC-AGI/data", "ConceptARC/corpus"]
# rather than replacing it, so the resulting `data/arc-2-aug-1000` ends up
# containing ARC-1 + ConceptARC + ARC-AGI-2, producing
# num_puzzle_identifiers ≈ 2x the published checkpoint's puzzle_emb shape
# and a load_state_dict failure at eval time. Instead, swap the script's
# defaults in-place so the build uses ONLY ARC-AGI-2 raw data.
if [[ ! -f "$DATA_PATH/identifiers.json" ]]; then
    echo "[2/4] Building $DATA_PATH ..."
    if [[ ! -d "dataset/raw-data/ARC-AGI-2/data" ]]; then
        git submodule update --init --recursive
    fi
    # Swap the ARC-1 default to ARC-2 in build_arc_dataset.py if not already.
    sed -i \
        -e 's|^    dataset_dirs: List\[str\] = \["dataset/raw-data/ARC-AGI/data", "dataset/raw-data/ConceptARC/corpus"\]|    # dataset_dirs: List[str] = ["dataset/raw-data/ARC-AGI/data", "dataset/raw-data/ConceptARC/corpus"]|' \
        -e 's|^    output_dir: str = "data/arc-aug-1000"|    # output_dir: str = "data/arc-aug-1000"|' \
        -e 's|^    # dataset_dirs: List\[str\] = \["dataset/raw-data/ARC-AGI-2/data"\]|    dataset_dirs: List[str] = ["dataset/raw-data/ARC-AGI-2/data"]|' \
        -e 's|^    # output_dir: str = "data/arc-2-aug-1000"|    output_dir: str = "data/arc-2-aug-1000"|' \
        dataset/build_arc_dataset.py
    "$PYTHON" dataset/build_arc_dataset.py    # no --dataset-dirs (default is now ARC-2 only)
fi

# Step 3: Run evaluate.py to produce <ckpt>_all_preds.<rank> files.
# DISABLE_COMPILE=1: the published checkpoint was saved WITHOUT torch.compile
# (state-dict keys have no _orig_mod. prefix), so wrap the model uncompiled
# to match.
echo "[3/4] Running evaluate.py on the published checkpoint ..."
DISABLE_COMPILE=1 "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
    evaluate.py checkpoint="$CKPT_FILE"

# Step 4: Run paper-protocol voting eval.
echo "[4/4] Running arc_eval_script.py ..."
# Copy our script into HRM_DIR so its imports (from dataset.common) work.
cp "$(dirname "$0")/arc_eval_script.py" "$HRM_DIR/arc_eval_script.py"
"$PYTHON" arc_eval_script.py --data "$DATA_PATH" --checkpoint "$CKPT_FILE"

cat <<'EOF'

============================================================================
Expected: K=2 ≈ 5.0% on ARC-AGI-2 (HRM paper headline).
If K=2 is within ±1 pp of 5%, the eval pipeline is reproducing paper. Proceed
to experiment 2 (training pipeline smoke test).

If K=2 is far from 5% (e.g. < 3% or > 8%), the eval pipeline is wrong; do
NOT proceed to training until we understand why.
============================================================================
EOF
