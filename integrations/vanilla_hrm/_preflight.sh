#!/bin/bash
# Shared preflight check — sourced by every script in this directory.
# Aborts the script (exits the parent) if any prerequisite is missing.
#
# Why this exists: HRM's published pip package adam_atan2==0.0.3 has broken
# metadata, so `pip install adam_atan2` fails on PyPI version-mismatch even
# though the source builds fine. In addition, the C-extension build silently
# fails when CUDA toolkit isn't on PATH, and the Python wrapper happily
# imports but throws ModuleNotFoundError on adam_atan2_backend the moment
# the optimizer is constructed. That fallback to AdamW silently breaks
# reproduction. This preflight catches all of that BEFORE the run starts.

set -e

# 1. Standard environment variables every HRM run needs.
export OMP_NUM_THREADS=8
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"   # need ONLINE for the eval-pipeline validation script

PYTHON="${PYTHON:-python}"

# 2. adam_atan2 C-extension importability check — THE single most important
# correctness gate. Without this, pretrain.py silently uses AdamW instead of
# AdamATan2 and the trained model will not match the paper.
# `import torch` first to populate rpaths for libc10/libtorch, matching how
# pretrain.py loads them in practice.
if ! "$PYTHON" -c "import torch; import adam_atan2_backend" 2>/dev/null; then
    cat <<'EOF' 1>&2

================================================================================
FATAL: adam_atan2_backend (the C-extension) is NOT importable in this env.

HRM uses AdamATan2 as the main optimizer; pretrain.py's `from adam_atan2
import AdamATan2` will silently get a broken wrapper that raises at first
call. Either you'll see a runtime crash, or — if pretrain.py has been
patched with a fallback — a SILENT switch to AdamW that ruins reproduction.

Fix: build the C-extension from source against your local CUDA toolchain.

    # Pin CUDA module that matches your torch build (torch.version.cuda):
    module load cudatoolkit/12.6        # adjust to your cluster
    # Download HRM's published sdist (PyPI metadata is broken; install
    # from the local tarball after extracting it):
    cd /tmp && rm -rf adam_atan2-0.0.3*
    curl -sL -O https://files.pythonhosted.org/packages/06/a2/8ef8f0bb022f39d87edda9e04e77458195eb0011d13561c4e7a85e6c1c22/adam_atan2-0.0.3.tar.gz
    tar xzf adam_atan2-0.0.3.tar.gz
    cd adam_atan2-0.0.3
    SETUPTOOLS_SCM_PRETEND_VERSION=0.0.3 \
        $PYTHON -m pip install --no-cache-dir --no-build-isolation .

After install, verify:
    $PYTHON -c "import adam_atan2_backend; print(adam_atan2_backend.__file__)"
should print a .so path. Then re-run this script.
================================================================================
EOF
    exit 1
fi

# 3. pretrain.py must NOT have the AdamW fallback — verify the upstream
# version is in place. The diff is small, just one line `from adam_atan2
# import AdamATan2` at module load. Pretrain.py with a try/except fallback
# is a sign someone patched it; refuse to run since we cannot guarantee
# AdamATan2 actually got selected.
if grep -q "from torch.optim import AdamW as AdamATan2" "$HRM_DIR/pretrain.py" 2>/dev/null; then
    echo "FATAL: $HRM_DIR/pretrain.py has the AdamW fallback patch. Restore upstream pretrain.py before running."
    exit 1
fi

# 4. Dataset present (skipped by validate_eval which downloads on the fly).
# Skip this check if SKIP_DATA_CHECK=1 (used by setup-only scripts).
if [[ "${SKIP_DATA_CHECK:-0}" != "1" ]]; then
    if [[ ! -f "$HRM_DIR/$DATA_PATH/identifiers.json" ]]; then
        cat <<EOF 1>&2

FATAL: $HRM_DIR/$DATA_PATH/identifiers.json missing. Build the dataset first:

    cd $HRM_DIR
    git submodule update --init --recursive   # gets ARC-AGI, ARC-AGI-2, ConceptARC raw data
    # ARC-1 (default args produce data/arc-aug-1000):
    $PYTHON dataset/build_arc_dataset.py
    # ARC-2 — DO NOT pass --dataset-dirs; argdantic APPENDS it to the default
    # [ARC-AGI/data, ConceptARC/corpus] rather than replacing, which would
    # build a mixed dataset producing num_puzzle_identifiers ≈ 2x the
    # published checkpoint's puzzle_emb shape. Sed-swap the script's
    # defaults in-place to ARC-2, then call with no flag:
    sed -i \\
      -e 's|^    dataset_dirs: List\\[str\\] = \\["dataset/raw-data/ARC-AGI/data", "dataset/raw-data/ConceptARC/corpus"\\]|    # dataset_dirs: List[str] = ["dataset/raw-data/ARC-AGI/data", "dataset/raw-data/ConceptARC/corpus"]|' \\
      -e 's|^    output_dir: str = "data/arc-aug-1000"|    # output_dir: str = "data/arc-aug-1000"|' \\
      -e 's|^    # dataset_dirs: List\\[str\\] = \\["dataset/raw-data/ARC-AGI-2/data"\\]|    dataset_dirs: List[str] = ["dataset/raw-data/ARC-AGI-2/data"]|' \\
      -e 's|^    # output_dir: str = "data/arc-2-aug-1000"|    output_dir: str = "data/arc-2-aug-1000"|' \\
      dataset/build_arc_dataset.py
    $PYTHON dataset/build_arc_dataset.py
    # If you need ARC-1 again afterwards, swap the defaults back manually.
EOF
        exit 1
    fi
fi

# 5. CUDA visibility.
if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available(); print(f'GPUs: {torch.cuda.device_count()}')" ; then
    echo "FATAL: torch.cuda.is_available() is False. Allocate GPUs before running."
    exit 1
fi

echo "[preflight] OK — adam_atan2_backend present, pretrain.py unmodified, dataset OK, CUDA available."
