#!/bin/bash
# ============================================================================
# EXPERIMENT 6 — re-evaluate the published ARC-2 checkpoint on the verified
# reference dataset (the corrected version of experiment 1).
#
# Hard-gates on the dataset hash BEFORE spending GPU time: the published
# checkpoint's per-augmentation puzzle_emb is welded to its training build, so
# evaluating on any other build silently degrades K=2 (~5% -> ~1.7%). Run
# experiment 5 first if the gate fails.
#
# Eval only — no training. ~30-45 min on 4 GPUs.
#
# Expected (count-first paper-protocol voting):
#   K=2 = 5.00% +/- 1 puzzle (+/-0.83%, cross-hardware bf16 jitter)
#   K=1000 ~ 5.8%
# If K=2 < 3.3%: the dataset gate was bypassed or the checkpoint is wrong —
# do not trust the number.
# ============================================================================
set -euo pipefail

HRM_DIR="${HRM_DIR:?set HRM_DIR to your HRM clone}"
PYTHON="${PYTHON:?set PYTHON to the venv python}"
DATA_PATH="${DATA_PATH:-data/arc-2-aug-1000}"
PKG_DIR="$(cd "$(dirname "$0")" && pwd)"
NPROC="${NPROC:-4}"
MPORT="${MPORT:-$((20000 + (${SLURM_JOB_ID:-$$} % 20000)))}"

cd "$HRM_DIR"

# ---- Gate 1: dataset must be the verified reference build -------------------
echo "[6E] verifying $DATA_PATH against reference hashes (test split) ..."
"$PYTHON" - "$PKG_DIR" "$DATA_PATH" <<'PYEOF'
import hashlib, json, os, sys
pkg, data = sys.argv[1], sys.argv[2]
ref = json.load(open(os.path.join(pkg, "reference_hashes.json")))["arc2"]["files"]
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
# Eval only touches identifiers + test split; gate on exactly those.
needed = [r for r in ref if r == "identifiers.json" or r.startswith("test/")]
bad = [r for r in needed
       if not os.path.isfile(os.path.join(data, r)) or md5(os.path.join(data, r)) != ref[r]]
if bad:
    print(f"FATAL: {data} is not the reference build (mismatch: {bad}).")
    print("Run experiment 5 (5_rebuild_arc2) first — evaluating the published")
    print("checkpoint on a divergent build produces misleading K=2 numbers.")
    sys.exit(1)
print("dataset gate: PASS (byte-identical to reference)")
PYEOF

# ---- Gate 2: published checkpoint present (download if we have network) -----
CKPT_DIR="published_checkpoints/HRM-checkpoint-ARC-2"
CKPT_FILE="$CKPT_DIR/checkpoint"
if [[ ! -f "$CKPT_FILE" ]]; then
    echo "[6E] downloading sapientinc/HRM-checkpoint-ARC-2 ..."
    HF_HUB_OFFLINE=0 "$PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='sapientinc/HRM-checkpoint-ARC-2', local_dir='$CKPT_DIR')"
fi
[[ -f "$CKPT_DIR/all_config.yaml" ]] || { echo "FATAL: $CKPT_DIR/all_config.yaml missing"; exit 1; }

# ---- Eval --------------------------------------------------------------------
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export WANDB_MODE=offline
export OMP_NUM_THREADS=8

# DISABLE_COMPILE=1: the published checkpoint was saved WITHOUT torch.compile.
echo "[6E] evaluate.py (preds regeneration, $NPROC GPU, port $MPORT) ..."
DISABLE_COMPILE=1 "$PYTHON" -m torch.distributed.run --nnodes=1 --nproc_per_node="$NPROC" \
    --master_addr=127.0.0.1 --master_port="$MPORT" \
    evaluate.py checkpoint="$CKPT_FILE"

echo "[6E] paper-protocol voting (count-first ranking + a1 dedup fix) ..."
PYTHONPATH="$HRM_DIR" "$PYTHON" "$PKG_DIR/arc_eval_script.py" \
    --data "$DATA_PATH" --checkpoint "$CKPT_FILE"

cat <<'EOF'

============================================================================
Expected on the verified reference build:
  K=1 ~ 3.3%   K=2 = 5.00% +/- 1 puzzle   K=1000 ~ 5.8%
K=2 in [4.2, 5.8]%  -> aligned with the reference/paper protocol. Done.
K=2 < 3.3%          -> something is still divergent; check the two gates above
                       actually ran (do NOT bypass them) and the checkpoint md5.
Note: K=1000 (~5%) is the build-robust headline; K=2 carries +/-1-puzzle
cross-hardware jitter at this noise floor.
============================================================================
EOF
