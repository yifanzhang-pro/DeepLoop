#!/bin/bash
# ============================================================================
# EXPERIMENT 5 — deterministic ARC-AGI-2 dataset rebuild (hash-verified).
#
# Rebuilds data/arc-2-aug-1000 so that it is BYTE-IDENTICAL on any machine to
# the reference build — the one that reproduces the paper's K=2 = 5.0% on the
# published sapientinc/HRM-checkpoint-ARC-2. Upstream build_arc_dataset.py is
# NOT byte-reproducible across machines (it feeds filesystem-ordered file
# lists into a seeded RNG; two same-seed builds were measured to share only
# 0.14% of augmentations — see h010nf_voting_recompute.md, CLOSED section).
# This script pins the enumeration order via a recorded manifest and verifies
# every output file's md5. It either produces THE reference dataset or fails
# loudly — silent divergence is impossible.
#
# Requirements: ~120 GB RAM (use a fat CPU node / the .slurm wrapper), no GPU.
# Runtime: ~10-20 min.
# ============================================================================
set -euo pipefail

HRM_DIR="${HRM_DIR:?set HRM_DIR to your HRM clone}"
PYTHON="${PYTHON:?set PYTHON to the venv python}"
OUTPUT_DIR="${OUTPUT_DIR:-data/arc-2-aug-1000}"
PKG_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$HRM_DIR"

# Raw data present?
if [[ ! -d "dataset/raw-data/ARC-AGI-2/data" ]]; then
    echo "[5R] raw data missing -> git submodule update --init --recursive"
    git submodule update --init --recursive
fi

# If the output already exists, verify instead of rebuilding; refuse to
# overwrite a divergent build silently (the user may want to inspect it).
if [[ -f "$OUTPUT_DIR/identifiers.json" ]]; then
    echo "[5R] $OUTPUT_DIR already exists — verifying against reference hashes ..."
    if "$PYTHON" - "$PKG_DIR" "$OUTPUT_DIR" <<'PYEOF'
import hashlib, json, os, sys
pkg, out = sys.argv[1], sys.argv[2]
ref = json.load(open(os.path.join(pkg, "reference_hashes.json")))["arc2"]["files"]
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()
bad = [r for r, e in ref.items()
       if not os.path.isfile(os.path.join(out, r)) or md5(os.path.join(out, r)) != e]
print("hash check:", "PASS" if not bad else f"FAIL ({bad})")
sys.exit(0 if not bad else 1)
PYEOF
    then
        echo "[5R] existing $OUTPUT_DIR is ALREADY the reference build — nothing to do."
        exit 0
    else
        echo "[5R] FATAL: existing $OUTPUT_DIR DIFFERS from the reference build."
        echo "     It was probably built with the unsorted-enumeration upstream script"
        echo "     on this machine. Move it away and re-run:"
        echo "         mv '$OUTPUT_DIR' '${OUTPUT_DIR}.divergent.bak'"
        exit 1
    fi
fi

echo "[5R] rebuilding $OUTPUT_DIR with manifest-pinned enumeration ..."
"$PYTHON" "$PKG_DIR/build_from_manifest.py" \
    --hrm-dir "$HRM_DIR" --dataset arc2 --output-dir "$OUTPUT_DIR"

cat <<'EOF'

============================================================================
Dataset is byte-identical to the reference build (all md5 verified).
Next: experiment 6 — re-evaluate the published ARC-2 checkpoint on it:
    sbatch repro_vanilla/6_reeval_arc2.slurm
Expected: K=2 = 5.00% +/- 1 puzzle (0.83%), K=1000 ~ 5.8%.
============================================================================
EOF
