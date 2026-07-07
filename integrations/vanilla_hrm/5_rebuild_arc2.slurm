#!/bin/bash
# SLURM submission for experiment 5 — deterministic ARC-2 dataset rebuild.
# CPU-only; needs ~120 GB RAM. Assumes the package is deployed at
# <HRM_DIR>/repro_vanilla/ and you sbatch from <HRM_DIR>.
#SBATCH --job-name=a2build
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/%x-%j.out
#SBATCH --error=jobs/logs/%x-%j.err

set -euo pipefail
mkdir -p jobs/logs

export HRM_DIR="${HRM_DIR:-$SLURM_SUBMIT_DIR}"
export PYTHON="${PYTHON:?set PYTHON}"

# $SLURM_SUBMIT_DIR, NOT dirname $0 (which resolves to the SLURM spool copy).
bash "$SLURM_SUBMIT_DIR/repro_vanilla/5_rebuild_arc2.sh"
