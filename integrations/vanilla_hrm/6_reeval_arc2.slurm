#!/bin/bash
# SLURM submission for experiment 6 — published ARC-2 checkpoint re-eval on the
# verified reference dataset. Assumes the package is deployed at
# <HRM_DIR>/repro_vanilla/ and you sbatch from <HRM_DIR>.
#SBATCH --job-name=a2reeval
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=jobs/logs/%x-%j.out
#SBATCH --error=jobs/logs/%x-%j.err

set -euo pipefail
mkdir -p jobs/logs

export HRM_DIR="${HRM_DIR:-$SLURM_SUBMIT_DIR}"
export PYTHON="${PYTHON:?set PYTHON}"
module load cudatoolkit/12.6 2>/dev/null || true

# $SLURM_SUBMIT_DIR, NOT dirname $0 (which resolves to the SLURM spool copy).
bash "$SLURM_SUBMIT_DIR/repro_vanilla/6_reeval_arc2.sh"
