#!/bin/bash
# SLURM submission for experiment 1 — eval-pipeline validation against the
# paper-published HRM-checkpoint-ARC-2.
#SBATCH --job-name=vh_validate
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=jobs/logs/%x-%j.out
#SBATCH --error=jobs/logs/%x-%j.err

set -euo pipefail
mkdir -p jobs/logs

# Fill these in for your environment (or export before sbatch).
export HRM_DIR="${HRM_DIR:?set HRM_DIR}"
export PYTHON="${PYTHON:?set PYTHON}"

# CUDA toolkit for any on-the-fly C-extension imports.
module load cudatoolkit/12.6 2>/dev/null || true

bash "$SLURM_SUBMIT_DIR/repro_vanilla/1_validate_eval.sh"
