#!/bin/bash
# SLURM submission for experiment 4 — full vanilla ARC-AGI-2 cross-check.
#SBATCH --job-name=vh_arc2
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=36:00:00
#SBATCH --output=jobs/logs/%x-%j.out
#SBATCH --error=jobs/logs/%x-%j.err

set -euo pipefail
mkdir -p jobs/logs

export HRM_DIR="${HRM_DIR:?set HRM_DIR}"
export PYTHON="${PYTHON:?set PYTHON}"

module load cudatoolkit/12.6 2>/dev/null || true

bash "$SLURM_SUBMIT_DIR/repro_vanilla/4_full_train_arc2.sh"
