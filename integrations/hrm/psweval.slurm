#!/bin/bash
# SLURM wrapper for psweep_eval.sh — evaluate one sweep checkpoint (4 GPU).
# Usage (from HRM_DIR, scripts deployed at HRM_DIR root):
#   sbatch --export=ALL,ARCH=hrm_v1_dl_h013_noemb psweval.slurm
#   sbatch --export=ALL,RUN_DIR="checkpoints/.../... <coolname>" psweval.slurm
#   sbatch --export=ALL,ARCH=...,STEP=259035 psweval.slurm   # specific ckpt
#SBATCH --job-name=psweval
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
bash "$SLURM_SUBMIT_DIR/psweep_eval.sh"
