#!/bin/bash
# p-sweep TRAIN wrapper — DeepLoop H-only p=0.05, no embed RMSNorm. Train only; eval via psweval.slurm afterwards.
# Deploy package files to <HRM_DIR> root, then: cd <HRM_DIR> && sbatch pst_h005.slurm
#SBATCH --job-name=pst_h005
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

export HRM_DIR="${HRM_DIR:-$SLURM_SUBMIT_DIR}"
export PYTHON="${PYTHON:?set PYTHON}"
export ARCH=hrm_v1_dl_h005_noemb
module load cudatoolkit/12.6 2>/dev/null || true

# $SLURM_SUBMIT_DIR, NOT dirname $0 (which resolves to the SLURM spool copy).
bash "$SLURM_SUBMIT_DIR/psweep_train.sh"
