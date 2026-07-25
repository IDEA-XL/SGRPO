#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:30:00
#SBATCH -J bxlaunch
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/baseline_expansion_sweep/logs/launcher_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/baseline_expansion_sweep/logs/launcher_%j.err

set -euo pipefail

: "${DENOVO_DMB_JOB_ID:?DENOVO_DMB_JOB_ID is required}"
: "${DENOVO_ENTROPY_JOB_ID:?DENOVO_ENTROPY_JOB_ID is required}"
: "${MMGENMOL_DMB_JOB_ID:?MMGENMOL_DMB_JOB_ID is required}"
: "${MMGENMOL_ENTROPY_JOB_ID:?MMGENMOL_ENTROPY_JOB_ID is required}"
: "${PROGEN2_DMB_JOB_ID:?PROGEN2_DMB_JOB_ID is required}"
: "${PROGEN2_ENTROPY_JOB_ID:?PROGEN2_ENTROPY_JOB_ID is required}"

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:${PATH}

cd "${REPO_DIR}"
python -u scripts/launch_baseline_expansion_sweeps.py \
  --denovo-dmb-job-id "${DENOVO_DMB_JOB_ID}" \
  --denovo-entropy-job-id "${DENOVO_ENTROPY_JOB_ID}" \
  --mmgenmol-dmb-job-id "${MMGENMOL_DMB_JOB_ID}" \
  --mmgenmol-entropy-job-id "${MMGENMOL_ENTROPY_JOB_ID}" \
  --progen2-dmb-job-id "${PROGEN2_DMB_JOB_ID}" \
  --progen2-entropy-job-id "${PROGEN2_ENTROPY_JOB_ID}"
