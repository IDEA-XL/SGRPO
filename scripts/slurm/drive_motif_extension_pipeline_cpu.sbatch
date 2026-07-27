#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 5-00:00:00
#SBATCH --exclude=server13,server46,server59
#SBATCH -J motif_driver
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/motif_extension/logs/controller_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/motif_extension/logs/controller_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
SMOKE_JOB_ID=${SMOKE_JOB_ID:?SMOKE_JOB_ID is required}

source ~/miniconda/etc/profile.d/conda.sh
conda activate genmol
cd "${REPO_DIR}"
mkdir -p /public/home/xinwuye/ai4s-tool-joint-train/runs/motif_extension/logs
export PYTHONPATH=src
python -u scripts/drive_motif_extension_pipeline.py \
  --smoke-job-id "${SMOKE_JOB_ID}" \
  --gpu-submit-limit 40 \
  --poll-seconds 10
