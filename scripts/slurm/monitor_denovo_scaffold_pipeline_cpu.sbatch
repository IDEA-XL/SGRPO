#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 5-00:00:00
#SBATCH -J dnscafmon
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/denovo_scaffold_monitor/monitor_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/denovo_scaffold_monitor/monitor_%j.err

set -euo pipefail

: "${TRAINING_JOB_ID:?TRAINING_JOB_ID is required}"
: "${LAUNCHER_JOB_ID:?LAUNCHER_JOB_ID is required}"

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
OUTPUT_DIR=/public/home/xinwuye/ai4s-tool-joint-train/runs/denovo_scaffold_monitor

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}"
python -u scripts/monitor_denovo_scaffold_pipeline.py \
  --training-job-id "${TRAINING_JOB_ID}" \
  --launcher-job-id "${LAUNCHER_JOB_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --poll-seconds 120 \
  --stale-seconds 1800 \
  --first-step-grace-seconds 1800
