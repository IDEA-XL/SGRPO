#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:30:00
#SBATCH -J dnscaflaunch
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_denovo/logs/dnscaflaunch_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_denovo/logs/dnscaflaunch_%j.err

set -euo pipefail

: "${SGRPO_JOB_ID:?SGRPO_JOB_ID is required}"

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:${PATH}

cd "${REPO_DIR}"
python -u scripts/launch_denovo_scaffold_sweep.py \
  --sgrpo-job-id "${SGRPO_JOB_ID}"
