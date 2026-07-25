#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 7-00:00:00
#SBATCH -J bxdriver
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/baseline_expansion_sweep/logs/controller_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/baseline_expansion_sweep/logs/controller_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
RUN_ROOT=/public/home/xinwuye/ai4s-tool-joint-train/runs/baseline_expansion_sweep

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:${PATH}
export REBUTTAL_RUN_OUTPUT_ROOT="${RUN_ROOT}"

cd "${REPO_DIR}"
python -u scripts/drive_rebuttal_dense_sweeps.py
