#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -t 7-00:00:00
#SBATCH -J mm100ctl

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
RUN_ROOT=${MMGENMOL_100X_RUN_ROOT:-/public/home/xinwuye/ai4s-tool-joint-train/runs/rebuttal_mmgenmol_100x_sweep}

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:${PATH}
export MMGENMOL_100X_RUN_ROOT="${RUN_ROOT}"

cd "${REPO_DIR}"
python scripts/drive_mmgenmol_100x_sweep.py
