#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 1-00:00:00
#SBATCH -J rbdriver
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/rebuttal_dense_sweep/logs/controller_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/rebuttal_dense_sweep/logs/controller_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:${PATH}

cd "${REPO_DIR}"
python -u scripts/drive_rebuttal_dense_sweeps.py
