#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J mmscaf3
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/mmgenmol_scaffold_three_model_logs/reaggregate_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/mmgenmol_scaffold_three_model_logs/reaggregate_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PYTHONPATH=${REPO_DIR}/src

cd "${REPO_DIR}"
python -u scripts/reaggregate_mmgenmol_scaffold_three_model.py
