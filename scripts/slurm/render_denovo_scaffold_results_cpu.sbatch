#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J dnscafrender
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/denovo_scaffold_sweep/logs/render_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/denovo_scaffold_sweep/logs/render_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
RUN_ROOT=/public/home/xinwuye/ai4s-tool-joint-train/runs/denovo_scaffold_sweep
OUTPUT_DIR=${RUN_ROOT}/results

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol

cd "${REPO_DIR}"
python -u vis-code/render_denovo_scaffold_results.py \
  --run-root "${RUN_ROOT}" \
  --output-dir "${OUTPUT_DIR}"
