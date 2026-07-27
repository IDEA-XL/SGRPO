#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:30:00
#SBATCH -J bxmaterialize
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/baseline_expansion_sweep/logs/materialize_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/baseline_expansion_sweep/logs/materialize_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
RUN_ROOT=/public/home/xinwuye/ai4s-tool-joint-train/runs/baseline_expansion_sweep

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol

cd "${REPO_DIR}"
python -u scripts/materialize_baseline_expansion_results.py \
  --run-root "${RUN_ROOT}" \
  --output-root "${RUN_ROOT}/materialized-results"
