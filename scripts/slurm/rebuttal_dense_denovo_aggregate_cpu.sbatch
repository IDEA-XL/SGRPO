#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -J rbdenagg
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/rebuttal_dense_sweep/logs/denovo_aggregate_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/rebuttal_dense_sweep/logs/denovo_aggregate_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
RUN_ROOT=${RUN_ROOT:-/public/home/xinwuye/ai4s-tool-joint-train/runs/rebuttal_dense_sweep}
EXPECTED_EXPERIMENTS=${EXPECTED_EXPERIMENTS:-12}

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:${PATH}

cd "${REPO_DIR}"
python scripts/assemble_rebuttal_dense_denovo.py \
  --run-root "${RUN_ROOT}" \
  --expected-experiments "${EXPECTED_EXPERIMENTS}"
