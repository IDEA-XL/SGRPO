#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J dncrfinal
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/denovo_credit_ablation_sweep/logs/finalize_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/denovo_credit_ablation_sweep/logs/finalize_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
RUN_ROOT=/public/home/xinwuye/ai4s-tool-joint-train/runs/denovo_credit_ablation_sweep

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:${PATH}
export PYTHONPATH=${REPO_DIR}/src
export MPLBACKEND=Agg

cd "${REPO_DIR}"
python -u scripts/assemble_rebuttal_dense_denovo.py \
  --run-root "${RUN_ROOT}" \
  --expected-experiments 3
python -u scripts/materialize_denovo_credit_ablation_results.py \
  --run-root "${RUN_ROOT}"
