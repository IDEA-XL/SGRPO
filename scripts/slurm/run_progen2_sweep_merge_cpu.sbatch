#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J p2sweepmerge
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_progen2sweepmerge_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_progen2sweepmerge_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_PATH=${CONFIG_PATH:?CONFIG_PATH is required}

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:$PATH

cd "${REPO_DIR}"
python scripts/progen2_sweep_pipeline.py merge-packed-gpu-rewards \
  --config "${CONFIG_PATH}"
