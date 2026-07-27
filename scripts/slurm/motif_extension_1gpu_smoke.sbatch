#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH --exclude=server13,server46,server59
#SBATCH -J motif_smoke
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_motif_extension/logs/motif_smoke_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_motif_extension/logs/motif_smoke_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_PATH=${CONFIG_PATH:-configs/cpgrpo_motif_extension_sgrpo_smoke2.yaml}

source ~/miniconda/etc/profile.d/conda.sh
conda activate genmol
cd "${REPO_DIR}"
mkdir -p /public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_motif_extension/logs
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/train_cpgrpo_denovo.py --config "${CONFIG_PATH}"
