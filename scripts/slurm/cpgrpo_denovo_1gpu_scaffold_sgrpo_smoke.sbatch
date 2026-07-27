#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH --exclude=server13
#SBATCH -J dnscafsmk
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_denovo/logs/dnscafsmk_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_denovo/logs/dnscafsmk_%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/genmol}
CONFIG_PATH=${CONFIG_PATH:-configs/cpgrpo_denovo_sgrpo_scaffold_ng64_sg8_bs1024_rewardsum_loo_1gpu_smoke10.yaml}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-genmol}
LOG_DIR=/public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_denovo/logs

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

export HF_HOME=/tmp/xinwuye_hf_dnscafsmk_${SLURM_JOB_ID}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
accelerate launch \
  --config_file configs/accelerate_zero2.yaml \
  --num_processes 1 \
  --main_process_port "${MASTER_PORT}" \
  scripts/train_cpgrpo_denovo.py \
  --config "${CONFIG_PATH}"
