#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -t 48:00:00
#SBATCH --exclude=server13,server46,server59
#SBATCH -J motif_train
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_motif_extension/logs/motif_train_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_motif_extension/logs/motif_train_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_PATH=${CONFIG_PATH:?CONFIG_PATH must identify a motif-extension config}
LOG_DIR=/public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_motif_extension/logs
CONDA_ENV_NAME=${CONDA_ENV_NAME:-genmol}

source ~/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_NAME}"
export PATH=/opt/gridview/slurm/bin:$PATH

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

export HF_HOME=/tmp/xinwuye_hf_motif_${SLURM_JOB_ID}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="${WANDB_PROJECT:-tools4sys}"
export WANDB_NAME="${WANDB_NAME:-motif-extension-${SLURM_JOB_ID}}"

MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
accelerate launch \
  --config_file configs/accelerate_zero2.yaml \
  --num_processes 8 \
  --main_process_port "${MASTER_PORT}" \
  scripts/train_cpgrpo_denovo.py \
  --config "${CONFIG_PATH}"
