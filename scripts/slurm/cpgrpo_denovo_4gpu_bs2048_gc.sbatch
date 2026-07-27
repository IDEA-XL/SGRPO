#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -t 72:00:00
#SBATCH --exclude=server13
#SBATCH -J dnscaf4g
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_denovo/logs/dnscaf4g_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_denovo/logs/dnscaf4g_%j.err

set -euo pipefail

: "${CONFIG_PATH:?CONFIG_PATH is required}"

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
LOG_DIR=/public/home/xinwuye/ai4s-tool-joint-train/runs/cpgrpo_denovo/logs
MIHOMO_DIR=/public/home/xinwuye/mihomo
CONDA_ENV_NAME=${CONDA_ENV_NAME:-genmol}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-}
WANDB_NETRC_PATH="${WANDB_NETRC_PATH:-$HOME/.netrc}"

if [[ -n "${CONDA_ENV_NAME}" && -n "${CONDA_ENV_PREFIX}" ]]; then
  echo "Set at most one of CONDA_ENV_NAME and CONDA_ENV_PREFIX" >&2
  exit 1
fi
if [[ ! -f "${REPO_DIR}/${CONFIG_PATH}" ]]; then
  echo "Training config not found: ${REPO_DIR}/${CONFIG_PATH}" >&2
  exit 1
fi
if [[ -z "${WANDB_API_KEY:-}" && ! -f "${WANDB_NETRC_PATH}" ]]; then
  echo "Either WANDB_API_KEY or ~/.netrc is required for wandb auth" >&2
  exit 1
fi

CONFIG_STEM=$(basename "${CONFIG_PATH}" .yaml)
export WANDB_PROJECT="${WANDB_PROJECT:-tools4sys}"
export WANDB_NAME="${WANDB_NAME:-${CONFIG_STEM}-slurm${SLURM_JOB_ID}}"

source ~/miniconda/etc/profile.d/conda.sh
if [[ -n "${CONDA_ENV_PREFIX}" ]]; then
  if [[ ! -d "${CONDA_ENV_PREFIX}" ]]; then
    echo "Conda env prefix not found: ${CONDA_ENV_PREFIX}" >&2
    exit 1
  fi
  conda activate "${CONDA_ENV_PREFIX}"
else
  conda activate "${CONDA_ENV_NAME}"
fi
export PATH=/opt/gridview/slurm/bin:${PATH}

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

export HF_HOME=/tmp/xinwuye_hf_cpgrpo_${SLURM_JOB_ID}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ "${http_proxy:-}${https_proxy:-}${all_proxy:-}" == *"127.0.0.1:7897"* ]]; then
  if ! pgrep -u "$USER" -f "${MIHOMO_DIR}/mihomo" >/dev/null 2>&1; then
    nohup "${MIHOMO_DIR}/mihomo" -d "${MIHOMO_DIR}" >/tmp/mihomo_${SLURM_JOB_ID}.log 2>&1 &
    sleep 5
  fi
fi

MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
accelerate launch \
  --config_file configs/accelerate_zero2.yaml \
  --num_processes 4 \
  --main_process_port "${MASTER_PORT}" \
  scripts/train_cpgrpo_denovo.py \
  --config "${CONFIG_PATH}"
