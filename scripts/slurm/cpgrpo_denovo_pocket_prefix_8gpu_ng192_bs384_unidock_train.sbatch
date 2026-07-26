#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH -t 48:00:00
#SBATCH -J mmuni8t
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/unidock_official/logs/train_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/unidock_official/logs/train_%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/genmol}
CONFIG_PATH=${CONFIG_PATH:-configs/cpgrpo_denovo_pocket_prefix_ng192_bs384_lr5e-5_beta5e-3_ni1_q03_sa02_unidock05.yaml}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-/public/home/xinwuye/ai4s-tool-joint-train/runs/unidock_official/conda_env}
GENMOL_REWARD_WORKERS=${GENMOL_REWARD_WORKERS:-7}
WANDB_NETRC_PATH=${WANDB_NETRC_PATH:-${HOME}/.netrc}
LOG_DIR=/public/home/xinwuye/ai4s-tool-joint-train/runs/unidock_official/logs
MIHOMO_DIR=/public/home/xinwuye/mihomo

if [[ -z "${WANDB_API_KEY:-}" && ! -f "${WANDB_NETRC_PATH}" ]]; then
  echo "Either WANDB_API_KEY or ~/.netrc is required for wandb auth" >&2
  exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-tools4sys}"
export WANDB_NAME="${WANDB_NAME:-cpgrpo-denovo-pocket-prefix-8gpu-ng192-bs384-unidock}"
export WANDB_DISABLE_SERVICE="${WANDB_DISABLE_SERVICE:-true}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV_PREFIX}"
export PATH=/opt/gridview/slurm/bin:$PATH

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

export HF_HOME=/tmp/xinwuye_hf_mmunidock8_${SLURM_JOB_ID}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export TRITON_CACHE_DIR=${HF_HOME}/triton_cache
export TMPDIR=/tmp/xinwuye_tmp_mmunidock8_${SLURM_JOB_ID}
export TMP=${TMPDIR}
export TEMP=${TMPDIR}
export PYTHONPATH=src
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GENMOL_REWARD_WORKERS
mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TMPDIR}"

if [[ "${http_proxy:-}${https_proxy:-}${all_proxy:-}" == *"127.0.0.1:7897"* ]]; then
  if ! pgrep -u "$USER" -f "${MIHOMO_DIR}/mihomo" >/dev/null 2>&1; then
    nohup "${MIHOMO_DIR}/mihomo" -d "${MIHOMO_DIR}" >/tmp/mihomo_${SLURM_JOB_ID}.log 2>&1 &
    sleep 5
  fi
fi

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  wandb login --relogin "${WANDB_API_KEY}" >/dev/null 2>&1 || true
fi

MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))
accelerate launch \
  --config_file configs/accelerate_zero2.yaml \
  --num_processes 8 \
  --main_process_port "${MASTER_PORT}" \
  scripts/train_cpgrpo_denovo_pocket_prefix.py \
  --config "${CONFIG_PATH}"
