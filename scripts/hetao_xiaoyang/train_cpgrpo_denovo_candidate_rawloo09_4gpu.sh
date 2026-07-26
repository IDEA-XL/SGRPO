#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

ENV_PREFIX="${SGRPO_ENV_PREFIX:-/mnt/afs/250010210/xinwu/envs/sgrpo}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/configs/cpgrpo_denovo_candidate_rawloo09_ng512_bs2048_lr5e-5_beta5e-3_ni1_ms2000_4gpu.yaml}"
TOKENIZER_PATH="${GENMOL_SAFE_TOKENIZER_PATH:-/mnt/afs/250010210/xinwu/assets/safe-gpt-tokenizer}"
CHECKPOINT_PATH="${REPO_DIR}/checkpoints/genmol_v2_v1.0/model_v2.ckpt"
MASTER_PORT="${MASTER_PORT:-29500}"

PYTHON="${ENV_PREFIX}/bin/python"
ACCELERATE="${ENV_PREFIX}/bin/accelerate"

test -x "${PYTHON}"
test -x "${ACCELERATE}"
test -f "${CONFIG_PATH}"
test -d "${TOKENIZER_PATH}"
test -f "${CHECKPOINT_PATH}"

if [[ ! "${MASTER_PORT}" =~ ^[0-9]+$ ]] || ((MASTER_PORT < 1024 || MASTER_PORT > 65535)); then
  echo "MASTER_PORT must be an integer in [1024, 65535], got: ${MASTER_PORT}" >&2
  exit 1
fi

GPU_COUNT="$("${PYTHON}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${GPU_COUNT}" != "4" ]]; then
  echo "This launcher requires exactly 4 visible GPUs, found: ${GPU_COUNT}" >&2
  exit 1
fi

export GENMOL_SAFE_TOKENIZER_PATH="${TOKENIZER_PATH}"
export GENMOL_REWARD_WORKERS="${GENMOL_REWARD_WORKERS:-8}"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-SGRPO}"
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="${REPO_DIR}"

CACHE_TAG="sgrpo_rawloo_4gpu_${UID}_$$"
export HF_HOME="${HF_HOME:-/tmp/${CACHE_TAG}/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${CACHE_TAG}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/${CACHE_TAG}/torchinductor}"
mkdir -p "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

cd "${REPO_DIR}"

COMMIT="$(git rev-parse --short HEAD)"
echo "Launching 4-GPU training from commit ${COMMIT}"
echo "Config: ${CONFIG_PATH}"
echo "Visible GPUs: ${GPU_COUNT}"
echo "W&B mode: ${WANDB_MODE}"

exec "${ACCELERATE}" launch \
  --config_file configs/accelerate_zero2.yaml \
  --num_processes 4 \
  --main_process_port "${MASTER_PORT}" \
  scripts/train_cpgrpo_denovo.py \
  --config "${CONFIG_PATH}"
