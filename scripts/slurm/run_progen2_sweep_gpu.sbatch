#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -t 24:00:00
#SBATCH -J p2sweepgpu
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_progen2sweepgpu_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/genmol/slurm_progen2sweepgpu_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
CONFIG_PATH=${CONFIG_PATH:-configs/progen2_temperature_sweep_pipeline_20260430.yaml}
MODE=${MODE:-}
TASK_ID=${TASK_ID:-${SLURM_ARRAY_TASK_ID:-}}
REWARD_NAME=${REWARD_NAME:-}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-}
PYTHON_OVERLAY_DIR=${PYTHON_OVERLAY_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/python_overlay}
PROTBERT_DIR=${PROTBERT_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/prot_bert_bfd}
CUDA_TOOLCHAIN_DIR=${CUDA_TOOLCHAIN_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/progen2_models/cuda_toolchain_cu124}
MIHOMO_DIR=/public/home/xinwuye/mihomo
CONDA_ENV_NAME=${CONDA_ENV_NAME:-genmol}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-}

if [[ -n "${CONDA_ENV_NAME}" && -n "${CONDA_ENV_PREFIX}" ]]; then
  echo "Set at most one of CONDA_ENV_NAME and CONDA_ENV_PREFIX" >&2
  exit 1
fi

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
if [[ -n "${CONDA_ENV_PREFIX}" ]]; then
  if [[ ! -d "${CONDA_ENV_PREFIX}" ]]; then
    echo "Conda env prefix not found: ${CONDA_ENV_PREFIX}" >&2
    exit 1
  fi
  conda activate "${CONDA_ENV_PREFIX}"
else
  conda activate "${CONDA_ENV_NAME}"
fi
export PATH=/opt/gridview/slurm/bin:$PATH
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

TORCH_LIB_DIR=$(python - <<'PY'
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent / "lib")
PY
)
if [[ ! -d "${TORCH_LIB_DIR}" ]]; then
  echo "Missing torch library directory: ${TORCH_LIB_DIR}" >&2
  exit 1
fi
CUDA_LIB_DIR=${CUDA_TOOLCHAIN_DIR}/lib
if [[ ! -d "${CUDA_LIB_DIR}" ]]; then
  echo "Missing CUDA toolchain library directory: ${CUDA_LIB_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CUDA_LIB_DIR}/libcudart.so.12" ]]; then
  echo "Missing CUDA runtime library: ${CUDA_LIB_DIR}/libcudart.so.12" >&2
  exit 1
fi
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"

if ! pgrep -u "$USER" -f "${MIHOMO_DIR}/mihomo" >/dev/null 2>&1; then
  nohup "${MIHOMO_DIR}/mihomo" -d "${MIHOMO_DIR}" >/tmp/mihomo_${SLURM_JOB_ID}.log 2>&1 &
  sleep 5
fi

export http_proxy=http://127.0.0.1:7897
export https_proxy=http://127.0.0.1:7897
export HTTP_PROXY=${http_proxy}
export HTTPS_PROXY=${https_proxy}

if [[ ! -d "${PYTHON_OVERLAY_DIR}" ]]; then
  echo "Missing ProGen2 python overlay: ${PYTHON_OVERLAY_DIR}" >&2
  exit 1
fi
if [[ ! -d "${PROTBERT_DIR}" ]]; then
  echo "Missing preprovisioned ProtBERT directory: ${PROTBERT_DIR}" >&2
  exit 1
fi
for required_file in config.json pytorch_model.bin vocab.txt tokenizer_config.json special_tokens_map.json; do
  if [[ ! -f "${PROTBERT_DIR}/${required_file}" ]]; then
    echo "Incomplete ProtBERT directory: missing ${PROTBERT_DIR}/${required_file}" >&2
    exit 1
  fi
done
if [[ ! -d "${PYTHON_OVERLAY_DIR}/openfold" ]]; then
  echo "Missing preprovisioned openfold package under overlay: ${PYTHON_OVERLAY_DIR}/openfold" >&2
  exit 1
fi
if ! compgen -G "${PYTHON_OVERLAY_DIR}/attn_core_inplace_cuda*.so" >/dev/null; then
  echo "Missing compiled OpenFold attention extension under overlay: ${PYTHON_OVERLAY_DIR}/attn_core_inplace_cuda*.so" >&2
  exit 1
fi

export PYTHONPATH=${PYTHON_OVERLAY_DIR}:${REPO_DIR}/src

cd "${REPO_DIR}"

cmd=(python scripts/progen2_sweep_pipeline.py "${MODE}" --config "${CONFIG_PATH}")

if [[ "${MODE}" == "generate-task" ]]; then
  if [[ -z "${TASK_ID}" ]]; then
    echo "TASK_ID is required for MODE=generate-task" >&2
    exit 1
  fi
  cmd+=(--task-id "${TASK_ID}")
elif [[ "${MODE}" == "score-packed-gpu-reward" ]]; then
  if [[ -z "${REWARD_NAME}" ]]; then
    echo "REWARD_NAME is required for MODE=score-packed-gpu-reward" >&2
    exit 1
  fi
  cmd+=(--reward-name "${REWARD_NAME}")
  if [[ -n "${EXPERIMENT_NAME}" ]]; then
    cmd+=(--experiment-name "${EXPERIMENT_NAME}")
  fi
elif [[ "${MODE}" == "score-point-reward-task" ]]; then
  if [[ -z "${REWARD_NAME}" || -z "${TASK_ID}" ]]; then
    echo "REWARD_NAME and TASK_ID are required for MODE=score-point-reward-task" >&2
    exit 1
  fi
  cmd+=(--reward-name "${REWARD_NAME}" --task-id "${TASK_ID}")
else
  echo "Unsupported MODE=${MODE}" >&2
  exit 1
fi

"${cmd[@]}"
