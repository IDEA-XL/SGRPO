#!/bin/bash

set -euo pipefail

CONDA_ENV_NAME=${CONDA_ENV_NAME:-}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-}
INSTALL_MM_PROGEN2_DEPS=${INSTALL_MM_PROGEN2_DEPS:-1}

if [[ -n "${CONDA_ENV_NAME}" && -n "${CONDA_ENV_PREFIX}" ]]; then
  echo "Set at most one of CONDA_ENV_NAME and CONDA_ENV_PREFIX" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found on PATH" >&2
  exit 1
fi

CONDA_BASE=$(conda info --base)
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "Missing conda activation script: ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
  exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ -z "${CONDA_ENV_NAME}" && -z "${CONDA_ENV_PREFIX}" ]]; then
  CONDA_ENV_NAME=genmol
fi

if [[ -n "${CONDA_ENV_PREFIX}" ]]; then
  conda create -y -p "${CONDA_ENV_PREFIX}" python==3.10 pip==23.3.1
  conda activate "${CONDA_ENV_PREFIX}"
else
  conda create -y -n "${CONDA_ENV_NAME}" python==3.10 pip==23.3.1
  conda activate "${CONDA_ENV_NAME}"
fi

# wandb==0.13.5 still imports pkg_resources via setuptools.
python -m pip install "setuptools<81"
pip install -r env/requirements.txt
pip install -e .
pip install scikit-learn==1.2.2
pip install \
  accelerate==1.13.0 \
  deepspeed==0.16.7 \
  ninja==1.13.0 \
  pytorch-lightning==2.6.1 \
  torch-geometric==2.7.0
pip install \
  torch-scatter==2.1.2+pt26cu124 \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

if [[ "${INSTALL_MM_PROGEN2_DEPS}" == "1" ]]; then
  pip install \
    fair-esm==2.0.0 \
    lmdb==2.2.0 \
    biotite==1.2.0 \
    scipy==1.15.3 \
    requests==2.33.1
fi
