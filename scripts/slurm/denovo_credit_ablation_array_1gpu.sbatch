#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -J dncredit
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/denovo_credit_ablation_sweep/logs/eval_%A_%a.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/denovo_credit_ablation_sweep/logs/eval_%A_%a.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
RUN_ROOT=/public/home/xinwuye/ai4s-tool-joint-train/runs/denovo_credit_ablation_sweep
TASKS_PATH=${RUN_ROOT}/specs/denovo/tasks.tsv

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is required" >&2
  exit 1
fi

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:${PATH}
export HF_HOME=/tmp/xinwuye_denovo_credit_${SLURM_JOB_ID}
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export TRITON_CACHE_DIR=${HF_HOME}/triton
export PYTHONPATH=${REPO_DIR}/src
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${TRITON_CACHE_DIR}"

cd "${REPO_DIR}"
task_line=$(awk -F '\t' -v task_id="${SLURM_ARRAY_TASK_ID}" \
  'NR > 1 && $1 == task_id {print; found=1} END {exit found ? 0 : 1}' \
  "${TASKS_PATH}") || {
  echo "No task ${SLURM_ARRAY_TASK_ID} in ${TASKS_PATH}" >&2
  exit 1
}
IFS=$'\t' read -r task_id seed category experiment config_path <<< "${task_line}"
if [[ "${task_id}" != "${SLURM_ARRAY_TASK_ID}" || ! -s "${config_path}" ]]; then
  echo "Invalid task row: ${task_line}" >&2
  exit 1
fi

echo "task_id=${task_id} seed=${seed} category=${category} experiment=${experiment}"
echo "config_path=${config_path}"
python -u scripts/eval_denovo_sgrpo.py --config "${config_path}"
