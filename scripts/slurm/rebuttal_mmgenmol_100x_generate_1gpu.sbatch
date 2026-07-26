#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -J mm100gen

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
: "${TASKS_PATH:?TASKS_PATH is required}"
: "${SEED:?SEED is required}"
MANIFEST_PATH=${MANIFEST_PATH:-/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix/data/gdrive_exact_only/crossdocked_pocket10_pocket_prefix_manifest.pt}

if [[ -n "${TASK_IDS:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "Set TASK_IDS or SLURM_ARRAY_TASK_ID, not both" >&2
  exit 1
fi
if [[ -n "${TASK_IDS:-}" ]]; then
  IFS=':' read -r -a requested_task_ids <<< "${TASK_IDS}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  requested_task_ids=("${SLURM_ARRAY_TASK_ID}")
else
  echo "TASK_IDS or SLURM_ARRAY_TASK_ID is required" >&2
  exit 1
fi
if [[ "${#requested_task_ids[@]}" -le 0 ]]; then
  echo "No generation task IDs were provided" >&2
  exit 1
fi

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:${PATH}
export TOKENIZERS_PARALLELISM=false

cd "${REPO_DIR}"
test -s "${TASKS_PATH}"
test -s "${MANIFEST_PATH}"

run_generation() {
  local batch_size=$1
  echo "task_id=${task_id} seed=${SEED} model=${model_name} randomness=${randomness} temperature=${temperature} generation_batch_size=${batch_size}"
  python scripts/sample_denovo_pocket_prefix.py \
    --checkpoint_path "${checkpoint_path}" \
    --manifest_path "${MANIFEST_PATH}" \
    --split test \
    --num_pockets 100 \
    --num_samples_per_pocket 100 \
    --generation_batch_size "${batch_size}" \
    --generation_temperature "${temperature}" \
    --randomness "${randomness}" \
    --min_add_len 60 \
    --seed "${SEED}" \
    --device cuda \
    --bf16 \
    --output_path "${output_path}" 2>&1 | tee "${attempt_log}"
  return "${PIPESTATUS[0]}"
}

for requested_task_id in "${requested_task_ids[@]}"; do
  if [[ ! "${requested_task_id}" =~ ^[0-9]+$ ]]; then
    echo "Invalid generation task ID: ${requested_task_id}" >&2
    exit 1
  fi
  task_line=$(awk -F '\t' -v task_id="${requested_task_id}" \
    'NR > 1 && $1 == task_id {print; found=1} END {exit found ? 0 : 1}' "${TASKS_PATH}") || {
    echo "No task ${requested_task_id} in ${TASKS_PATH}" >&2
    exit 1
  }
  IFS=$'\t' read -r task_id model_name sweep_type sweep_value randomness temperature checkpoint_path output_path <<< "${task_line}"
  if [[ "${task_id}" != "${requested_task_id}" ]]; then
    echo "Task ID mismatch" >&2
    exit 1
  fi
  test -s "${checkpoint_path}"
  if [[ -e "${output_path}" ]]; then
    echo "Output already exists: ${output_path}" >&2
    exit 1
  fi
  mkdir -p "$(dirname "${output_path}")"
  attempt_log="${output_path}.generation-attempt.log"

  set +e
  run_generation 10000
  generation_status=$?
  set -e
  if [[ "${generation_status}" -ne 0 ]]; then
    if ! grep -Eqi 'CUDA out of memory|OutOfMemoryError|out of memory' "${attempt_log}"; then
      echo "Generation failed for a non-OOM reason; refusing fallback" >&2
      exit "${generation_status}"
    fi
    echo "Detected OOM at generation_batch_size=10000; retrying once with 5000" >&2
    rm -f "${output_path}"
    run_generation 5000
  fi

  row_count=$(wc -l < "${output_path}")
  if [[ "${row_count}" != "10000" ]]; then
    echo "Generated row count mismatch: ${row_count} vs 10000" >&2
    exit 1
  fi
done
