#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -J mm100dock

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
: "${TASKS_PATH:?TASKS_PATH is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
MANIFEST_PATH=${MANIFEST_PATH:-/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix/data/gdrive_exact_only/crossdocked_pocket10_pocket_prefix_manifest.pt}
CROSSDOCKED_ROOT=${CROSSDOCKED_ROOT:-/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/test_set/test_set}
QVINA_PATH=${QVINA_PATH:-/public/home/xinwuye/ai4s-tool-joint-train/genmol/scripts/exps/lead/docking/qvina02}
DOCKING_CACHE_DIR=${DOCKING_CACHE_DIR:-/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/pocket_prefix_crossdocked_5500ckpt/docking_cache}
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/conda_env}
NUM_SHARDS=10

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is required" >&2
  exit 1
fi
task_id=$((SLURM_ARRAY_TASK_ID / NUM_SHARDS))
shard_index=$((SLURM_ARRAY_TASK_ID % NUM_SHARDS))

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
test -d "${CONDA_ENV_PREFIX}"
conda activate "${CONDA_ENV_PREFIX}"
export PATH=/opt/gridview/slurm/bin:${PATH}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${REPO_DIR}"
test -s "${TASKS_PATH}"
test -s "${MANIFEST_PATH}"
test -d "${CROSSDOCKED_ROOT}"
task_line=$(awk -F '\t' -v task_id="${task_id}" \
  'NR > 1 && $1 == task_id {print; found=1} END {exit found ? 0 : 1}' "${TASKS_PATH}") || {
  echo "No task ${task_id} in ${TASKS_PATH}" >&2
  exit 1
}
IFS=$'\t' read -r row_task_id model_name sweep_type sweep_value randomness temperature checkpoint_path generated_rows_path <<< "${task_line}"
if [[ "${row_task_id}" != "${task_id}" ]]; then
  echo "Task ID mismatch" >&2
  exit 1
fi
test -s "${generated_rows_path}"
row_count=$(wc -l < "${generated_rows_path}")
if [[ "${row_count}" != "10000" ]]; then
  echo "Generated row count mismatch: ${row_count} vs 10000" >&2
  exit 1
fi

output_dir="${OUTPUT_ROOT}/${model_name}/${sweep_type}_${sweep_value}/shards/shard_$(printf '%02d' "${shard_index}")"
records_path="${output_dir}/docking.records.jsonl"
summary_path="${output_dir}/docking.summary.json"
if [[ -e "${records_path}" || -e "${summary_path}" ]]; then
  echo "Docking shard output already exists under ${output_dir}" >&2
  exit 1
fi
mkdir -p "${output_dir}"

echo "task_id=${task_id} shard_index=${shard_index} model=${model_name} generated_rows_path=${generated_rows_path}"
/usr/bin/time -p python scripts/dock_pocket_prefix_generated_rows.py \
  --generated_rows_path "${generated_rows_path}" \
  --manifest_path "${MANIFEST_PATH}" \
  --split test \
  --crossdocked_root "${CROSSDOCKED_ROOT}" \
  --qvina_path "${QVINA_PATH}" \
  --docking_cache_dir "${DOCKING_CACHE_DIR}" \
  --output_rows_path "${records_path}" \
  --output_summary_path "${summary_path}" \
  --num_workers "${SLURM_CPUS_PER_TASK}" \
  --num_shards "${NUM_SHARDS}" \
  --shard_index "${shard_index}" \
  --docking_modes vina_dock \
  --docking_exhaustiveness 8 \
  --docking_num_cpu 1 \
  --docking_num_modes 10 \
  --docking_timeout_gen3d 30 \
  --docking_timeout_dock 100 \
  --docking_box_size 20.0 20.0 20.0 \
  --progress_every 100

record_count=$(wc -l < "${records_path}")
if [[ "${record_count}" != "1000" ]]; then
  echo "Docking shard record count mismatch: ${record_count} vs 1000" >&2
  exit 1
fi
