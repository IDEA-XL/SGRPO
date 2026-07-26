#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -J mm100merge

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
: "${TASKS_PATH:?TASKS_PATH is required}"
: "${DOCKING_ROOT:?DOCKING_ROOT is required}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID is required" >&2
  exit 1
fi

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:${PATH}

cd "${REPO_DIR}"
task_line=$(awk -F '\t' -v task_id="${SLURM_ARRAY_TASK_ID}" \
  'NR > 1 && $1 == task_id {print; found=1} END {exit found ? 0 : 1}' "${TASKS_PATH}") || {
  echo "No task ${SLURM_ARRAY_TASK_ID} in ${TASKS_PATH}" >&2
  exit 1
}
IFS=$'\t' read -r task_id model_name sweep_type sweep_value randomness temperature checkpoint_path generated_rows_path <<< "${task_line}"
output_dir="${DOCKING_ROOT}/${model_name}/${sweep_type}_${sweep_value}"

python scripts/merge_pocket_prefix_docking_shards.py \
  --shard_root "${output_dir}/shards" \
  --num_shards 10 \
  --expected_num_rows 10000 \
  --output_rows_path "${output_dir}/docking.records.jsonl" \
  --output_summary_path "${output_dir}/docking.summary.json" \
  --docking_modes vina_dock

record_count=$(wc -l < "${output_dir}/docking.records.jsonl")
if [[ "${record_count}" != "10000" ]]; then
  echo "Merged docking record count mismatch: ${record_count} vs 10000" >&2
  exit 1
fi
