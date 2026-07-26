#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -J mm100agg

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
: "${TASKS_PATH:?TASKS_PATH is required}"
: "${DOCKING_ROOT:?DOCKING_ROOT is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:${PATH}
export MPLBACKEND=Agg

cd "${REPO_DIR}"
python scripts/aggregate_mmgenmol_sweep_results.py \
  --tasks_path "${TASKS_PATH}" \
  --docking_root "${DOCKING_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_prefix mmgenmol_100x \
  --expected_num_tasks 30 \
  --expected_rows_per_task 10000 \
  --expected_num_pockets 100 \
  --expected_samples_per_pocket 100 \
  --docking_mode vina_dock \
  --plot_name_prefix mmgenmol_100x \
  --plot_title_prefix "Pocket-Based Design 100 Samples per Pocket" \
  --qed_weight 0.3 \
  --sa_score_weight 0.2 \
  --drugclip_score_weight 0.0
