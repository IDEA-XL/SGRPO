#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J rbmmagg
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/rebuttal_dense_sweep/logs/mmagg_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/rebuttal_dense_sweep/logs/mmagg_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
: "${TASKS_PATH:?TASKS_PATH is required}"
: "${DOCKING_ROOT:?DOCKING_ROOT is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
EXPECTED_NUM_TASKS=${EXPECTED_NUM_TASKS:-30}

source /public/home/xinwuye/miniconda/etc/profile.d/conda.sh
conda activate genmol
export PATH=/opt/gridview/slurm/bin:${PATH}
export MPLBACKEND=Agg

cd "${REPO_DIR}"
python scripts/aggregate_mmgenmol_sweep_results.py \
  --tasks_path "${TASKS_PATH}" \
  --docking_root "${DOCKING_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --output_prefix mmgenmol_dense \
  --expected_num_tasks "${EXPECTED_NUM_TASKS}" \
  --expected_rows_per_task 1600 \
  --expected_num_pockets 100 \
  --expected_samples_per_pocket 16 \
  --docking_mode vina_dock \
  --plot_name_prefix mmgenmol_dense \
  --plot_title_prefix "Pocket-Based Design Dense Sweep" \
  --qed_weight 0.3 \
  --sa_score_weight 0.2 \
  --drugclip_score_weight 0.0
