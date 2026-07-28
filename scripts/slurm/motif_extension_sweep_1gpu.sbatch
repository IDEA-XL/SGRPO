#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 00:30:00
#SBATCH --exclude=server13,server46,server59
#SBATCH -J motif_eval
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/motif_extension/logs/motif_eval_%A_%a.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/motif_extension/logs/motif_eval_%A_%a.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
MANIFEST_PATH=${MANIFEST_PATH:?MANIFEST_PATH is required}
TASK_INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}

source ~/miniconda/etc/profile.d/conda.sh
conda activate genmol
cd "${REPO_DIR}"
mkdir -p /public/home/xinwuye/ai4s-tool-joint-train/runs/motif_extension/logs
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SPEC_PATH=$(python - "${MANIFEST_PATH}" "${TASK_INDEX}" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
task_index = int(sys.argv[2])
tasks = {
    int(task["task_index"]): task
    for task in manifest["tasks"]
}
if task_index not in tasks:
    raise KeyError(f"task index {task_index} not present in manifest")
print(tasks[task_index]["spec_path"])
PY
)
python scripts/eval_motif_extension_sweep.py --spec "${SPEC_PATH}"
