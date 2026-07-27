#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J motif_data
#SBATCH -o /public/home/xinwuye/ai4s-tool-joint-train/runs/motif_extension/logs/motif_data_%j.out
#SBATCH -e /public/home/xinwuye/ai4s-tool-joint-train/runs/motif_extension/logs/motif_data_%j.err

set -euo pipefail

REPO_DIR=/public/home/xinwuye/ai4s-tool-joint-train/genmol
DATA_DIR=/public/home/xinwuye/ai4s-tool-joint-train/runs/motif_extension/data
ARCHIVE="${DATA_DIR}/chembl22_sa2_hac12.db.gz"
DATABASE="${DATA_DIR}/chembl22_sa2_hac12.db"
OUTPUT="${DATA_DIR}/crem_chembl22_sa2_hac12_r3_f5_hac5_12.jsonl"
METADATA="${DATA_DIR}/crem_chembl22_sa2_hac12_r3_f5_hac5_12.metadata.json"
URL=https://zenodo.org/records/16909329/files/chembl22_sa2_hac12.db.gz?download=1

source ~/miniconda/etc/profile.d/conda.sh
conda activate genmol
cd "${REPO_DIR}"
mkdir -p "${DATA_DIR}" /public/home/xinwuye/ai4s-tool-joint-train/runs/motif_extension/logs

if [[ ! -s "${ARCHIVE}" ]]; then
  curl --fail --location --retry 5 --retry-delay 10 \
    --output "${ARCHIVE}.part" "${URL}"
  mv "${ARCHIVE}.part" "${ARCHIVE}"
fi
gzip -t "${ARCHIVE}"
if [[ ! -s "${DATABASE}" ]]; then
  gzip -dc "${ARCHIVE}" > "${DATABASE}.tmp"
  mv "${DATABASE}.tmp" "${DATABASE}"
fi
if [[ -e "${OUTPUT}" || -e "${METADATA}" ]]; then
  echo "Refusing to overwrite existing motif dataset outputs" >&2
  exit 1
fi

export PYTHONPATH=src
python scripts/build_crem_motif_extension_dataset.py \
  --database "${DATABASE}" \
  --test-fragments data/fragments.csv \
  --output "${OUTPUT}" \
  --metadata "${METADATA}" \
  --min-frequency 5 \
  --min-heavy-atoms 5 \
  --max-heavy-atoms 12
