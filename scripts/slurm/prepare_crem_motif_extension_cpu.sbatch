#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
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
MIHOMO_DIR=/public/home/xinwuye/mihomo
MIHOMO_PROXY=http://127.0.0.1:7897
MIHOMO_PID=

cleanup() {
  if [[ -n "${MIHOMO_PID}" ]] && kill -0 "${MIHOMO_PID}" 2>/dev/null; then
    kill "${MIHOMO_PID}"
    wait "${MIHOMO_PID}" || true
  fi
}
trap cleanup EXIT

source ~/miniconda/etc/profile.d/conda.sh
conda activate genmol
cd "${REPO_DIR}"
mkdir -p "${DATA_DIR}" /public/home/xinwuye/ai4s-tool-joint-train/runs/motif_extension/logs

if [[ ! -s "${ARCHIVE}" ]]; then
  test -x "${MIHOMO_DIR}/mihomo"
  test -s "${MIHOMO_DIR}/config.yaml"
  "${MIHOMO_DIR}/mihomo" \
    -d "${MIHOMO_DIR}" \
    -f "${MIHOMO_DIR}/config.yaml" \
    > "${DATA_DIR}/mihomo_${SLURM_JOB_ID}.log" 2>&1 &
  MIHOMO_PID=$!
  proxy_ready=0
  for _ in $(seq 1 12); do
    if curl --silent --show-error --fail --head \
      --max-time 30 --proxy "${MIHOMO_PROXY}" "${URL}" >/dev/null; then
      proxy_ready=1
      break
    fi
    sleep 5
  done
  if [[ "${proxy_ready}" != 1 ]]; then
    echo "Mihomo proxy did not become ready" >&2
    exit 1
  fi
  curl --fail --location --continue-at - \
    --retry 20 --retry-all-errors --retry-delay 5 \
    --connect-timeout 30 --speed-limit 1024 --speed-time 180 \
    --proxy "${MIHOMO_PROXY}" \
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
