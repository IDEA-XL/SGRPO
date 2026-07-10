# SGRPO

This repository contains the code for **Supergroup Relative Policy Optimization (SGRPO)** across three biomolecular post-training settings:

- de novo small-molecule design
- pocket-based small-molecule design
- de novo protein design

For pocket-based design, evaluation is reported with **AutoDock Vina** docking metrics such as `vina_dock_mean`. The training-time scalar utility still uses `unidock_score` as the internal field name for the corresponding Vina-derived high-is-better proxy.

## Recommended Repo-Relative Layout

All commands below assume:

- you clone the repository as `SGRPO`
- you run commands from the repository root
- you place assets at the repo-relative paths shown below

```text
SGRPO/
  checkpoints/
    genmol_v2_v1.0/
      model_v2.ckpt
    genmol_p_v1.0/
      5500.ckpt
  configs/
    public/
      protein_denovo_eval.yaml
      protein_denovo_train.yaml
      small_molecule_denovo_eval.yaml
      small_molecule_denovo_train.yaml
      small_molecule_pocket_tasks.tsv
      small_molecule_pocket_train.yaml
  data/
    crossdocked/
      processed/
        crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb
        crossdocked_pocket10_pose_split.pt
        crossdocked_pocket10_pocket_prefix_manifest.pt
      raw/
        test_set/
          ...
  runs/
    cpgrpo_denovo/
      sgrpo_denovo/
        checkpoint-002000/
          model.ckpt
    cpgrpo_denovo_pocket_prefix/
      sgrpo_pocket/
        checkpoint-001000/
          model.ckpt
    progen2_official/
    progen2_models/
      openfold_official/
      python_overlay/
      prot_bert_bfd/
      proteinsol_official/
      temberture_official/
    progen2_sgrpo/
      sgrpo_protein/
        checkpoint-000100/
          config.json
          generation_config.json
          model.safetensors
  scripts/
    exps/
      lead/
        docking/
          qvina02
  sgrpo-main-results/
    ...
```

`sgrpo-main-results/` is git-ignored. It is the default workspace for sweep outputs, aggregated JSON files, and plotting inputs.

## Environment

The final experiments used Linux x86-64, Python 3.10, PyTorch 2.6.0 with the CUDA 12.4 runtime, Transformers 4.52.4, Accelerate 1.13.0, and DeepSpeed 0.16.7. The supported recreation path is `env/setup.sh`; it installs the molecular, pocket-based, and protein dependencies used by the training and sweep pipelines.

The host must provide an NVIDIA driver compatible with CUDA 12.4. A local CUDA toolkit is not required for the small-molecule workflows, but it is required later when compiling the OpenFold extension for the protein workflow.

```bash
git clone <YOUR_GITHUB_URL>/SGRPO.git
cd SGRPO

source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_ENV_NAME=sgrpo
export INSTALL_MM_PROGEN2_DEPS=1
bash env/setup.sh
conda activate "${CONDA_ENV_NAME}"
```

If you prefer an explicit prefix instead of a named conda environment:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_ENV_PREFIX="$(pwd)/.conda/sgrpo"
bash env/setup.sh
conda activate "${CONDA_ENV_PREFIX}"
```

`env/setup.sh` performs the following steps:

- creates a fresh Python 3.10 environment with `pip==23.3.1`
- installs the base repo requirements from `env/requirements.txt`
- installs the repo in editable mode
- installs the Accelerate, DeepSpeed, PyTorch Geometric, and CUDA-12.4-compatible `torch-scatter` versions used by the final runs
- installs the extra packages needed by the pocket-based and ProGen2 pipelines
- pins `setuptools<81` so that `wandb==0.13.5` remains importable

Verify the recreated base environment before downloading model and dataset assets:

```bash
python - <<'PY'
import accelerate
import deepspeed
import torch
import transformers

expected = {
    "torch": "2.6.0",
    "transformers": "4.52.4",
    "accelerate": "1.13.0",
    "deepspeed": "0.16.7",
}
actual = {
    "torch": torch.__version__.split("+")[0],
    "transformers": transformers.__version__,
    "accelerate": accelerate.__version__,
    "deepspeed": deepspeed.__version__,
}
if actual != expected:
    raise RuntimeError(f"Environment version mismatch: expected={expected}, actual={actual}")
print(actual)
PY
```

Uni-Dock is intentionally installed into its own Conda environment, and the protein workflow uses a repo-local Python overlay plus a repo-local CUDA 12.4 compiler toolchain. Those two additions are described in their respective asset sections below; they are not hidden dependencies of the base environment.

## One-Time Assets

### 1. De novo small-molecule checkpoint

The pretrained de novo initialization checkpoint is provided through the official NGC release:

```bash
ngc registry resource download-version "nvidia/clara/genmol_v2:1.0"
```

After the NGC download completes, copy `model_v2.ckpt` into the repo-relative path expected by the public config:

```bash
mkdir -p checkpoints/genmol_v2_v1.0
cp /path/to/downloaded/model_v2.ckpt checkpoints/genmol_v2_v1.0/model_v2.ckpt
```

This creates:

```text
checkpoints/genmol_v2_v1.0/model_v2.ckpt
```

### 2. Pocket-based small-molecule assets

The pocket-based setup needs:

- the initialization checkpoint
- a processed CrossDocked LMDB
- the corresponding split file
- a pocket-prefix manifest
- the raw CrossDocked test-set structure tree for Vina evaluation
- a working `unidock` executable on `PATH`
- a Vina-compatible executable at `scripts/exps/lead/docking/qvina02`

The pocket-based initialization checkpoint is mirrored in the anonymous model browser:

```text
https://anonymous-hf.up.railway.app/a/5vre4umkd3wk/
```

The anonymous mirror also exposes a direct file-download endpoint, so you can fetch the checkpoint from the command line:

```bash
mkdir -p checkpoints/genmol_p_v1.0
curl -L \
  'https://anonymous-hf.up.railway.app/api/a/5vre4umkd3wk/resolve/checkpoints/genmol_p_v1.0/5500.ckpt' \
  -o checkpoints/genmol_p_v1.0/5500.ckpt
```

This creates:

```text
checkpoints/genmol_p_v1.0/5500.ckpt
```

Normal file downloads from this anonymous mirror have been verified. The mirror does not advertise range support, so resumable downloads are not assumed.

The remaining pocket-based assets are fetched from their official sources and are not mirrored by this repository.

#### CrossDocked processed files and docking test set

Official sources:

- processed LMDB / split / `test_set.zip`: TargetDiff data folder `https://drive.google.com/drive/folders/1j21cc7-97TedKh_El5E34yI8o5ckI7eK?usp=share_link`
- original CrossDocked release: `https://bits.csb.pitt.edu/files/crossdock2020/`

The following commands download the official TargetDiff data folder to a temporary directory, copy the required files into the repo-relative paths expected by the public configs, and unpack the Vina docking test set:

```bash
python -m pip install gdown

mkdir -p data/crossdocked/processed data/crossdocked/raw

gdown --folder "https://drive.google.com/drive/folders/1j21cc7-97TedKh_El5E34yI8o5ckI7eK?usp=share_link" \
  -O .cache/targetdiff_crossdocked

cp "$(find .cache/targetdiff_crossdocked -name 'crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb' -print -quit)" \
  data/crossdocked/processed/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb

cp "$(find .cache/targetdiff_crossdocked -name 'crossdocked_pocket10_pose_split.pt' -print -quit)" \
  data/crossdocked/processed/crossdocked_pocket10_pose_split.pt

cp "$(find .cache/targetdiff_crossdocked -name 'test_set.zip' -print -quit)" \
  data/crossdocked/raw/test_set.zip

unzip -q data/crossdocked/raw/test_set.zip -d data/crossdocked/raw
```

This should leave the public-config-required files at:

```text
data/crossdocked/processed/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb
data/crossdocked/processed/crossdocked_pocket10_pose_split.pt
data/crossdocked/raw/test_set/
```

Then build the pocket-prefix manifest:

```bash
python scripts/build_crossdocked_pocket_prefix_manifest.py \
  --crossdocked_lmdb_path data/crossdocked/processed/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb \
  --crossdocked_split_path data/crossdocked/processed/crossdocked_pocket10_pose_split.pt \
  --output_path data/crossdocked/processed/crossdocked_pocket10_pocket_prefix_manifest.pt
```

This creates:

```text
data/crossdocked/processed/crossdocked_pocket10_pocket_prefix_manifest.pt
```

If you want to regenerate the processed files from the original CrossDocked release instead of using the official TargetDiff preprocessed artifacts, the official v1.1 archive is:

```bash
mkdir -p data/CrossDocked2020_v1.1

curl -L https://bits.csb.pitt.edu/files/crossdock2020/v1.1/CrossDocked2020_v1.1.tgz \
  -o data/CrossDocked2020_v1.1/CrossDocked2020_v1.1.tgz

tar -xzf data/CrossDocked2020_v1.1/CrossDocked2020_v1.1.tgz \
  -C data/CrossDocked2020_v1.1
```

The CrossDocked2020 data files published at `bits.csb.pitt.edu/files/crossdock2020/` are released under `CC0 1.0`.

#### Uni-Dock

Official source:

- repository: `https://github.com/dptech-corp/Uni-Dock`

Official installation path:

```bash
conda create -y -n unidock_env unidock -c conda-forge
conda run -n unidock_env unidock --help
export PATH="$(conda info --base)/envs/unidock_env/bin:${PATH}"
```

The public training config expects a `unidock` executable on `PATH` when you launch the pocket-based workflow.

#### QuickVina 2

Official sources:

- repository: `https://github.com/QVina/qvina`
- compile instructions: `https://qvina.github.io/compilingQvina2.html`

The public evaluation command expects a `qvina02` executable at `scripts/exps/lead/docking/qvina02`.

The quickest official route is to download the upstream binary and rename it in place:

```bash
mkdir -p scripts/exps/lead/docking

curl -L https://raw.githubusercontent.com/QVina/qvina/master/bin/qvina2.1 \
  -o scripts/exps/lead/docking/qvina02

chmod +x scripts/exps/lead/docking/qvina02
```

### 3. De novo protein assets

The ProGen2 setup needs:

- the official ProGen2 tokenizer and checkpoint tree
- a Python overlay containing pinned `transformers` and `adapters`
- a compiled OpenFold extension installed into that overlay
- ProtBERT
- TemBERTure
- Protein-Sol

These assets are fetched from their official sources and are not mirrored by this repository. The commands below download each asset into its expected repo-relative location with the expected on-disk name.

#### ProGen2 official release

Official source:

- repository: `https://github.com/enijkamp/progen2`
- checkpoints: `https://storage.googleapis.com/sfr-progen-research/checkpoints/`

```bash
python scripts/setup_progen2_official.py \
  --output-dir runs/progen2_official \
  --model progen2-small
```

This creates:

```text
runs/progen2_official/tokenizer.json
runs/progen2_official/prompts_unconditional.txt
runs/progen2_official/checkpoints/progen2-small/
```

#### OpenFold

Official source:

- repository: `https://github.com/aqlaboratory/openfold`

```bash
git clone https://github.com/aqlaboratory/openfold.git \
  runs/progen2_models/openfold_official

python scripts/setup_progen2_python_overlay.py \
  --overlay-dir runs/progen2_models/python_overlay

conda create -y \
  -p runs/progen2_models/cuda_toolchain_cu124 \
  -c nvidia/label/cuda-12.4.1 \
  cuda-nvcc cuda-cudart-dev cuda-libraries-dev

export CUDA_HOME="$(pwd)/runs/progen2_models/cuda_toolchain_cu124"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

test -x "${CUDA_HOME}/bin/nvcc"
test -f "${CUDA_HOME}/include/cusparse.h"

python scripts/setup_openfold_extension.py \
  --source-dir runs/progen2_models/openfold_official \
  --overlay-dir runs/progen2_models/python_overlay
```

The OpenFold extension is compiled against the active PyTorch environment and must therefore be rebuilt if the PyTorch or CUDA minor version changes.

#### TemBERTure

Official source:

- repository: `https://github.com/ibmm-unibe-ch/TemBERTure`

```bash
python scripts/setup_temberture_official.py \
  --output-dir runs/progen2_models/temberture_official
```

This populates:

```text
runs/progen2_models/temberture_official/
```

#### Protein-Sol

Official source:

- download endpoint: `https://protein-sol.manchester.ac.uk/cgi-bin/utilities/download_sequence_code.php`

```bash
python scripts/setup_proteinsol_official.py \
  --output-dir runs/progen2_models/proteinsol_official
```

This downloads the official zip bundle and extracts it to:

```text
runs/progen2_models/proteinsol_official/protein-sol-sequence-prediction-software/
```

#### ProtBERT-BFD

Official source:

- model card: `https://huggingface.co/Rostlab/prot_bert_bfd`

```bash
huggingface-cli download Rostlab/prot_bert_bfd \
  --local-dir runs/progen2_models/prot_bert_bfd
```

This downloads the official Hugging Face snapshot to:

```text
runs/progen2_models/prot_bert_bfd/
```

For ProGen2 training and evaluation, the OpenFold overlay must be importable:

```bash
export PYTHONPATH="$(pwd)/runs/progen2_models/python_overlay:$(pwd)/src:${PYTHONPATH:-}"
```

### 4. Canonical SGRPO evaluation checkpoints

The public evaluation configs use fixed repo-relative checkpoint paths. If you want to reproduce the canonical SGRPO evaluation results without retraining, download the three released SGRPO checkpoints from the same anonymous model browser:

```text
https://anonymous-hf.up.railway.app/a/5vre4umkd3wk/
```

The same anonymous mirror also provides direct file-download endpoints, so the checkpoints can be fetched from the command line into the exact repo-relative paths used by the public evaluation configs:

#### De novo small-molecule SGRPO checkpoint

```bash
mkdir -p runs/cpgrpo_denovo/sgrpo_denovo/checkpoint-002000
curl -L \
  'https://anonymous-hf.up.railway.app/api/a/5vre4umkd3wk/resolve/checkpoints/sgrpo_main/genmol_denovo_sgrpo_rewardsum_loo_2000/model.ckpt' \
  -o runs/cpgrpo_denovo/sgrpo_denovo/checkpoint-002000/model.ckpt
```

#### Pocket-based small-molecule SGRPO checkpoint

```bash
mkdir -p runs/cpgrpo_denovo_pocket_prefix/sgrpo_pocket/checkpoint-001000
curl -L \
  'https://anonymous-hf.up.railway.app/api/a/5vre4umkd3wk/resolve/checkpoints/sgrpo_main/mmgenmol_sgrpo_unidock_rewardsum_loo_1000/model.ckpt' \
  -o runs/cpgrpo_denovo_pocket_prefix/sgrpo_pocket/checkpoint-001000/model.ckpt
```

#### De novo protein SGRPO checkpoint

```bash
mkdir -p runs/progen2_sgrpo/sgrpo_protein/checkpoint-000100
base='https://anonymous-hf.up.railway.app/api/a/5vre4umkd3wk/resolve/checkpoints/sgrpo_main/progen2_sgrpo_gw08_rewardsum_loo_100'
curl -L "$base/config.json" \
  -o runs/progen2_sgrpo/sgrpo_protein/checkpoint-000100/config.json
curl -L "$base/generation_config.json" \
  -o runs/progen2_sgrpo/sgrpo_protein/checkpoint-000100/generation_config.json
curl -L "$base/model.safetensors" \
  -o runs/progen2_sgrpo/sgrpo_protein/checkpoint-000100/model.safetensors
curl -L "$base/trainer_state.pt" \
  -o runs/progen2_sgrpo/sgrpo_protein/checkpoint-000100/trainer_state.pt
```

## Public Configs

The files under `configs/public/` are ready to use without editing if you follow the repo-relative layout above:

- `configs/public/small_molecule_denovo_train.yaml`
- `configs/public/small_molecule_pocket_train.yaml`
- `configs/public/protein_denovo_train.yaml`
- `configs/public/small_molecule_denovo_eval.yaml`
- `configs/public/small_molecule_pocket_tasks.tsv`
- `configs/public/protein_denovo_eval.yaml`

These public configs use fixed repo-relative output directories so that the training and evaluation commands line up directly. If you need a different layout or want multiple concurrent runs of the same workflow, copy a file out of `configs/public/` and edit the copy.

## Train the Canonical SGRPO Models

### 1. De novo small-molecule design

```bash
accelerate launch \
  --config_file configs/accelerate_zero2.yaml \
  --num_processes 8 \
  --main_process_port 29501 \
  scripts/train_cpgrpo_denovo.py \
  --config configs/public/small_molecule_denovo_train.yaml
```

This writes checkpoints to:

```text
runs/cpgrpo_denovo/sgrpo_denovo/
```

### 2. Pocket-based small-molecule design

```bash
accelerate launch \
  --config_file configs/accelerate_zero2.yaml \
  --num_processes 8 \
  --main_process_port 29502 \
  scripts/train_cpgrpo_denovo_pocket_prefix.py \
  --config configs/public/small_molecule_pocket_train.yaml
```

This writes checkpoints to:

```text
runs/cpgrpo_denovo_pocket_prefix/sgrpo_pocket/
```

### 3. De novo protein design

```bash
export PYTHONPATH="$(pwd)/runs/progen2_models/python_overlay:$(pwd)/src:${PYTHONPATH:-}"

accelerate launch \
  --config_file configs/accelerate_ddp_8gpu.yaml \
  --num_processes 8 \
  --main_process_port 29503 \
  scripts/train_progen2_sgrpo.py \
  configs/public/protein_denovo_train.yaml
```

This writes checkpoints to:

```text
runs/progen2_sgrpo/sgrpo_protein/
```

The supported evaluation checkpoint for the protein workflow is `checkpoint-000100`.

If you rerun one of the public training configs after its output directory already exists, either remove the existing output directory first or copy the config and change `output_dir`. The training scripts are fail-fast and will not silently merge runs.

## Reproduce the Canonical SGRPO Sweeps

### 1. De novo small-molecule paired \((\rho,\tau)\) sweep

```bash
python scripts/eval_denovo_sgrpo.py \
  --config configs/public/small_molecule_denovo_eval.yaml
```

This evaluates the six supported operating points:

- `(randomness, temperature) = (0.1, 0.5)`
- `(0.3, 0.8)`
- `(0.5, 1.1)`
- `(0.7, 1.4)`
- `(0.9, 1.7)`
- `(1.0, 2.0)`

Outputs are written under:

```text
sgrpo-main-results/small_molecule_denovo/paired_sweep/
```

### 2. Pocket-based paired \((\rho,\tau)\) sweep with Vina docking

This workflow:

1. generates molecules at the six paired operating points
2. docks those generations with AutoDock Vina
3. aggregates the six operating points into one JSON summary

The public task manifest is:

```text
configs/public/small_molecule_pocket_tasks.tsv
```

Create the output workspace:

```bash
mkdir -p \
  sgrpo-main-results/small_molecule_pocket/generation \
  sgrpo-main-results/small_molecule_pocket/docking \
  sgrpo-main-results/small_molecule_pocket/docking_cache \
  sgrpo-main-results/small_molecule_pocket/aggregate
```

Run generation:

```bash
tail -n +2 configs/public/small_molecule_pocket_tasks.tsv | while IFS=$'\t' read -r task_id model_name sweep_type sweep_value randomness temperature checkpoint_path output_path; do
  python scripts/sample_denovo_pocket_prefix.py \
    --checkpoint_path "${checkpoint_path}" \
    --manifest_path data/crossdocked/processed/crossdocked_pocket10_pocket_prefix_manifest.pt \
    --split test \
    --num_pockets 100 \
    --num_samples_per_pocket 16 \
    --generation_batch_size 1600 \
    --generation_temperature "${temperature}" \
    --randomness "${randomness}" \
    --min_add_len 60 \
    --seed 42 \
    --device cuda \
    --bf16 \
    --output_path "${output_path}"
done
```

Run docking:

```bash
tail -n +2 configs/public/small_molecule_pocket_tasks.tsv | while IFS=$'\t' read -r task_id model_name sweep_type sweep_value randomness temperature checkpoint_path generated_rows_path; do
  output_dir="sgrpo-main-results/small_molecule_pocket/docking/${model_name}/${sweep_type}_${sweep_value}"
  mkdir -p "${output_dir}"
  python scripts/dock_pocket_prefix_generated_rows.py \
    --generated_rows_path "${generated_rows_path}" \
    --manifest_path data/crossdocked/processed/crossdocked_pocket10_pocket_prefix_manifest.pt \
    --split test \
    --crossdocked_root data/crossdocked/raw/test_set \
    --qvina_path scripts/exps/lead/docking/qvina02 \
    --docking_cache_dir sgrpo-main-results/small_molecule_pocket/docking_cache \
    --output_rows_path "${output_dir}/docking.records.jsonl" \
    --output_summary_path "${output_dir}/docking.summary.json" \
    --num_workers 64 \
    --docking_modes vina_dock \
    --docking_exhaustiveness 8 \
    --docking_num_cpu 1 \
    --docking_num_modes 10 \
    --docking_timeout_gen3d 30 \
    --docking_timeout_dock 100 \
    --docking_box_size 20.0 20.0 20.0 \
    --progress_every 100
done
```

Aggregate the six points:

```bash
python scripts/aggregate_mmgenmol_sweep_results.py \
  --tasks_path configs/public/small_molecule_pocket_tasks.tsv \
  --docking_root sgrpo-main-results/small_molecule_pocket/docking \
  --output_dir sgrpo-main-results/small_molecule_pocket/aggregate \
  --output_prefix paired_sweep \
  --expected_num_tasks 6 \
  --expected_rows_per_task 1600 \
  --expected_num_pockets 100 \
  --expected_samples_per_pocket 16 \
  --docking_mode vina_dock \
  --plot_name_prefix small_molecule_pocket \
  --plot_title_prefix "Pocket-Based Design" \
  --qed_weight 0.3 \
  --sa_score_weight 0.2 \
  --drugclip_score_weight 0.0
```

The key docking metric in the aggregate output is `vina_dock_mean`.

### 3. De novo protein temperature sweep

```bash
export PYTHONPATH="$(pwd)/runs/progen2_models/python_overlay:$(pwd)/src:${PYTHONPATH:-}"
export PROGEN2_TASKS_PATH=sgrpo-main-results/protein_denovo/temperature_sweep/tasks.tsv

python scripts/progen2_sweep_pipeline.py build-tasks \
  --config configs/public/protein_denovo_eval.yaml
```

Run generation for each task id listed in `PROGEN2_TASKS_PATH`:

```bash
for task_id in $(tail -n +2 "${PROGEN2_TASKS_PATH}" | cut -f1); do
  python scripts/progen2_sweep_pipeline.py generate-task \
    --config configs/public/protein_denovo_eval.yaml \
    --task-id "${task_id}"
done
```

Run the packed GPU rewards:

```bash
python scripts/progen2_sweep_pipeline.py score-packed-gpu-reward \
  --config configs/public/protein_denovo_eval.yaml \
  --reward-name naturalness

python scripts/progen2_sweep_pipeline.py score-packed-gpu-reward \
  --config configs/public/protein_denovo_eval.yaml \
  --reward-name stability
```

Run the point-wise rewards and diversity jobs:

```bash
for task_id in $(tail -n +2 "${PROGEN2_TASKS_PATH}" | cut -f1); do
  python scripts/progen2_sweep_pipeline.py score-point-reward-task \
    --config configs/public/protein_denovo_eval.yaml \
    --reward-name foldability \
    --task-id "${task_id}"

  python scripts/progen2_sweep_pipeline.py score-point-reward-task \
    --config configs/public/protein_denovo_eval.yaml \
    --reward-name developability \
    --task-id "${task_id}"

  python scripts/progen2_sweep_pipeline.py score-point-diversity-task \
    --config configs/public/protein_denovo_eval.yaml \
    --task-id "${task_id}"
done
```

Aggregate the final sweep:

```bash
python scripts/progen2_sweep_pipeline.py aggregate \
  --config configs/public/protein_denovo_eval.yaml
```

Outputs are written under:

```text
sgrpo-main-results/protein_denovo/temperature_sweep/
```

## Acknowledgements

This repository builds on ideas, code, and assets from the following upstream projects:

- [NVIDIA-Digital-Bio/genmol](https://github.com/NVIDIA-Digital-Bio/genmol)
- [facebookresearch/esm](https://github.com/facebookresearch/esm/tree/main)
- [enijkamp/progen2](https://github.com/enijkamp/progen2)
- [apple/ml-diffucoder](https://github.com/apple/ml-diffucoder)
