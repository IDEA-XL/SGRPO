#!/usr/bin/env python3
"""Build the five-run dense sweep specifications used for rebuttal experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPO_REMOTE_ROOT = Path("/public/home/xinwuye/ai4s-tool-joint-train/genmol")
RUN_OUTPUT_ROOT = Path("/public/home/xinwuye/ai4s-tool-joint-train/runs/rebuttal_dense_sweep")
SPEC_OUTPUT_ROOT = RUN_OUTPUT_ROOT / "specs"
SEEDS = (42, 43, 44, 45, 46)

MOLECULE_SWEEP = (
    (0.1, 0.5),
    (0.2, 0.65),
    (0.3, 0.8),
    (0.4, 0.95),
    (0.5, 1.1),
    (0.6, 1.25),
    (0.7, 1.4),
    (0.8, 1.55),
    (0.9, 1.7),
    (1.0, 2.0),
)
PROGEN2_TEMPERATURES = tuple(round(0.1 + 0.05 * index, 2) for index in range(23))

DE_NOVO_EXPERIMENTS = (
    {
        "category": "main",
        "name": "original_genmol_v2",
        "display_name": "Original GenMol v2",
        "checkpoint_path": REPO_REMOTE_ROOT / "checkpoints/genmol_v2_v1.0/model_v2.ckpt",
    },
    {
        "category": "main",
        "name": "genmol_denovo_grpo_2000",
        "display_name": "GenMol De Novo GRPO 2000",
        "checkpoint_path": RUN_OUTPUT_ROOT.parents[0]
        / "cpgrpo_denovo/cpgrpo_denovo_ng512_bs1024_lr5e-5_beta5e-3_ni1_ms2000_20260422_161812"
        / "checkpoint-002000/model.ckpt",
    },
    {
        "category": "main",
        "name": "genmol_denovo_sgrpo_rewardsum_loo_2000",
        "display_name": "GenMol De Novo SGRPO RewardSum LOO 2000",
        "checkpoint_path": RUN_OUTPUT_ROOT.parents[0]
        / "cpgrpo_denovo/cpgrpo_denovo_sgrpo_ng64_sg8_bs1024_lr5e-5_beta5e-3_gw09_rewardsum_loo_ms2000_20260426_115639"
        / "checkpoint-002000/model.ckpt",
    },
    {
        "category": "main",
        "name": "genmol_denovo_grpo_hbd_2000",
        "display_name": "GenMol De Novo GRPO HBD 2000",
        "checkpoint_path": RUN_OUTPUT_ROOT.parents[0]
        / "cpgrpo_denovo/cpgrpo_denovo_ng512_bs1024_lr5e-5_beta5e-3_ni1_ms2000_hbd_st09_sc04_20260503_141949"
        / "checkpoint-002000/model.ckpt",
    },
    {
        "category": "ablation",
        "name": "genmol_denovo_sgrpo_rewardsum_2000",
        "display_name": "GenMol De Novo SGRPO RewardSum 2000",
        "checkpoint_path": RUN_OUTPUT_ROOT.parents[0]
        / "cpgrpo_denovo/cpgrpo_denovo_sgrpo_ng64_sg8_bs1024_lr5e-5_beta5e-3_gw09_rewardsum_ms2000_20260424_115413"
        / "checkpoint-002000/model.ckpt",
    },
    {
        "category": "hyperparam",
        "name": "genmol_denovo_sgrpo_ng4_rewardsum_loo_2000",
        "display_name": "GenMol De Novo SGRPO RewardSum LOO NG4 2000",
        "checkpoint_path": RUN_OUTPUT_ROOT.parents[0]
        / "cpgrpo_denovo/cpgrpo_denovo_sgrpo_ng4_sg8_bs1024_lr5e-5_beta5e-3_gw09_rewardsum_loo_ms2000_20260501_153027"
        / "checkpoint-002000/model.ckpt",
    },
    {
        "category": "hyperparam",
        "name": "genmol_denovo_sgrpo_ng16_rewardsum_loo_2000",
        "display_name": "GenMol De Novo SGRPO RewardSum LOO NG16 2000",
        "checkpoint_path": RUN_OUTPUT_ROOT.parents[0]
        / "cpgrpo_denovo/cpgrpo_denovo_sgrpo_ng16_sg8_bs1024_lr5e-5_beta5e-3_gw09_rewardsum_loo_ms2000_20260501_153029"
        / "checkpoint-002000/model.ckpt",
    },
    {
        "category": "hyperparam",
        "name": "genmol_denovo_sgrpo_ng32_rewardsum_loo_2000",
        "display_name": "GenMol De Novo SGRPO RewardSum LOO NG32 2000",
        "checkpoint_path": RUN_OUTPUT_ROOT.parents[0]
        / "cpgrpo_denovo/cpgrpo_denovo_sgrpo_ng32_sg8_bs1024_lr5e-5_beta5e-3_gw09_rewardsum_loo_ms2000_20260501_153029"
        / "checkpoint-002000/model.ckpt",
    },
    {
        "category": "hyperparam",
        "name": "genmol_denovo_sgrpo_gw01_rewardsum_loo_2000",
        "display_name": "GenMol De Novo SGRPO RewardSum LOO GW0.1 2000",
        "checkpoint_path": RUN_OUTPUT_ROOT.parents[0]
        / "cpgrpo_denovo/cpgrpo_denovo_sgrpo_ng64_sg8_bs1024_lr5e-5_beta5e-3_gw01_rewardsum_loo_ms2000_20260502_020043"
        / "checkpoint-002000/model.ckpt",
    },
    {
        "category": "hyperparam",
        "name": "genmol_denovo_sgrpo_gw03_rewardsum_loo_2000",
        "display_name": "GenMol De Novo SGRPO RewardSum LOO GW0.3 2000",
        "checkpoint_path": RUN_OUTPUT_ROOT.parents[0]
        / "cpgrpo_denovo/cpgrpo_denovo_sgrpo_ng64_sg8_bs1024_lr5e-5_beta5e-3_gw03_rewardsum_loo_ms2000_20260502_020043"
        / "checkpoint-002000/model.ckpt",
    },
    {
        "category": "hyperparam",
        "name": "genmol_denovo_sgrpo_gw05_rewardsum_loo_2000",
        "display_name": "GenMol De Novo SGRPO RewardSum LOO GW0.5 2000",
        "checkpoint_path": RUN_OUTPUT_ROOT.parents[0]
        / "cpgrpo_denovo/cpgrpo_denovo_sgrpo_ng64_sg8_bs1024_lr5e-5_beta5e-3_gw05_rewardsum_loo_ms2000_20260427_005454"
        / "checkpoint-002000/model.ckpt",
    },
    {
        "category": "hyperparam",
        "name": "genmol_denovo_sgrpo_gw07_rewardsum_loo_2000",
        "display_name": "GenMol De Novo SGRPO RewardSum LOO GW0.7 2000",
        "checkpoint_path": RUN_OUTPUT_ROOT.parents[0]
        / "cpgrpo_denovo/cpgrpo_denovo_sgrpo_ng64_sg8_bs1024_lr5e-5_beta5e-3_gw07_rewardsum_loo_ms2000_20260502_020043"
        / "checkpoint-002000/model.ckpt",
    },
)

MMGENMOL_EXPERIMENTS = (
    {
        "name": "original_5500",
        "checkpoint_path": RUN_OUTPUT_ROOT.parents[0]
        / "pocket_prefix_supervised_8gpu/20260416_151741/checkpoints/5500.ckpt",
    },
    {
        "name": "grpo_unidock_1000",
        "checkpoint_path": RUN_OUTPUT_ROOT.parents[0]
        / "cpgrpo_denovo_pocket_prefix/cpgrpo_denovo_pocket_prefix_ng192_bs384_lr5e-5_beta5e-3_ni1_q03_sa02_unidock05_20260430_192150"
        / "checkpoint-001000/model.ckpt",
    },
    {
        "name": "sgrpo_unidock_rewardsum_loo_1000",
        "checkpoint_path": RUN_OUTPUT_ROOT.parents[0]
        / "cpgrpo_denovo_pocket_prefix/cpgrpo_denovo_pocket_prefix_sgrpo_ng24_sg8_bs384_lr5e-5_beta5e-3_gw09_q03_sa02_unidock05_rewardsum_loo_20260501_160306"
        / "checkpoint-001000/model.ckpt",
    },
)

PROGEN2_EXPERIMENTS = (
    {
        "name": "original",
        "display_name": "Original",
        "checkpoint_dir": RUN_OUTPUT_ROOT.parents[0] / "progen2_official/checkpoints/progen2-small",
    },
    {
        "name": "grpo_step100",
        "display_name": "GRPO 100",
        "checkpoint_dir": RUN_OUTPUT_ROOT.parents[0]
        / "progen2_sgrpo/progen2_grpo_ng96_bs2_len256_rbs16_slurm52245/checkpoint-000100",
    },
    {
        "name": "sgrpo_gw08_rewardsum_loo_step100",
        "display_name": "SGRPO gw0.8 RewardSum LOO 100",
        "checkpoint_dir": RUN_OUTPUT_ROOT.parents[0]
        / "progen2_sgrpo/progen2_sgrpo_ng12_sg8_bs2_len256_rbs16_gw08_rewardsum_loo_slurm53602/checkpoint-000100",
    },
    {
        "name": "grpo_hbd_step100",
        "display_name": "GRPO HBD 100",
        "checkpoint_dir": RUN_OUTPUT_ROOT.parents[0]
        / "progen2_sgrpo/progen2_grpo_ng96_bs2_len256_rbs16_hbd_slurm55873/checkpoint-000100",
    },
)

PROGEN2_REWARD_WEIGHTS = {
    "naturalness": 0.25,
    "foldability": 0.30,
    "stability": 0.20,
    "developability": 0.25,
}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _dump_yaml(mapping: dict, indent: int = 0) -> list[str]:
    lines: list[str] = []
    prefix = " " * indent
    for key, value in mapping.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.extend(_dump_yaml(value, indent + 2))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(f"{prefix}  -")
                    lines.extend(_dump_yaml(item, indent + 4))
                else:
                    lines.append(f"{prefix}  - {json.dumps(item)}")
        elif value is None:
            lines.append(f"{prefix}{key}: null")
        elif isinstance(value, bool):
            lines.append(f"{prefix}{key}: {'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            lines.append(f"{prefix}{key}: {value}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return lines


def _denovo_config(seed: int, experiment: dict, spec_root: Path) -> tuple[Path, str]:
    run_root = (
        RUN_OUTPUT_ROOT
        / "denovo"
        / f"seed{seed}"
        / experiment["category"]
        / experiment["name"]
    )
    config_path = (
        spec_root
        / "denovo"
        / f"seed{seed}"
        / experiment["category"]
        / f"{experiment['name']}.yaml"
    )
    payload = {
        "output_markdown_path": str(run_root / "aggregate/dense.md"),
        "output_json_path": str(run_root / "aggregate/dense.json"),
        "output_qed_diversity_plot_path": str(run_root / "aggregate/qed_vs_diversity.png"),
        "output_sa_score_diversity_plot_path": str(run_root / "aggregate/sa_score_vs_diversity.png"),
        "output_soft_reward_diversity_plot_path": str(run_root / "aggregate/soft_reward_vs_diversity.png"),
        "output_rows_path": str(run_root / "aggregate/dense.rows.jsonl"),
        "seed": seed,
        "bf16": True,
        "device": "cuda",
        "num_samples": 1000,
        "generation_batch_size": 2048,
        "randomness_temperature_pairs": [
            {"randomness": randomness, "generation_temperature": temperature}
            for randomness, temperature in MOLECULE_SWEEP
        ],
        "min_add_len": 60,
        "max_completion_length": None,
        "experiments": [
            {
                "name": experiment["name"],
                "display_name": experiment["display_name"],
                "checkpoint_path": str(experiment["checkpoint_path"]),
            }
        ],
    }
    return config_path, "\n".join(_dump_yaml(payload)) + "\n"


def _build_denovo_specs(spec_root: Path) -> None:
    manifest_path = spec_root / "denovo/tasks.tsv"
    rows = []
    task_id = 0
    for seed in SEEDS:
        for experiment in DE_NOVO_EXPERIMENTS:
            config_path, config_text = _denovo_config(seed, experiment, spec_root)
            _write_text(config_path, config_text)
            rows.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "category": experiment["category"],
                    "experiment": experiment["name"],
                    "config_path": config_path,
                }
            )
            task_id += 1
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _build_mmgenmol_specs(spec_root: Path) -> None:
    for seed in SEEDS:
        task_path = spec_root / f"mmgenmol/seed{seed}.tsv"
        rows = []
        task_id = 0
        for experiment in MMGENMOL_EXPERIMENTS:
            for point_index, (randomness, temperature) in enumerate(MOLECULE_SWEEP, start=1):
                rows.append(
                    {
                        "task_id": task_id,
                        "model_name": experiment["name"],
                        "sweep_type": "paired",
                        "sweep_value": point_index,
                        "randomness": randomness,
                        "temperature": temperature,
                        "checkpoint_path": experiment["checkpoint_path"],
                        "output_path": RUN_OUTPUT_ROOT
                        / "mmgenmol"
                        / f"seed{seed}"
                        / "generation"
                        / experiment["name"]
                        / f"paired_{point_index}"
                        / "generated.rows.jsonl",
                    }
                )
                task_id += 1
        task_path.parent.mkdir(parents=True, exist_ok=True)
        with task_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=rows[0].keys(),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)


def _progen2_config(seed: int, spec_root: Path) -> tuple[Path, str]:
    run_root = RUN_OUTPUT_ROOT / "progen2" / f"seed{seed}"
    config_path = spec_root / f"progen2/seed{seed}.yaml"
    experiments = []
    for experiment in PROGEN2_EXPERIMENTS:
        experiments.append(
            {
                "name": experiment["name"],
                "display_name": experiment["display_name"],
                "checkpoint_dir": str(experiment["checkpoint_dir"]),
                **PROGEN2_REWARD_WEIGHTS,
            }
        )
    payload = {
        "tasks_path": str(spec_root / f"progen2/seed{seed}_tasks.tsv"),
        "generation_output_root": str(run_root / "generation"),
        "foldability_output_root": str(run_root / "foldability"),
        "developability_output_root": str(run_root / "developability"),
        "diversity_output_root": str(run_root / "diversity"),
        "packed_naturalness_scores_path": str(run_root / "naturalness/naturalness.rows.jsonl"),
        "packed_stability_scores_path": str(run_root / "stability/stability.rows.jsonl"),
        "output_markdown_path": str(run_root / "aggregate/progen2_dense_sweep.md"),
        "output_json_path": str(run_root / "aggregate/progen2_dense_sweep.json"),
        "output_rows_path": str(run_root / "aggregate/progen2_dense_sweep.rows.jsonl"),
        "output_naturalness_diversity_plot_path": str(run_root / "aggregate/naturalness_vs_diversity.png"),
        "output_foldability_diversity_plot_path": str(run_root / "aggregate/foldability_vs_diversity.png"),
        "output_stability_diversity_plot_path": str(run_root / "aggregate/stability_vs_diversity.png"),
        "output_developability_diversity_plot_path": str(
            run_root / "aggregate/developability_vs_diversity.png"
        ),
        "output_soft_reward_diversity_plot_path": str(run_root / "aggregate/soft_reward_vs_diversity.png"),
        "official_code_dir": str(RUN_OUTPUT_ROOT.parents[0] / "progen2_official"),
        "tokenizer_path": str(RUN_OUTPUT_ROOT.parents[0] / "progen2_official/tokenizer.json"),
        "prompt_path": str(RUN_OUTPUT_ROOT.parents[0] / "progen2_official/prompts_unconditional.txt"),
        "seed": seed,
        "bf16": False,
        "device": "cuda",
        "num_samples": 512,
        "generation_prompt_batch_size": 1,
        "num_return_sequences": 512,
        "max_new_tokens": 256,
        "top_p": 0.95,
        "temperature_values": list(PROGEN2_TEMPERATURES),
        "calibration_temperature": 0.8,
        "reward_calibration_size": 256,
        "reward_calibration_prompt_batch_size": 128,
        "rewards": {
            "naturalness": {
                "model_name": "esm2_t33_650M_UR50D",
                "device": "cuda",
                "batch_size": 4096,
            },
            "foldability": {"device": "cuda", "batch_size": 64},
            "stability": {
                "model_name_or_path": str(
                    RUN_OUTPUT_ROOT.parents[0] / "progen2_models/temberture_official"
                ),
                "base_model_name_or_path": str(
                    RUN_OUTPUT_ROOT.parents[0] / "progen2_models/prot_bert_bfd"
                ),
                "device": "cuda",
                "batch_size": 8192,
            },
            "developability": {
                "model_name_or_path": str(
                    RUN_OUTPUT_ROOT.parents[0] / "progen2_models/proteinsol_official"
                ),
                "device": "cpu",
                "batch_size": 24,
                "num_workers": 64,
            },
        },
        "experiments": experiments,
    }
    return config_path, "\n".join(_dump_yaml(payload)) + "\n"


def _build_progen2_specs(spec_root: Path) -> None:
    for seed in SEEDS:
        config_path, config_text = _progen2_config(seed, spec_root)
        _write_text(config_path, config_text)
        run_root = RUN_OUTPUT_ROOT / "progen2" / f"seed{seed}"
        task_path = spec_root / f"progen2/seed{seed}_tasks.tsv"
        rows = []
        task_id = 0
        for experiment in PROGEN2_EXPERIMENTS:
            for temperature in PROGEN2_TEMPERATURES:
                leaf = f"temperature_{temperature:g}"
                rows.append(
                    {
                        "task_id": task_id,
                        "experiment": experiment["name"],
                        "display_name": experiment["display_name"],
                        "checkpoint_dir": experiment["checkpoint_dir"],
                        "checkpoint_subdir": "",
                        "naturalness_weight": PROGEN2_REWARD_WEIGHTS["naturalness"],
                        "foldability_weight": PROGEN2_REWARD_WEIGHTS["foldability"],
                        "stability_weight": PROGEN2_REWARD_WEIGHTS["stability"],
                        "developability_weight": PROGEN2_REWARD_WEIGHTS["developability"],
                        "temperature": temperature,
                        "generation_rows_path": run_root
                        / "generation"
                        / experiment["name"]
                        / leaf
                        / "generated.rows.jsonl",
                        "foldability_scores_path": run_root
                        / "foldability"
                        / experiment["name"]
                        / leaf
                        / "foldability.rows.jsonl",
                        "developability_scores_path": run_root
                        / "developability"
                        / experiment["name"]
                        / leaf
                        / "developability.rows.jsonl",
                        "diversity_scores_path": run_root
                        / "diversity"
                        / experiment["name"]
                        / leaf
                        / "diversity.json",
                    }
                )
                task_id += 1
        task_path.parent.mkdir(parents=True, exist_ok=True)
        with task_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=rows[0].keys(),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-root", type=Path, default=SPEC_OUTPUT_ROOT)
    args = parser.parse_args()
    spec_root = args.spec_root
    _build_denovo_specs(spec_root)
    _build_mmgenmol_specs(spec_root)
    _build_progen2_specs(spec_root)
    manifest = {
        "repo_root": str(REPO_REMOTE_ROOT),
        "run_output_root": str(RUN_OUTPUT_ROOT),
        "seeds": list(SEEDS),
        "molecule_sweep": [list(point) for point in MOLECULE_SWEEP],
        "progen2_temperatures": list(PROGEN2_TEMPERATURES),
        "denovo_tasks": len(SEEDS) * len(DE_NOVO_EXPERIMENTS),
        "mmgenmol_tasks_per_seed": len(MMGENMOL_EXPERIMENTS) * len(MOLECULE_SWEEP),
        "progen2_tasks_per_seed": len(PROGEN2_EXPERIMENTS) * len(PROGEN2_TEMPERATURES),
    }
    _write_text(spec_root / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(spec_root)


if __name__ == "__main__":
    main()
