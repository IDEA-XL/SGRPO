#!/usr/bin/env python3
"""Build the independent five-run de novo credit-ablation sweep specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import build_rebuttal_dense_sweep_specs as dense


DEFAULT_RUN_ROOT = Path(
    "/public/home/xinwuye/ai4s-tool-joint-train/runs/"
    "denovo_credit_ablation_sweep"
)
RUNS_ROOT = Path("/public/home/xinwuye/ai4s-tool-joint-train/runs")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    checkpoint_path: Path


DEFAULT_MODELS = (
    ModelSpec(
        name="denovo_raw_loo_diversity_2000",
        display_name="Raw-LOO Diversity",
        checkpoint_path=RUNS_ROOT
        / "cpgrpo_denovo/"
        "cpgrpo_denovo_candidate_rawloo09_ng512_bs2048_lr5e-5_beta5e-3_ni1_"
        "ms2000_4gpu_20260726_133332/checkpoint-002000/model.ckpt",
    ),
    ModelSpec(
        name="denovo_mean_baseline_2000",
        display_name="Mean Baseline",
        checkpoint_path=RUNS_ROOT
        / "cpgrpo_denovo/"
        "cpgrpo_denovo_mean_baseline_ng512_bs2048_lr5e-5_beta5e-3_ni1_"
        "ms2000_4gpu_20260726_164325/checkpoint-002000/model.ckpt",
    ),
    ModelSpec(
        name="denovo_mean_baseline_std_2000",
        display_name="Mean Baseline + Std",
        checkpoint_path=RUNS_ROOT
        / "cpgrpo_denovo/"
        "cpgrpo_denovo_mean_baseline_std_ng512_bs2048_lr5e-5_beta5e-3_ni1_"
        "ms2000_4gpu_20260727_063420/checkpoint-002000/model.ckpt",
    ),
)


def _validate_models(models: tuple[ModelSpec, ...]) -> None:
    if not models:
        raise ValueError("At least one model is required")
    names = [model.name for model in models]
    if len(set(names)) != len(names):
        raise ValueError(f"Model names must be unique: {names}")
    missing = [
        str(model.checkpoint_path)
        for model in models
        if not model.checkpoint_path.is_file()
        or model.checkpoint_path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            "Missing or empty credit-ablation checkpoints:\n"
            + "\n".join(missing)
        )


def build_specs(
    run_root: Path,
    models: tuple[ModelSpec, ...] = DEFAULT_MODELS,
    *,
    overwrite_specs: bool = False,
) -> Path:
    _validate_models(models)
    spec_root = run_root / "specs"
    manifest_path = spec_root / "manifest.json"
    if manifest_path.exists() and not overwrite_specs:
        raise FileExistsError(
            f"Sweep specs already exist: {manifest_path}; "
            "pass --overwrite-specs only to recreate the same experiment specs"
        )

    dense.RUN_OUTPUT_ROOT = run_root
    dense.SPEC_OUTPUT_ROOT = spec_root
    dense.DENOVO_DIVERSITY_METRIC = "morgan_internal_diversity"
    dense.DE_NOVO_EXPERIMENTS = tuple(
        {
            "category": "credit_ablation",
            "name": model.name,
            "display_name": model.display_name,
            "checkpoint_path": model.checkpoint_path,
        }
        for model in models
    )
    dense._build_denovo_specs(spec_root)

    manifest = {
        "profile": "denovo_credit_ablation",
        "run_output_root": str(run_root),
        "seeds": list(dense.SEEDS),
        "sweep_points": [list(point) for point in dense.MOLECULE_SWEEP],
        "diversity_metric": dense.DENOVO_DIVERSITY_METRIC,
        "num_samples_per_model_point": 1000,
        "generation_batch_size": 2048,
        "models": [
            {
                "name": model.name,
                "display_name": model.display_name,
                "checkpoint_path": str(model.checkpoint_path),
                "checkpoint_bytes": model.checkpoint_path.stat().st_size,
            }
            for model in models
        ],
        "denovo_tasks": len(dense.SEEDS) * len(models),
        "total_expected_raw_rows": (
            len(dense.SEEDS)
            * len(models)
            * len(dense.MOLECULE_SWEEP)
            * 1000
        ),
    }
    dense._write_text(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--overwrite-specs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        build_specs(
            args.run_root,
            overwrite_specs=args.overwrite_specs,
        )
    )


if __name__ == "__main__":
    main()
