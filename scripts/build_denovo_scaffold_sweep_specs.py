#!/usr/bin/env python3
"""Build five-run de novo sweep specs evaluated with scaffold diversity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_rebuttal_dense_sweep_specs as dense


DEFAULT_RUN_ROOT = Path(
    "/public/home/xinwuye/ai4s-tool-joint-train/runs/denovo_scaffold_sweep"
)
SCAFFOLD_DIVERSITY_METRIC = "relative_scaffold_diversity"


def _checkpoint_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file() or path.stat().st_size == 0:
        raise argparse.ArgumentTypeError(f"Checkpoint file does not exist or is empty: {path}")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--grpo-checkpoint", type=_checkpoint_file, required=True)
    parser.add_argument("--sgrpo-checkpoint", type=_checkpoint_file, required=True)
    parser.add_argument("--entropy-checkpoint", type=_checkpoint_file, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_root: Path = args.run_root
    spec_root = run_root / "specs"
    if run_root.exists():
        raise FileExistsError(f"Run root already exists; refusing to overwrite it: {run_root}")

    dense.RUN_OUTPUT_ROOT = run_root
    dense.SPEC_OUTPUT_ROOT = spec_root
    dense.DENOVO_DIVERSITY_METRIC = SCAFFOLD_DIVERSITY_METRIC
    dense.DE_NOVO_EXPERIMENTS = (
        {
            "category": "scaffold",
            "name": "scaffold_original_genmol_v2",
            "display_name": "Original",
            "checkpoint_path": (
                dense.REPO_REMOTE_ROOT
                / "checkpoints/genmol_v2_v1.0/model_v2.ckpt"
            ),
        },
        {
            "category": "scaffold",
            "name": "scaffold_grpo_2000",
            "display_name": "GRPO",
            "checkpoint_path": args.grpo_checkpoint,
        },
        {
            "category": "scaffold",
            "name": "scaffold_entropy_2000",
            "display_name": "Entropy-Regularized GRPO",
            "checkpoint_path": args.entropy_checkpoint,
        },
        {
            "category": "scaffold",
            "name": "scaffold_sgrpo_2000",
            "display_name": "SGRPO",
            "checkpoint_path": args.sgrpo_checkpoint,
        },
    )

    # The shared controller currently validates every domain's spec graph before
    # selecting REBUTTAL_DOMAINS=denovo. Build the unused specs but never submit them.
    dense._build_denovo_specs(spec_root)
    dense._build_mmgenmol_specs(spec_root)
    dense._build_progen2_specs(spec_root)
    manifest = {
        "profile": "denovo_scaffold_diversity",
        "repo_root": str(dense.REPO_REMOTE_ROOT),
        "run_output_root": str(run_root),
        "seeds": list(dense.SEEDS),
        "molecule_sweep": [list(point) for point in dense.MOLECULE_SWEEP],
        "diversity_metric": SCAFFOLD_DIVERSITY_METRIC,
        "denovo_experiments": [
            {
                "name": experiment["name"],
                "display_name": experiment["display_name"],
                "checkpoint_path": str(experiment["checkpoint_path"]),
            }
            for experiment in dense.DE_NOVO_EXPERIMENTS
        ],
        "denovo_tasks": len(dense.SEEDS) * len(dense.DE_NOVO_EXPERIMENTS),
    }
    dense._write_text(
        spec_root / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(spec_root)


if __name__ == "__main__":
    main()
