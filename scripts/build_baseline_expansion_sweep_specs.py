#!/usr/bin/env python3
"""Build five-run sweep specs for the two added GRPO baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_rebuttal_dense_sweep_specs as dense


DEFAULT_RUN_ROOT = Path(
    "/public/home/xinwuye/ai4s-tool-joint-train/runs/baseline_expansion_sweep"
)
PROGEN2_TEMPERATURES = tuple(round(index / 10.0, 1) for index in range(1, 13))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--denovo-dmb-checkpoint", type=Path, required=True)
    parser.add_argument("--denovo-entropy-checkpoint", type=Path, required=True)
    parser.add_argument("--mmgenmol-dmb-checkpoint", type=Path, required=True)
    parser.add_argument("--mmgenmol-entropy-checkpoint", type=Path, required=True)
    parser.add_argument("--progen2-dmb-checkpoint", type=Path, required=True)
    parser.add_argument("--progen2-entropy-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--allow-pending-checkpoints",
        action="store_true",
        help="Build specs for checkpoints that are expected from active training jobs.",
    )
    args = parser.parse_args()
    file_fields = (
        "denovo_dmb_checkpoint",
        "denovo_entropy_checkpoint",
        "mmgenmol_dmb_checkpoint",
        "mmgenmol_entropy_checkpoint",
    )
    directory_fields = ("progen2_dmb_checkpoint", "progen2_entropy_checkpoint")
    missing = [
        str(getattr(args, field))
        for field in file_fields
        if not getattr(args, field).is_file()
        or getattr(args, field).stat().st_size == 0
    ]
    missing.extend(
        str(getattr(args, field))
        for field in directory_fields
        if not getattr(args, field).is_dir()
    )
    if missing and not args.allow_pending_checkpoints:
        parser.error("Missing checkpoint paths:\n" + "\n".join(missing))
    return args


def main() -> None:
    args = _parse_args()
    run_root: Path = args.run_root
    spec_root = run_root / "specs"

    dense.RUN_OUTPUT_ROOT = run_root
    dense.SPEC_OUTPUT_ROOT = spec_root
    dense.PROGEN2_TEMPERATURES = PROGEN2_TEMPERATURES
    dense.DE_NOVO_EXPERIMENTS = (
        {
            "category": "baseline",
            "name": "denovo_dmb_2000",
            "display_name": "Diverse Mini-Batch GRPO",
            "checkpoint_path": args.denovo_dmb_checkpoint,
        },
        {
            "category": "baseline",
            "name": "denovo_entropy_2000",
            "display_name": "Entropy-Regularized GRPO",
            "checkpoint_path": args.denovo_entropy_checkpoint,
        },
    )
    dense.MMGENMOL_EXPERIMENTS = (
        {
            "name": "mmgenmol_dmb_1000",
            "checkpoint_path": args.mmgenmol_dmb_checkpoint,
        },
        {
            "name": "mmgenmol_entropy_1000",
            "checkpoint_path": args.mmgenmol_entropy_checkpoint,
        },
    )
    dense.PROGEN2_EXPERIMENTS = (
        {
            "name": "dmb_grpo_step100",
            "display_name": "Diverse Mini-Batch GRPO",
            "checkpoint_dir": args.progen2_dmb_checkpoint,
        },
        {
            "name": "entropy_grpo_step100",
            "display_name": "Entropy-Regularized GRPO",
            "checkpoint_dir": args.progen2_entropy_checkpoint,
        },
    )

    dense._build_denovo_specs(spec_root)
    dense._build_mmgenmol_specs(spec_root)
    dense._build_progen2_specs(spec_root)
    manifest = {
        "profile": "baseline_expansion",
        "repo_root": str(dense.REPO_REMOTE_ROOT),
        "run_output_root": str(run_root),
        "seeds": list(dense.SEEDS),
        "molecule_sweep": [list(point) for point in dense.MOLECULE_SWEEP],
        "progen2_temperatures": list(PROGEN2_TEMPERATURES),
        "denovo_tasks": len(dense.SEEDS) * len(dense.DE_NOVO_EXPERIMENTS),
        "mmgenmol_tasks_per_seed": len(dense.MMGENMOL_EXPERIMENTS)
        * len(dense.MOLECULE_SWEEP),
        "progen2_tasks_per_seed": len(dense.PROGEN2_EXPERIMENTS)
        * len(PROGEN2_TEMPERATURES),
    }
    dense._write_text(
        spec_root / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(spec_root)


if __name__ == "__main__":
    main()
