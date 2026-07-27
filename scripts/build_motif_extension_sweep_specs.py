#!/usr/bin/env python3
"""Build the fixed 25-task motif-extension sweep manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SEEDS = (42, 43, 44, 45, 46)
MODELS = (
    ("motif_original_genmol_v2", "Original", "original_checkpoint"),
    ("motif_grpo_2000", "GRPO", "grpo_checkpoint"),
    ("motif_dmb_2000", "Diverse Mini-Batch GRPO", "dmb_checkpoint"),
    (
        "motif_entropy_2000",
        "Entropy-Regularized GRPO",
        "entropy_checkpoint",
    ),
    ("motif_sgrpo_2000", "SGRPO", "sgrpo_checkpoint"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--grpo-checkpoint", type=Path, required=True)
    parser.add_argument("--dmb-checkpoint", type=Path, required=True)
    parser.add_argument("--entropy-checkpoint", type=Path, required=True)
    parser.add_argument("--sgrpo-checkpoint", type=Path, required=True)
    parser.add_argument("--allow-pending-checkpoints", action="store_true")
    args = parser.parse_args()
    if args.run_root.exists():
        raise FileExistsError(
            f"run root already exists; refusing to overwrite: {args.run_root}"
        )

    checkpoint_by_field = {
        field: getattr(args, field)
        for _, _, field in MODELS
    }
    missing = [
        str(path)
        for path in checkpoint_by_field.values()
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing and not args.allow_pending_checkpoints:
        raise FileNotFoundError(
            "missing checkpoints:\n" + "\n".join(missing)
        )

    spec_dir = args.run_root / "specs"
    spec_dir.mkdir(parents=True)
    manifest_rows = []
    task_index = 0
    for experiment, display_name, checkpoint_field in MODELS:
        checkpoint_path = checkpoint_by_field[checkpoint_field]
        for seed in SEEDS:
            output_dir = (
                args.run_root
                / "results"
                / experiment
                / f"seed{seed}"
            )
            spec_path = spec_dir / f"task-{task_index:03d}.json"
            payload = {
                "experiment": experiment,
                "display_name": display_name,
                "checkpoint_path": str(checkpoint_path),
                "seed": seed,
                "output_dir": str(output_dir),
            }
            spec_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
            manifest_rows.append(
                {
                    "task_index": task_index,
                    "spec_path": str(spec_path),
                    **payload,
                }
            )
            task_index += 1
    manifest_path = args.run_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "task_count": len(manifest_rows),
                "seeds": list(SEEDS),
                "tasks": manifest_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
