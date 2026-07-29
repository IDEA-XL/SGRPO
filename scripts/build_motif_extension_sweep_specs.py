#!/usr/bin/env python3
"""Build the fixed 25-task motif-extension sweep manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SEEDS = (42, 43, 44, 45, 46)


def model_specs(checkpoint_step: int) -> tuple[tuple[str, str, str], ...]:
    if checkpoint_step <= 0:
        raise ValueError("checkpoint_step must be positive")
    return (
        ("motif_original_genmol_v2", "Original", "original_checkpoint"),
        (f"motif_grpo_{checkpoint_step}", "GRPO", "grpo_checkpoint"),
        (
            f"motif_dmb_{checkpoint_step}",
            "Diverse Mini-Batch GRPO",
            "dmb_checkpoint",
        ),
        (
            f"motif_entropy_{checkpoint_step}",
            "Entropy-Regularized GRPO",
            "entropy_checkpoint",
        ),
        (f"motif_sgrpo_{checkpoint_step}", "SGRPO", "sgrpo_checkpoint"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--grpo-checkpoint", type=Path, required=True)
    parser.add_argument("--dmb-checkpoint", type=Path, required=True)
    parser.add_argument("--entropy-checkpoint", type=Path, required=True)
    parser.add_argument("--sgrpo-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=2000)
    parser.add_argument("--allow-pending-checkpoints", action="store_true")
    args = parser.parse_args()
    models = model_specs(args.checkpoint_step)
    if args.run_root.exists():
        raise FileExistsError(
            f"run root already exists; refusing to overwrite: {args.run_root}"
        )

    checkpoint_by_field = {
        field: getattr(args, field)
        for _, _, field in models
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
    for experiment, display_name, checkpoint_field in models:
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
                "checkpoint_step": args.checkpoint_step,
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
