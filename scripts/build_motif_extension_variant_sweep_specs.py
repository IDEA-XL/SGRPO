#!/usr/bin/env python3
"""Build a five-seed motif-extension sweep manifest for one model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SEEDS = (42, 43, 44, 45, 46)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--allow-pending-checkpoint", action="store_true")
    args = parser.parse_args()

    if args.checkpoint_step <= 0:
        raise ValueError("checkpoint_step must be positive")
    if not args.experiment.strip() or not args.display_name.strip():
        raise ValueError("experiment and display_name must be non-empty")
    checkpoint_ready = (
        args.checkpoint.is_file()
        and args.checkpoint.stat().st_size > 0
    )
    if not checkpoint_ready and not args.allow_pending_checkpoint:
        raise FileNotFoundError(
            f"checkpoint is missing or empty: {args.checkpoint}"
        )
    if args.run_root.exists():
        raise FileExistsError(
            f"run root already exists; refusing to overwrite: {args.run_root}"
        )

    spec_dir = args.run_root / "specs"
    spec_dir.mkdir(parents=True)
    tasks = []
    for task_index, seed in enumerate(SEEDS):
        output_dir = (
            args.run_root
            / "results"
            / args.experiment
            / f"seed{seed}"
        )
        spec_path = spec_dir / f"task-{task_index:03d}.json"
        payload = {
            "checkpoint_path": str(args.checkpoint),
            "display_name": args.display_name,
            "experiment": args.experiment,
            "output_dir": str(output_dir),
            "seed": seed,
        }
        spec_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        tasks.append(
            {
                "task_index": task_index,
                "spec_path": str(spec_path),
                **payload,
            }
        )

    manifest_path = args.run_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "checkpoint_step": args.checkpoint_step,
                "seeds": list(SEEDS),
                "task_count": len(tasks),
                "tasks": tasks,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
