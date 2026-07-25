#!/usr/bin/env python3
"""Assemble and validate per-experiment dense de novo sweep summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SEEDS = (42, 43, 44, 45, 46)
EXPECTED_POINTS_PER_EXPERIMENT = 10
EXPECTED_EXPERIMENTS = 12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("/public/home/xinwuye/ai4s-tool-joint-train/runs/rebuttal_dense_sweep"),
    )
    parser.add_argument(
        "--expected-experiments",
        type=int,
        default=EXPECTED_EXPERIMENTS,
    )
    parser.add_argument(
        "--expected-points-per-experiment",
        type=int,
        default=EXPECTED_POINTS_PER_EXPERIMENT,
    )
    args = parser.parse_args()
    if args.expected_experiments <= 0:
        raise ValueError("--expected-experiments must be positive")
    if args.expected_points_per_experiment <= 0:
        raise ValueError("--expected-points-per-experiment must be positive")

    for seed in SEEDS:
        seed_root = args.run_root / "denovo" / f"seed{seed}"
        paths = sorted(seed_root.glob("*/*/aggregate/dense.json"))
        if len(paths) != args.expected_experiments:
            raise RuntimeError(
                f"Expected {args.expected_experiments} experiment summaries for seed "
                f"{seed}, found {len(paths)}"
            )
        merged = []
        experiments = set()
        for path in paths:
            rows = json.loads(path.read_text())
            if (
                not isinstance(rows, list)
                or len(rows) != args.expected_points_per_experiment
            ):
                raise RuntimeError(
                    f"Expected {args.expected_points_per_experiment} rows in {path}, "
                    f"found {len(rows)}"
                )
            row_experiments = {str(row["experiment"]) for row in rows}
            if len(row_experiments) != 1:
                raise RuntimeError(f"Expected one experiment in {path}, found {row_experiments}")
            experiment = next(iter(row_experiments))
            if experiment in experiments:
                raise RuntimeError(f"Duplicate experiment {experiment} for seed {seed}")
            experiments.add(experiment)
            merged.extend(rows)
        output_path = seed_root / "aggregate/denovo_dense.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
        print(output_path)


if __name__ == "__main__":
    main()
