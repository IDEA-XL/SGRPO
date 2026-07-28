#!/usr/bin/env python3
"""Audit the legacy paper SGRPO sweep as Morgan internal diversity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import yaml

from genmol.diversity import (
    DEFAULT_DIVERSITY_METRIC,
    MORGAN_INTERNAL_DIVERSITY,
    compute_molecular_diversity,
)


DEFAULT_RUN_ROOT = Path(
    "/public/home/xinwuye/ai4s-tool-joint-train/runs/rebuttal_dense_sweep"
)
DEFAULT_OUTPUT_PATH = Path(
    "/public/home/xinwuye/ai4s-tool-joint-train/runs/"
    "denovo_credit_ablation_sweep/materialized/sgrpo-base-morgan-audit.json"
)
MODEL_ID = "genmol_denovo_sgrpo_rewardsum_loo_2000"
SEEDS = (42, 43, 44, 45, 46)
SWEEP_POINTS = (
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
SAMPLES_PER_POINT = 1000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _point_key(randomness: object, temperature: object) -> tuple[float, float]:
    return round(float(randomness), 8), round(float(temperature), 8)


def _finite(value: object, *, context: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{context} is not finite: {value!r}")
    return converted


def _assert_close(left: float, right: object, *, context: str) -> None:
    expected = _finite(right, context=context)
    if not math.isclose(left, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{context} mismatch: raw={left}, summary={expected}")


def _read_json(path: Path) -> object:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty JSON file: {path}")
    return json.loads(path.read_text())


def _audit_seed(run_root: Path, seed: int) -> dict:
    config_path = (
        run_root
        / "specs"
        / "denovo"
        / f"seed{seed}"
        / "main"
        / f"{MODEL_ID}.yaml"
    )
    if not config_path.is_file() or config_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing SGRPO sweep config: {config_path}")
    config = yaml.safe_load(config_path.read_text())
    resolved_metric = config.get("diversity_metric", DEFAULT_DIVERSITY_METRIC)
    if resolved_metric != MORGAN_INTERNAL_DIVERSITY:
        raise ValueError(
            f"Config {config_path} resolves diversity metric to {resolved_metric!r}"
        )
    if int(config.get("seed", -1)) != seed:
        raise ValueError(f"Seed mismatch in {config_path}")
    if int(config.get("num_samples", -1)) != SAMPLES_PER_POINT:
        raise ValueError(f"Sample count mismatch in {config_path}")
    config_points = tuple(
        (
            float(point["randomness"]),
            float(point["generation_temperature"]),
        )
        for point in config.get("randomness_temperature_pairs", ())
    )
    if config_points != SWEEP_POINTS:
        raise ValueError(f"Sweep grid mismatch in {config_path}")
    experiments = config.get("experiments")
    if (
        not isinstance(experiments, list)
        or len(experiments) != 1
        or experiments[0].get("name") != MODEL_ID
    ):
        raise ValueError(f"Experiment mismatch in {config_path}")

    summary_path = Path(config["output_json_path"])
    raw_path = Path(config["output_rows_path"])
    summaries = _read_json(summary_path)
    if not isinstance(summaries, list) or len(summaries) != len(SWEEP_POINTS):
        raise ValueError(f"Invalid SGRPO summaries: {summary_path}")
    summary_by_point = {
        _point_key(row["randomness"], row["generation_temperature"]): row
        for row in summaries
    }
    expected_points = {_point_key(*point) for point in SWEEP_POINTS}
    if len(summary_by_point) != len(summaries) or set(summary_by_point) != expected_points:
        raise ValueError(f"Summary sweep grid mismatch in {summary_path}")

    raw_by_point: dict[tuple[float, float], list[dict]] = defaultdict(list)
    row_count = 0
    with raw_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank raw row at {raw_path}:{line_number}")
            row = json.loads(line)
            if row.get("experiment") != MODEL_ID:
                raise ValueError(f"Model mismatch at {raw_path}:{line_number}")
            key = _point_key(
                row["randomness"],
                row["generation_temperature"],
            )
            if key not in expected_points:
                raise ValueError(
                    f"Unexpected sweep point {key} at {raw_path}:{line_number}"
                )
            raw_by_point[key].append(row)
            row_count += 1
    expected_rows = len(SWEEP_POINTS) * SAMPLES_PER_POINT
    if row_count != expected_rows:
        raise ValueError(f"{raw_path} has {row_count} rows; expected {expected_rows}")

    for key in expected_points:
        rows = raw_by_point[key]
        if len(rows) != SAMPLES_PER_POINT:
            raise ValueError(f"{raw_path} point {key} has an invalid row count")
        if sorted(int(row["sample_index"]) for row in rows) != list(
            range(SAMPLES_PER_POINT)
        ):
            raise ValueError(f"{raw_path} point {key} has invalid sample indices")
        utility_values = [
            _finite(row["soft_reward"], context=f"{raw_path} soft_reward")
            for row in rows
            if row.get("soft_reward") is not None
            and math.isfinite(float(row["soft_reward"]))
        ]
        if not utility_values:
            raise ValueError(f"No finite utility values in {raw_path} at {key}")
        utility = sum(utility_values) / len(utility_values)
        diversity = compute_molecular_diversity(
            [row.get("smiles") for row in rows],
            metric=MORGAN_INTERNAL_DIVERSITY,
        )
        summary = summary_by_point[key]
        if summary.get("experiment") != MODEL_ID:
            raise ValueError(f"Summary model mismatch in {summary_path} at {key}")
        _assert_close(
            utility,
            summary["soft_reward_mean"],
            context=f"{summary_path} soft_reward_mean at {key}",
        )
        _assert_close(
            float(diversity),
            summary["diversity"],
            context=f"{summary_path} diversity at {key}",
        )

    return {
        "seed": seed,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "raw_rows_path": str(raw_path),
        "raw_rows": row_count,
        "raw_rows_sha256": _sha256(raw_path),
    }


def audit(run_root: Path, output_path: Path) -> Path:
    if DEFAULT_DIVERSITY_METRIC != MORGAN_INTERNAL_DIVERSITY:
        raise RuntimeError(
            "The evaluator default is no longer Morgan internal diversity"
        )
    if output_path.exists():
        raise FileExistsError(f"Audit output already exists: {output_path}")
    sources = [_audit_seed(run_root, seed) for seed in SEEDS]
    if len({record["raw_rows_sha256"] for record in sources}) != len(SEEDS):
        raise ValueError("SGRPO raw generation files are not seed-independent")
    total_rows = sum(int(record["raw_rows"]) for record in sources)
    expected_rows = len(SEEDS) * len(SWEEP_POINTS) * SAMPLES_PER_POINT
    if total_rows != expected_rows:
        raise ValueError(f"Validated {total_rows} rows; expected {expected_rows}")
    payload = {
        "profile": "denovo_sgrpo_base_morgan_audit",
        "model": MODEL_ID,
        "diversity_metric": MORGAN_INTERNAL_DIVERSITY,
        "seeds": list(SEEDS),
        "sweep_points": [list(point) for point in SWEEP_POINTS],
        "samples_per_model_point": SAMPLES_PER_POINT,
        "total_validated_raw_rows": total_rows,
        "sources": sources,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(audit(args.run_root, args.output_path))


if __name__ == "__main__":
    main()
