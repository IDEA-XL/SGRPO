#!/usr/bin/env python3
"""Validate and materialize de novo credit-ablation sweep results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from genmol.diversity import compute_molecular_diversity


DEFAULT_RUN_ROOT = Path(
    "/public/home/xinwuye/ai4s-tool-joint-train/runs/"
    "denovo_credit_ablation_sweep"
)
EXPECTED_DIVERSITY_METRIC = "morgan_internal_diversity"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: object, *, context: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{context} is not finite: {value!r}")
    return converted


def _assert_close(left: float, right: object, *, context: str) -> None:
    expected = _finite(right, context=context)
    if not math.isclose(left, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{context} mismatch: raw={left}, summary={expected}")


def _point_key(randomness: object, temperature: object) -> tuple[float, float]:
    return round(float(randomness), 8), round(float(temperature), 8)


def _read_json(path: Path) -> object:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty JSON file: {path}")
    return json.loads(path.read_text())


def _validate_model_seed(
    *,
    run_root: Path,
    seed: int,
    model: dict,
    sweep_points: tuple[tuple[float, float], ...],
    samples_per_point: int,
) -> dict:
    model_name = str(model["name"])
    checkpoint_path = str(model["checkpoint_path"])
    aggregate_root = (
        run_root
        / "denovo"
        / f"seed{seed}"
        / "credit_ablation"
        / model_name
        / "aggregate"
    )
    summary_path = aggregate_root / "dense.json"
    rows_path = aggregate_root / "dense.rows.jsonl"
    summaries = _read_json(summary_path)
    if not isinstance(summaries, list) or len(summaries) != len(sweep_points):
        raise ValueError(
            f"{summary_path} must contain {len(sweep_points)} summaries"
        )
    summary_by_point = {
        _point_key(row["randomness"], row["generation_temperature"]): row
        for row in summaries
    }
    expected_points = {_point_key(*point) for point in sweep_points}
    if len(summary_by_point) != len(summaries) or set(summary_by_point) != expected_points:
        raise ValueError(f"Sweep grid mismatch in {summary_path}")

    raw_by_point: dict[tuple[float, float], list[dict]] = defaultdict(list)
    row_count = 0
    with rows_path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank raw row at {rows_path}:{line_number}")
            row = json.loads(line)
            if row.get("experiment") != model_name:
                raise ValueError(
                    f"Unexpected experiment at {rows_path}:{line_number}"
                )
            if row.get("checkpoint_path") != checkpoint_path:
                raise ValueError(
                    f"Checkpoint mismatch at {rows_path}:{line_number}"
                )
            if row.get("diversity_metric") != EXPECTED_DIVERSITY_METRIC:
                raise ValueError(
                    f"Diversity metric mismatch at {rows_path}:{line_number}"
                )
            key = _point_key(
                row["randomness"],
                row["generation_temperature"],
            )
            if key not in expected_points:
                raise ValueError(
                    f"Unexpected sweep point {key} at {rows_path}:{line_number}"
                )
            raw_by_point[key].append(row)
            row_count += 1

    expected_rows = len(sweep_points) * samples_per_point
    if row_count != expected_rows:
        raise ValueError(
            f"{rows_path} has {row_count} rows; expected {expected_rows}"
        )

    for key in expected_points:
        rows = raw_by_point[key]
        if len(rows) != samples_per_point:
            raise ValueError(
                f"{rows_path} point {key} has {len(rows)} rows; "
                f"expected {samples_per_point}"
            )
        sample_indices = sorted(int(row["sample_index"]) for row in rows)
        if sample_indices != list(range(samples_per_point)):
            raise ValueError(
                f"{rows_path} point {key} has invalid sample_index coverage"
            )
        summary = summary_by_point[key]
        if summary.get("experiment") != model_name:
            raise ValueError(f"Experiment mismatch in {summary_path} at {key}")
        if summary.get("checkpoint_path") != checkpoint_path:
            raise ValueError(f"Checkpoint mismatch in {summary_path} at {key}")
        if summary.get("diversity_metric") != EXPECTED_DIVERSITY_METRIC:
            raise ValueError(f"Diversity metric mismatch in {summary_path} at {key}")
        if int(summary.get("num_samples", -1)) != samples_per_point:
            raise ValueError(f"Sample count mismatch in {summary_path} at {key}")

        valid_fraction = sum(bool(row["is_valid"]) for row in rows) / len(rows)
        soft_rewards = [
            _finite(row["soft_reward"], context=f"{rows_path} soft_reward")
            for row in rows
            if row.get("soft_reward") is not None
            and math.isfinite(float(row["soft_reward"]))
        ]
        if not soft_rewards:
            raise ValueError(f"No finite utility values in {rows_path} at {key}")
        utility = sum(soft_rewards) / len(soft_rewards)
        diversity = compute_molecular_diversity(
            [row.get("smiles") for row in rows],
            metric=EXPECTED_DIVERSITY_METRIC,
        )
        _assert_close(
            valid_fraction,
            summary["valid_fraction"],
            context=f"{summary_path} valid_fraction at {key}",
        )
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
        "model": model_name,
        "checkpoint_path": checkpoint_path,
        "summary_path": str(summary_path),
        "summary_bytes": summary_path.stat().st_size,
        "summary_sha256": _sha256(summary_path),
        "raw_rows_path": str(rows_path),
        "raw_rows": row_count,
        "raw_rows_bytes": rows_path.stat().st_size,
        "raw_rows_sha256": _sha256(rows_path),
    }


def materialize(run_root: Path, output_root: Path | None = None) -> Path:
    source_manifest_path = run_root / "specs/manifest.json"
    source_manifest = _read_json(source_manifest_path)
    if not isinstance(source_manifest, dict):
        raise TypeError(f"Expected a manifest mapping in {source_manifest_path}")
    if source_manifest.get("profile") != "denovo_credit_ablation":
        raise ValueError(f"Unexpected profile in {source_manifest_path}")
    if source_manifest.get("diversity_metric") != EXPECTED_DIVERSITY_METRIC:
        raise ValueError(f"Unexpected diversity metric in {source_manifest_path}")

    seeds = tuple(int(seed) for seed in source_manifest["seeds"])
    if seeds != (42, 43, 44, 45, 46):
        raise ValueError(f"Unexpected seeds: {seeds}")
    sweep_points = tuple(
        (float(point[0]), float(point[1]))
        for point in source_manifest["sweep_points"]
    )
    if len(sweep_points) != 10 or len(set(sweep_points)) != 10:
        raise ValueError(f"Expected 10 unique sweep points: {sweep_points}")
    samples_per_point = int(source_manifest["num_samples_per_model_point"])
    if samples_per_point != 1000:
        raise ValueError(f"Expected 1000 samples per point, got {samples_per_point}")
    models = source_manifest["models"]
    if not isinstance(models, list) or len(models) != 3:
        raise ValueError("Expected exactly three credit-ablation models")

    target = output_root or run_root / "materialized"
    if target.exists():
        raise FileExistsError(f"Materialized output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent)
    )
    sources = []
    hashes_by_model: dict[str, set[str]] = defaultdict(set)
    try:
        for seed in seeds:
            merged_path = (
                run_root / "denovo" / f"seed{seed}" / "aggregate/denovo_dense.json"
            )
            merged = _read_json(merged_path)
            expected_summary_rows = len(models) * len(sweep_points)
            if not isinstance(merged, list) or len(merged) != expected_summary_rows:
                raise ValueError(
                    f"{merged_path} has invalid summary count; "
                    f"expected {expected_summary_rows}"
                )
            output_path = temporary / "denovo" / f"seed{seed}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(merged_path, output_path)
            for model in models:
                record = _validate_model_seed(
                    run_root=run_root,
                    seed=seed,
                    model=model,
                    sweep_points=sweep_points,
                    samples_per_point=samples_per_point,
                )
                if record["raw_rows_sha256"] in hashes_by_model[record["model"]]:
                    raise ValueError(
                        f"Duplicated seed-level raw generations for {record['model']}"
                    )
                hashes_by_model[record["model"]].add(record["raw_rows_sha256"])
                sources.append(record)

        total_raw_rows = sum(int(record["raw_rows"]) for record in sources)
        expected_total = len(seeds) * len(models) * len(sweep_points) * samples_per_point
        if total_raw_rows != expected_total:
            raise ValueError(
                f"Validated {total_raw_rows} raw rows; expected {expected_total}"
            )
        manifest = {
            **source_manifest,
            "profile": "denovo_credit_ablation_materialized",
            "source_manifest_path": str(source_manifest_path),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "total_validated_raw_rows": total_raw_rows,
            "sources": sources,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary.rename(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target / "manifest.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(materialize(args.run_root, args.output_root))


if __name__ == "__main__":
    main()
