#!/usr/bin/env python3
"""Recompute six-model de novo scaffold diversity from saved sweep rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from genmol.diversity import (  # noqa: E402
    RELATIVE_SCAFFOLD_DIVERSITY,
    compute_molecular_diversity,
)


DEFAULT_BASE_RUN_ROOT = Path(
    "/public/home/xinwuye/ai4s-tool-joint-train/runs/rebuttal_dense_sweep"
)
DEFAULT_EXPANSION_RUN_ROOT = Path(
    "/public/home/xinwuye/ai4s-tool-joint-train/runs/baseline_expansion_sweep"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/public/home/xinwuye/ai4s-tool-joint-train/runs/"
    "denovo_scaffold_six_model_reanalysis"
)
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
SAMPLES_PER_POINT = 1000


@dataclass(frozen=True)
class ModelSource:
    experiment: str
    run_kind: str
    category: str


MODEL_SOURCES = (
    ModelSource("original_genmol_v2", "base", "main"),
    ModelSource("genmol_denovo_grpo_2000", "base", "main"),
    ModelSource("genmol_denovo_grpo_hbd_2000", "base", "main"),
    ModelSource("denovo_dmb_2000", "expansion", "baseline"),
    ModelSource("denovo_entropy_2000", "expansion", "baseline"),
    ModelSource(
        "genmol_denovo_sgrpo_rewardsum_loo_2000",
        "base",
        "main",
    ),
)


def _finite(value: object, *, context: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{context} must be numeric, got {value!r}") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{context} must be finite, got {value!r}")
    return converted


def _point_key(randomness: object, temperature: object) -> tuple[float, float]:
    return round(float(randomness), 8), round(float(temperature), 8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty raw rows file: {path}")
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise TypeError(
                    f"Expected a JSON object at {path}:{line_number}, "
                    f"got {type(row).__name__}"
                )
            rows.append(row)
    return rows


def _read_summary(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty source summary: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise TypeError(f"Expected a list of objects in {path}")
    return value


def _mean_required(rows: list[dict], key: str, *, context: str) -> float:
    values = [
        _finite(row.get(key), context=f"{context} row {index} field {key}")
        for index, row in enumerate(rows)
    ]
    return sum(values) / len(values)


def _mean_optional(rows: list[dict], key: str, *, context: str) -> float:
    values = []
    for index, row in enumerate(rows):
        value = row.get(key)
        if value is None:
            continue
        converted = float(value)
        if math.isnan(converted):
            continue
        if not math.isfinite(converted):
            raise ValueError(
                f"{context} row {index} field {key} must be finite or null, "
                f"got {value!r}"
            )
        values.append(converted)
    if not values:
        raise ValueError(f"{context} has no finite values for {key}")
    return sum(values) / len(values)


def _assert_close(actual: float, expected: object, *, context: str) -> None:
    expected_float = _finite(expected, context=f"{context} source summary")
    if not math.isclose(actual, expected_float, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"{context} mismatch: raw rows produce {actual:.17g}, "
            f"source summary contains {expected_float:.17g}"
        )


def _source_root(
    source: ModelSource,
    base_run_root: Path,
    expansion_run_root: Path,
) -> Path:
    roots = {
        "base": base_run_root,
        "expansion": expansion_run_root,
    }
    try:
        return roots[source.run_kind]
    except KeyError as exc:
        raise ValueError(
            f"Unknown run_kind={source.run_kind!r} for {source.experiment}"
        ) from exc


def _source_paths(
    source: ModelSource,
    seed: int,
    base_run_root: Path,
    expansion_run_root: Path,
) -> tuple[Path, Path]:
    root = _source_root(source, base_run_root, expansion_run_root)
    aggregate_root = (
        root
        / "denovo"
        / f"seed{seed}"
        / source.category
        / source.experiment
        / "aggregate"
    )
    return aggregate_root / "dense.rows.jsonl", aggregate_root / "dense.json"


def _validate_rows(
    rows: list[dict],
    source: ModelSource,
    seed: int,
    sweep: tuple[tuple[float, float], ...],
    samples_per_point: int,
) -> dict[tuple[float, float], list[dict]]:
    expected_keys = {_point_key(*point) for point in sweep}
    rows_by_point = {key: [] for key in expected_keys}
    indices_by_point = {key: set() for key in expected_keys}
    context = f"seed={seed} experiment={source.experiment}"

    for row_number, row in enumerate(rows, start=1):
        if row.get("experiment") != source.experiment:
            raise ValueError(
                f"{context} row {row_number} has experiment={row.get('experiment')!r}"
            )
        key = _point_key(
            row.get("randomness"),
            row.get("generation_temperature"),
        )
        if key not in rows_by_point:
            raise ValueError(f"{context} row {row_number} has unexpected point {key}")
        sample_index = row.get("sample_index")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise TypeError(
                f"{context} row {row_number} has invalid sample_index={sample_index!r}"
            )
        if sample_index < 0 or sample_index >= samples_per_point:
            raise ValueError(
                f"{context} point={key} has out-of-range sample_index={sample_index}"
            )
        if sample_index in indices_by_point[key]:
            raise ValueError(
                f"{context} point={key} has duplicate sample_index={sample_index}"
            )
        indices_by_point[key].add(sample_index)
        if not isinstance(row.get("is_valid"), bool):
            raise TypeError(f"{context} row {row_number} has non-boolean is_valid")
        if not isinstance(row.get("alert_hit"), bool):
            raise TypeError(f"{context} row {row_number} has non-boolean alert_hit")
        smiles = row.get("smiles")
        if smiles is not None and not isinstance(smiles, str):
            raise TypeError(f"{context} row {row_number} has invalid smiles={smiles!r}")
        rows_by_point[key].append(row)

    expected_total = len(sweep) * samples_per_point
    if len(rows) != expected_total:
        raise ValueError(
            f"{context} has {len(rows)} rows; expected {expected_total}"
        )
    expected_indices = set(range(samples_per_point))
    for key in expected_keys:
        if len(rows_by_point[key]) != samples_per_point:
            raise ValueError(
                f"{context} point={key} has {len(rows_by_point[key])} rows; "
                f"expected {samples_per_point}"
            )
        if indices_by_point[key] != expected_indices:
            missing = sorted(expected_indices - indices_by_point[key])
            raise ValueError(
                f"{context} point={key} has incomplete sample indices: "
                f"missing={missing[:10]}"
            )
    return rows_by_point


def _summary_by_point(
    rows: list[dict],
    source: ModelSource,
    seed: int,
    sweep: tuple[tuple[float, float], ...],
) -> dict[tuple[float, float], dict]:
    expected_keys = {_point_key(*point) for point in sweep}
    output = {}
    for row in rows:
        if row.get("experiment") != source.experiment:
            raise ValueError(
                f"seed={seed} summary for {source.experiment} contains "
                f"experiment={row.get('experiment')!r}"
            )
        key = _point_key(
            row.get("randomness"),
            row.get("generation_temperature"),
        )
        if key in output:
            raise ValueError(
                f"seed={seed} summary for {source.experiment} duplicates point={key}"
            )
        output[key] = row
    if set(output) != expected_keys:
        raise ValueError(
            f"seed={seed} summary grid mismatch for {source.experiment}: "
            f"missing={sorted(expected_keys - set(output))}, "
            f"unexpected={sorted(set(output) - expected_keys)}"
        )
    return output


def _validate_utility_and_metrics(
    rows: list[dict],
    summary: dict,
    *,
    context: str,
) -> None:
    if int(summary.get("num_samples", -1)) != len(rows):
        raise ValueError(
            f"{context} summary num_samples={summary.get('num_samples')!r}; "
            f"expected {len(rows)}"
        )
    raw_metrics = {
        "reward_mean": _mean_required(rows, "reward", context=context),
        "qed_mean": _mean_optional(rows, "qed", context=context),
        "sa_mean": _mean_optional(rows, "sa", context=context),
        "sa_score_mean": _mean_optional(rows, "sa_score", context=context),
        "soft_reward_mean": _mean_optional(rows, "soft_reward", context=context),
        "valid_fraction": sum(row["is_valid"] for row in rows) / len(rows),
        "alert_hit_fraction": sum(row["alert_hit"] for row in rows) / len(rows),
        "invalid_fraction": sum(not row["is_valid"] for row in rows) / len(rows),
    }
    for key, actual in raw_metrics.items():
        _assert_close(actual, summary.get(key), context=f"{context} {key}")


def _reaggregate_source_seed(
    source: ModelSource,
    seed: int,
    base_run_root: Path,
    expansion_run_root: Path,
    sweep: tuple[tuple[float, float], ...],
    samples_per_point: int,
) -> tuple[list[dict], dict]:
    rows_path, summary_path = _source_paths(
        source,
        seed,
        base_run_root,
        expansion_run_root,
    )
    raw_rows = _read_jsonl(rows_path)
    rows_by_point = _validate_rows(
        raw_rows,
        source,
        seed,
        sweep,
        samples_per_point,
    )
    summary_rows = _read_summary(summary_path)
    summaries = _summary_by_point(summary_rows, source, seed, sweep)
    output = []
    for randomness, temperature in sweep:
        key = _point_key(randomness, temperature)
        point_rows = rows_by_point[key]
        context = (
            f"seed={seed} experiment={source.experiment} "
            f"randomness={randomness} temperature={temperature}"
        )
        _validate_utility_and_metrics(
            point_rows,
            summaries[key],
            context=context,
        )
        diversity = compute_molecular_diversity(
            [row["smiles"] for row in point_rows],
            metric=RELATIVE_SCAFFOLD_DIVERSITY,
        )
        if not 0.0 <= diversity <= 1.0:
            raise ValueError(f"{context} produced invalid scaffold diversity={diversity}")
        updated = dict(summaries[key])
        updated["diversity_metric"] = RELATIVE_SCAFFOLD_DIVERSITY
        updated["diversity"] = float(diversity)
        output.append(updated)
    provenance = {
        "seed": seed,
        "experiment": source.experiment,
        "run_kind": source.run_kind,
        "category": source.category,
        "rows_path": str(rows_path),
        "rows": len(raw_rows),
        "rows_bytes": rows_path.stat().st_size,
        "rows_sha256": _sha256(rows_path),
        "summary_path": str(summary_path),
        "summary_bytes": summary_path.stat().st_size,
        "summary_sha256": _sha256(summary_path),
    }
    return output, provenance


def reaggregate(
    *,
    base_run_root: Path,
    expansion_run_root: Path,
    output_root: Path,
    seeds: Iterable[int] = SEEDS,
    sweep: tuple[tuple[float, float], ...] = MOLECULE_SWEEP,
    samples_per_point: int = SAMPLES_PER_POINT,
    model_sources: tuple[ModelSource, ...] = MODEL_SOURCES,
) -> Path:
    frozen_seeds = tuple(int(seed) for seed in seeds)
    if not frozen_seeds or len(set(frozen_seeds)) != len(frozen_seeds):
        raise ValueError("seeds must be non-empty and unique")
    if not sweep or len({_point_key(*point) for point in sweep}) != len(sweep):
        raise ValueError("sweep points must be non-empty and unique")
    if samples_per_point <= 0:
        raise ValueError("samples_per_point must be positive")
    if not model_sources:
        raise ValueError("model_sources must be non-empty")
    model_names = [source.experiment for source in model_sources]
    if len(set(model_names)) != len(model_names):
        raise ValueError("model_sources contains duplicate experiment names")
    if output_root.exists():
        raise FileExistsError(
            f"Output root already exists; refusing to overwrite: {output_root}"
        )
    temporary_root = output_root.with_name(output_root.name + ".tmp")
    if temporary_root.exists():
        raise FileExistsError(
            f"Temporary output root already exists: {temporary_root}"
        )

    temporary_root.mkdir(parents=True)
    provenance = []
    output_records = []
    total_raw_rows = 0
    try:
        for seed in frozen_seeds:
            merged_rows = []
            for source in model_sources:
                rows, record = _reaggregate_source_seed(
                    source,
                    seed,
                    base_run_root,
                    expansion_run_root,
                    sweep,
                    samples_per_point,
                )
                merged_rows.extend(rows)
                provenance.append(record)
                total_raw_rows += int(record["rows"])
            output_path = temporary_root / "denovo" / f"seed{seed}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(merged_rows, indent=2, sort_keys=True) + "\n"
            )
            output_records.append(
                {
                    "seed": seed,
                    "path": str(output_root / "denovo" / f"seed{seed}.json"),
                    "rows": len(merged_rows),
                    "bytes": output_path.stat().st_size,
                    "sha256": _sha256(output_path),
                }
            )

        expected_total = (
            len(frozen_seeds)
            * len(model_sources)
            * len(sweep)
            * samples_per_point
        )
        if total_raw_rows != expected_total:
            raise RuntimeError(
                f"Validated {total_raw_rows} raw rows; expected {expected_total}"
            )
        manifest = {
            "profile": "denovo_scaffold_six_model_reanalysis",
            "diversity_metric": RELATIVE_SCAFFOLD_DIVERSITY,
            "base_run_root": str(base_run_root),
            "expansion_run_root": str(expansion_run_root),
            "seeds": list(frozen_seeds),
            "sweep_points": [list(point) for point in sweep],
            "samples_per_model_point": samples_per_point,
            "models": model_names,
            "total_raw_rows": total_raw_rows,
            "sources": provenance,
            "outputs": output_records,
        }
        manifest_path = temporary_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary_root.replace(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return output_root / "manifest.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-run-root",
        type=Path,
        default=DEFAULT_BASE_RUN_ROOT,
    )
    parser.add_argument(
        "--expansion-run-root",
        type=Path,
        default=DEFAULT_EXPANSION_RUN_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        reaggregate(
            base_run_root=args.base_run_root,
            expansion_run_root=args.expansion_run_root,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
