#!/usr/bin/env python3
"""Recompute three-model mmGenMol scaffold diversity from saved sweep rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from genmol.diversity import (  # noqa: E402
    RELATIVE_SCAFFOLD_DIVERSITY,
    compute_molecular_diversity,
)


DEFAULT_RUN_ROOT = Path(
    "/public/home/xinwuye/ai4s-tool-joint-train/runs/rebuttal_dense_sweep"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/public/home/xinwuye/ai4s-tool-joint-train/runs/"
    "mmgenmol_scaffold_three_model_reanalysis"
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
EXPECTED_NUM_POCKETS = 100
SAMPLES_PER_POCKET = 16


@dataclass(frozen=True)
class ModelSource:
    model_name: str
    uses_unidock_reward: bool


MODEL_SOURCES = (
    ModelSource("original_5500", False),
    ModelSource("grpo_unidock_1000", True),
    ModelSource("sgrpo_unidock_rewardsum_loo_1000", True),
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
        raise FileNotFoundError(f"Missing or empty JSONL file: {path}")
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def _read_json_list(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty JSON file: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise TypeError(f"Expected a list of objects in {path}")
    return value


def _read_tasks(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty task manifest: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    expected_header = {
        "task_id",
        "model_name",
        "sweep_type",
        "sweep_value",
        "randomness",
        "temperature",
        "checkpoint_path",
        "output_path",
    }
    if reader.fieldnames is None or set(reader.fieldnames) != expected_header:
        raise ValueError(f"Unexpected task manifest header in {path}")
    task_ids = [int(row["task_id"]) for row in rows]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"Duplicate task IDs in {path}")
    return rows


def _canonicalize_smiles(smiles: object) -> str | None:
    from rdkit import Chem

    if smiles is None or not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    except Exception:
        return None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute a mean from an empty list")
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute a median from an empty list")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _assert_close(actual: float, expected: object, *, context: str) -> None:
    expected_float = _finite(expected, context=f"{context} source summary")
    if not math.isclose(actual, expected_float, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            f"{context} mismatch: raw rows produce {actual:.17g}, "
            f"source summary contains {expected_float:.17g}"
        )


def _validate_task_grid(
    tasks: list[dict[str, str]],
    model_sources: tuple[ModelSource, ...],
    sweep: tuple[tuple[float, float], ...],
    *,
    context: str,
) -> dict[tuple[str, float, float], dict[str, str]]:
    selected_names = {source.model_name for source in model_sources}
    selected = [task for task in tasks if task["model_name"] in selected_names]
    expected_keys = {
        (source.model_name, *_point_key(*point))
        for source in model_sources
        for point in sweep
    }
    output = {}
    for task in selected:
        if task["sweep_type"] != "paired":
            raise ValueError(f"{context} contains non-paired task {task['task_id']}")
        key = (
            task["model_name"],
            *_point_key(task["randomness"], task["temperature"]),
        )
        if key in output:
            raise ValueError(f"{context} contains duplicate task key={key}")
        output[key] = task
    if set(output) != expected_keys:
        raise ValueError(
            f"{context} task grid mismatch: "
            f"missing={sorted(expected_keys - set(output))}, "
            f"unexpected={sorted(set(output) - expected_keys)}"
        )
    return output


def _summary_grid(
    rows: list[dict],
    model_sources: tuple[ModelSource, ...],
    sweep: tuple[tuple[float, float], ...],
    *,
    context: str,
) -> dict[tuple[str, float, float], dict]:
    selected_names = {source.model_name for source in model_sources}
    selected = [row for row in rows if row.get("model_name") in selected_names]
    expected_keys = {
        (source.model_name, *_point_key(*point))
        for source in model_sources
        for point in sweep
    }
    output = {}
    for row in selected:
        key = (
            row["model_name"],
            *_point_key(row["randomness"], row["temperature"]),
        )
        if key in output:
            raise ValueError(f"{context} contains duplicate summary key={key}")
        output[key] = row
    if set(output) != expected_keys:
        raise ValueError(
            f"{context} summary grid mismatch: "
            f"missing={sorted(expected_keys - set(output))}, "
            f"unexpected={sorted(set(output) - expected_keys)}"
        )
    return output


def _validate_generation_rows(
    rows: list[dict],
    *,
    expected_num_pockets: int,
    samples_per_pocket: int,
    context: str,
) -> tuple[list[str | None], list[int]]:
    expected_rows = expected_num_pockets * samples_per_pocket
    if len(rows) != expected_rows:
        raise ValueError(f"{context} has {len(rows)} generated rows; expected {expected_rows}")
    canonical_smiles = []
    source_indices = []
    for row_number, row in enumerate(rows):
        if "source_index" not in row:
            raise ValueError(f"{context} generated row {row_number} lacks source_index")
        source_index = int(row["source_index"])
        source_indices.append(source_index)
        canonical_smiles.append(_canonicalize_smiles(row.get("smiles")))
    counts = Counter(source_indices)
    if len(counts) != expected_num_pockets:
        raise ValueError(
            f"{context} covers {len(counts)} pockets; expected {expected_num_pockets}"
        )
    bad_counts = {
        source_index: count
        for source_index, count in counts.items()
        if count != samples_per_pocket
    }
    if bad_counts:
        raise ValueError(f"{context} has invalid per-pocket counts: {bad_counts}")
    return canonical_smiles, source_indices


def _docking_by_row(
    rows: list[dict],
    *,
    expected_rows: int,
    context: str,
) -> dict[int, dict]:
    if len(rows) != expected_rows:
        raise ValueError(f"{context} has {len(rows)} docking rows; expected {expected_rows}")
    output = {}
    for row in rows:
        row_idx = int(row.get("row_idx", -1))
        if row_idx < 0 or row_idx >= expected_rows:
            raise ValueError(f"{context} has invalid docking row_idx={row_idx}")
        if row_idx in output:
            raise ValueError(f"{context} has duplicate docking row_idx={row_idx}")
        record = row.get("record")
        if not isinstance(record, dict) or not isinstance(record.get("is_success"), bool):
            raise TypeError(f"{context} docking row {row_idx} has an invalid record")
        output[row_idx] = row
    if set(output) != set(range(expected_rows)):
        raise ValueError(f"{context} docking row_idx coverage is incomplete")
    return output


def _reaggregate_task(
    *,
    source: ModelSource,
    task: dict[str, str],
    source_summary: dict,
    expected_num_pockets: int,
    samples_per_pocket: int,
) -> tuple[dict, dict]:
    context = (
        f"model={source.model_name} randomness={task['randomness']} "
        f"temperature={task['temperature']}"
    )
    if source_summary.get("model_name") != source.model_name:
        raise ValueError(
            f"{context} source summary has model_name="
            f"{source_summary.get('model_name')!r}"
        )
    if not math.isclose(
        float(source_summary.get("sweep_value")),
        float(task["sweep_value"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(f"{context} sweep_value differs between task and summary")
    if source_summary.get("checkpoint_path") != task["checkpoint_path"]:
        raise ValueError(f"{context} checkpoint differs between task and summary")
    reward_weights = source_summary.get("reward_weights")
    if not isinstance(reward_weights, dict):
        raise TypeError(f"{context} source summary lacks reward_weights")
    has_unidock_reward = _finite(
        reward_weights.get("unidock_score"),
        context=f"{context} unidock reward weight",
    ) > 0.0
    if has_unidock_reward != source.uses_unidock_reward:
        raise ValueError(
            f"{context} UniDock reward classification mismatch: "
            f"source={has_unidock_reward}, expected={source.uses_unidock_reward}"
        )
    generated_path = Path(task["output_path"])
    summary_generated_path = Path(source_summary["generated_rows_path"])
    if generated_path != summary_generated_path:
        raise ValueError(
            f"{context} generated path mismatch: task={generated_path}, "
            f"summary={summary_generated_path}"
        )
    docking_path = Path(source_summary["docking_records_path"])
    generated_rows = _read_jsonl(generated_path)
    docking_rows = _read_jsonl(docking_path)
    canonical_smiles, source_indices = _validate_generation_rows(
        generated_rows,
        expected_num_pockets=expected_num_pockets,
        samples_per_pocket=samples_per_pocket,
        context=context,
    )
    expected_rows = expected_num_pockets * samples_per_pocket
    docking_by_row = _docking_by_row(
        docking_rows,
        expected_rows=expected_rows,
        context=context,
    )

    smiles_by_pocket: dict[int, list[str]] = defaultdict(list)
    valid_smiles = []
    docking_success_flags = []
    dock_affinities = []
    score_only_affinities = []
    minimize_affinities = []
    for row_idx, (smiles, source_index) in enumerate(
        zip(canonical_smiles, source_indices)
    ):
        docking_row = docking_by_row[row_idx]
        docking_source_index = int(docking_row.get("source_index", source_index))
        if docking_source_index != source_index:
            raise ValueError(
                f"{context} row {row_idx} source_index mismatch: "
                f"generation={source_index}, docking={docking_source_index}"
            )
        docking_record = docking_row["record"]
        is_success = docking_record["is_success"]
        is_final_valid = smiles is not None and (
            not source.uses_unidock_reward or is_success
        )
        if not is_final_valid:
            continue
        if smiles is None:
            raise AssertionError("Final-valid molecule cannot have null SMILES")
        smiles_by_pocket[source_index].append(smiles)
        valid_smiles.append(smiles)
        docking_success_flags.append(is_success)
        if is_success:
            dock_affinities.append(
                _finite(
                    docking_record.get("dock_affinity"),
                    context=f"{context} row {row_idx} dock_affinity",
                )
            )
            score_only_affinities.append(
                _finite(
                    docking_record.get("score_only_affinity"),
                    context=f"{context} row {row_idx} score_only_affinity",
                )
            )
            minimize_affinities.append(
                _finite(
                    docking_record.get("minimize_affinity"),
                    context=f"{context} row {row_idx} minimize_affinity",
                )
            )

    if len(smiles_by_pocket) != expected_num_pockets:
        raise ValueError(
            f"{context} final-valid rows cover {len(smiles_by_pocket)} pockets; "
            f"expected {expected_num_pockets}"
        )
    pocket_diversities = [
        compute_molecular_diversity(
            smiles_by_pocket[source_index],
            metric=RELATIVE_SCAFFOLD_DIVERSITY,
        )
        for source_index in sorted(smiles_by_pocket)
    ]
    diversity = _mean(pocket_diversities)
    valid_count = len(valid_smiles)
    valid_fraction = valid_count / expected_rows
    unique_valid_count = len(set(valid_smiles))
    duplicate_fraction = 1.0 - unique_valid_count / valid_count
    docking_success_fraction = sum(docking_success_flags) / valid_count

    exact_integer_fields = {
        "num_rows": expected_rows,
        "num_pockets": expected_num_pockets,
        "samples_per_pocket": samples_per_pocket,
        "valid_count": valid_count,
        "unique_valid_count": unique_valid_count,
        "vina_dock_num_docked": len(dock_affinities),
    }
    for field, expected in exact_integer_fields.items():
        if int(source_summary.get(field, -1)) != expected:
            raise ValueError(
                f"{context} {field} mismatch: raw={expected}, "
                f"summary={source_summary.get(field)!r}"
            )
    floating_fields = {
        "valid_fraction": valid_fraction,
        "duplicate_fraction": duplicate_fraction,
        "vina_dock_success_fraction": docking_success_fraction,
        "vina_dock_mean": _mean(dock_affinities),
        "vina_dock_median": _median(dock_affinities),
        "vina_score_mean": _mean(score_only_affinities),
        "vina_min_mean": _mean(minimize_affinities),
    }
    for field, actual in floating_fields.items():
        _assert_close(actual, source_summary.get(field), context=f"{context} {field}")
    for field in ("qed_mean", "sa_score_mean", "soft_reward_mean"):
        _finite(source_summary.get(field), context=f"{context} {field}")

    updated = dict(source_summary)
    updated["diversity_metric"] = RELATIVE_SCAFFOLD_DIVERSITY
    updated["diversity"] = diversity
    updated["diversity_definition"] = (
        "mean over pockets of unique canonical Bemis-Murcko scaffolds divided "
        "by final-valid generated molecules within each pocket"
    )
    provenance = {
        "model_name": source.model_name,
        "randomness": float(task["randomness"]),
        "temperature": float(task["temperature"]),
        "generated_rows_path": str(generated_path),
        "generated_rows": len(generated_rows),
        "generated_bytes": generated_path.stat().st_size,
        "generated_sha256": _sha256(generated_path),
        "docking_rows_path": str(docking_path),
        "docking_rows": len(docking_rows),
        "docking_bytes": docking_path.stat().st_size,
        "docking_sha256": _sha256(docking_path),
        "valid_count": valid_count,
        "pockets": len(smiles_by_pocket),
    }
    return updated, provenance


def reaggregate(
    *,
    run_root: Path,
    output_root: Path,
    seeds: Iterable[int] = SEEDS,
    sweep: tuple[tuple[float, float], ...] = MOLECULE_SWEEP,
    model_sources: tuple[ModelSource, ...] = MODEL_SOURCES,
    expected_num_pockets: int = EXPECTED_NUM_POCKETS,
    samples_per_pocket: int = SAMPLES_PER_POCKET,
) -> Path:
    frozen_seeds = tuple(int(seed) for seed in seeds)
    if not frozen_seeds or len(set(frozen_seeds)) != len(frozen_seeds):
        raise ValueError("seeds must be non-empty and unique")
    if not sweep or len({_point_key(*point) for point in sweep}) != len(sweep):
        raise ValueError("sweep points must be non-empty and unique")
    if expected_num_pockets <= 0 or samples_per_pocket <= 0:
        raise ValueError("pocket and sample counts must be positive")
    model_names = [source.model_name for source in model_sources]
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError("model_sources must be non-empty and unique")
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
    outputs = []
    total_generated_rows = 0
    total_docking_rows = 0
    try:
        for seed in frozen_seeds:
            seed_root = run_root / "mmgenmol" / f"seed{seed}"
            task_path = run_root / "specs" / "mmgenmol" / f"seed{seed}.tsv"
            summary_path = seed_root / "aggregate" / "mmgenmol_dense.json"
            tasks = _validate_task_grid(
                _read_tasks(task_path),
                model_sources,
                sweep,
                context=f"seed={seed}",
            )
            summaries = _summary_grid(
                _read_json_list(summary_path),
                model_sources,
                sweep,
                context=f"seed={seed}",
            )
            merged_rows = []
            for source in model_sources:
                for randomness, temperature in sweep:
                    key = (
                        source.model_name,
                        *_point_key(randomness, temperature),
                    )
                    updated, record = _reaggregate_task(
                        source=source,
                        task=tasks[key],
                        source_summary=summaries[key],
                        expected_num_pockets=expected_num_pockets,
                        samples_per_pocket=samples_per_pocket,
                    )
                    record["seed"] = seed
                    merged_rows.append(updated)
                    provenance.append(record)
                    total_generated_rows += int(record["generated_rows"])
                    total_docking_rows += int(record["docking_rows"])

            output_path = temporary_root / "mmgenmol" / f"seed{seed}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(merged_rows, indent=2, sort_keys=True) + "\n"
            )
            outputs.append(
                {
                    "seed": seed,
                    "path": str(output_root / "mmgenmol" / f"seed{seed}.json"),
                    "rows": len(merged_rows),
                    "bytes": output_path.stat().st_size,
                    "sha256": _sha256(output_path),
                    "source_summary_path": str(summary_path),
                    "source_summary_sha256": _sha256(summary_path),
                }
            )

        expected_total = (
            len(frozen_seeds)
            * len(model_sources)
            * len(sweep)
            * expected_num_pockets
            * samples_per_pocket
        )
        if total_generated_rows != expected_total or total_docking_rows != expected_total:
            raise RuntimeError(
                "Validated row totals do not match the experiment grid: "
                f"generation={total_generated_rows}, docking={total_docking_rows}, "
                f"expected={expected_total}"
            )
        manifest = {
            "profile": "mmgenmol_scaffold_three_model_reanalysis",
            "diversity_metric": RELATIVE_SCAFFOLD_DIVERSITY,
            "run_root": str(run_root),
            "seeds": list(frozen_seeds),
            "sweep_points": [list(point) for point in sweep],
            "models": model_names,
            "num_pockets": expected_num_pockets,
            "samples_per_pocket": samples_per_pocket,
            "samples_per_model_point": expected_num_pockets * samples_per_pocket,
            "total_generated_rows": total_generated_rows,
            "total_docking_rows": total_docking_rows,
            "sources": provenance,
            "outputs": outputs,
        }
        (temporary_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary_root.replace(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return output_root / "manifest.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(
        reaggregate(
            run_root=args.run_root,
            output_root=args.output_root,
        )
    )


if __name__ == "__main__":
    main()
