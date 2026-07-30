#!/usr/bin/env python3
"""Recompute motif-extension metrics without filtering on motif retention."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path


VIS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = VIS_ROOT.parent
sys.path.insert(0, str(VIS_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
import render_motif_extension_group_weight_comparisons as comparison  # noqa: E402
import render_motif_extension_results as motif  # noqa: E402
import render_rebuttal_dense_results as dense  # noqa: E402
from genmol.diversity import (  # noqa: E402
    MORGAN_INTERNAL_DIVERSITY,
    compute_molecular_diversity,
)
from genmol.rl.reward import MolecularReward  # noqa: E402


MODE = "raw_generation_without_motif_retention_filter"


@dataclass(frozen=True)
class PendingTask:
    model: dense.ModelSpec
    seed: int
    rows_path: Path
    rows_sha256: str
    rows: tuple[dict, ...]


class RawGenerationStore(dense.ResultStore):
    def __init__(self, rows_by_seed: dict[int, list[dict]]):
        self._rows_by_seed = rows_by_seed

    def rows(self, kind: str, seed: int) -> list[dict]:
        if kind != "denovo":
            raise ValueError(f"unsupported result kind: {kind}")
        return self._rows_by_seed[seed]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(
    cache_root: Path,
    model: dense.ModelSpec,
    seed: int,
) -> Path:
    return cache_root / model.source_id / f"seed{seed}.json"


def _load_cache(
    path: Path,
    *,
    model: dense.ModelSpec,
    seed: int,
    rows_sha256: str,
) -> list[dict] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    expected = {
        "mode": MODE,
        "experiment": model.source_id,
        "display_name": model.label,
        "seed": seed,
        "rows_sha256": rows_sha256,
        "diversity_metric": MORGAN_INTERNAL_DIVERSITY,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"cache metadata mismatch for {path}: {field}"
            )
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(
        dense.MOLECULE_SWEEP
    ):
        raise ValueError(f"invalid cached result count: {path}")
    return results


def _write_cache(
    path: Path,
    *,
    task: PendingTask,
    results: list[dict],
) -> None:
    payload = {
        "mode": MODE,
        "experiment": task.model.source_id,
        "display_name": task.model.label,
        "seed": task.seed,
        "rows_sha256": task.rows_sha256,
        "diversity_metric": MORGAN_INTERNAL_DIVERSITY,
        "results": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    if temp_path.exists():
        raise FileExistsError(f"stale cache temp file: {temp_path}")
    with temp_path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temp_path, path)


def _read_raw_rows(
    store: motif.MotifResultStore,
    model: dense.ModelSpec,
    seed: int,
) -> tuple[Path, str, tuple[dict, ...]]:
    task = store._tasks[(model.source_id, seed)]
    rows_path = (
        store.results_root
        / "results"
        / model.source_id
        / f"seed{seed}"
        / "rows.jsonl"
    )
    rows_sha256 = _sha256(rows_path)
    summary_path = rows_path.with_name("summary.json")
    summary = json.loads(summary_path.read_text())
    if summary["metadata"]["rows_sha256"] != rows_sha256:
        raise ValueError(f"raw row hash mismatch: {rows_path}")
    if task["output_dir"] != str(rows_path.parent):
        raise ValueError(f"manifest output mismatch: {rows_path}")
    rows = tuple(json.loads(line) for line in rows_path.read_text().splitlines())
    if len(rows) != 10_000:
        raise ValueError(f"expected 10,000 rows in {rows_path}")
    return rows_path, rows_sha256, rows


def _finite_mean(values: list[float | None], *, context: str) -> float:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not finite:
        raise ValueError(f"no finite values for {context}")
    return sum(finite) / len(finite)


def _summarize_task(
    task: PendingTask,
    records: list,
) -> list[dict]:
    if len(records) != len(task.rows):
        raise ValueError(
            f"reward count mismatch for {task.model.source_id}, "
            f"seed={task.seed}"
        )
    grouped: dict[tuple[int, int], list] = {}
    for row, record in zip(task.rows, records):
        key = (int(row["point_index"]), int(row["motif_index"]))
        grouped.setdefault(key, []).append((row, record))

    expected_keys = {
        (point_index, motif_index)
        for point_index in range(10)
        for motif_index in range(10)
    }
    if set(grouped) != expected_keys:
        raise ValueError(
            f"incomplete raw groups for {task.model.source_id}, "
            f"seed={task.seed}"
        )

    results = []
    for point_index, (randomness, temperature) in enumerate(
        dense.MOLECULE_SWEEP
    ):
        point_records = []
        per_motif_diversity = []
        for motif_index in range(10):
            pairs = sorted(
                grouped[(point_index, motif_index)],
                key=lambda pair: int(pair[0]["sample_index"]),
            )
            if len(pairs) != 100 or [
                int(row["sample_index"]) for row, _ in pairs
            ] != list(range(100)):
                raise ValueError(
                    f"invalid samples for {task.model.source_id}, "
                    f"seed={task.seed}, point={point_index}, "
                    f"motif={motif_index}"
                )
            motif_records = [record for _, record in pairs]
            point_records.extend(motif_records)
            per_motif_diversity.append(
                compute_molecular_diversity(
                    [record.smiles for record in motif_records],
                    metric=MORGAN_INTERNAL_DIVERSITY,
                )
            )
        results.append(
            {
                "experiment": task.model.source_id,
                "display_name": task.model.label,
                "seed": task.seed,
                "point_index": point_index,
                "randomness": randomness,
                "generation_temperature": temperature,
                "soft_reward_mean": _finite_mean(
                    [record.soft_reward for record in point_records],
                    context=(
                        f"{task.model.source_id}, seed={task.seed}, "
                        f"point={point_index}"
                    ),
                ),
                "diversity": (
                    sum(per_motif_diversity)
                    / len(per_motif_diversity)
                ),
                "per_motif_diversity": per_motif_diversity,
                "raw_valid_fraction": (
                    sum(record.is_valid for record in point_records)
                    / len(point_records)
                ),
            }
        )
    return results


def _load_or_compute(
    *,
    base_store: motif.MotifResultStore,
    variant_store: motif.MotifResultStore,
    variant_model: dense.ModelSpec,
    cache_root: Path,
) -> dict[int, list[dict]]:
    rows_by_seed = {seed: [] for seed in dense.SEEDS}
    pending = []
    models_and_stores = [
        *(
            (model, base_store)
            for model in comparison.BASE_MODELS
        ),
        (variant_model, variant_store),
    ]
    for model, source_store in models_and_stores:
        for seed in dense.SEEDS:
            rows_path, rows_sha256, rows = _read_raw_rows(
                source_store,
                model,
                seed,
            )
            cache_path = _cache_path(cache_root, model, seed)
            cached = _load_cache(
                cache_path,
                model=model,
                seed=seed,
                rows_sha256=rows_sha256,
            )
            if cached is not None:
                rows_by_seed[seed].extend(cached)
                continue
            pending.append(
                PendingTask(
                    model=model,
                    seed=seed,
                    rows_path=rows_path,
                    rows_sha256=rows_sha256,
                    rows=rows,
                )
            )

    if pending:
        raw_smiles = [
            row["raw_smiles"]
            for task in pending
            for row in task.rows
        ]
        reward_model = MolecularReward(always_compute_metrics=True)
        try:
            records = reward_model.score(raw_smiles)
        finally:
            reward_model.close()
        if len(records) != len(raw_smiles):
            raise ValueError("global raw reward count mismatch")
        offset = 0
        for task in pending:
            end = offset + len(task.rows)
            results = _summarize_task(task, records[offset:end])
            offset = end
            _write_cache(
                _cache_path(cache_root, task.model, task.seed),
                task=task,
                results=results,
            )
            rows_by_seed[task.seed].extend(results)
        if offset != len(records):
            raise ValueError("unused raw reward records")

    return rows_by_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run-root", type=Path, required=True)
    parser.add_argument("--variant-run-root", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=tuple(comparison.VARIANT_MODELS),
        required=True,
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    variant_model = comparison.VARIANT_MODELS[args.variant]
    base_store = motif.MotifResultStore(
        args.base_run_root,
        panel=comparison.BASE_PANEL,
    )
    base_store.validate_raw_seed_independence()
    validation_panel = comparison._single_model_panel(
        args.variant,
        variant_model,
    )
    variant_store = motif.MotifResultStore(
        args.variant_run_root,
        panel=validation_panel,
    )
    variant_store.validate_raw_seed_independence()

    rows_by_seed = _load_or_compute(
        base_store=base_store,
        variant_store=variant_store,
        variant_model=variant_model,
        cache_root=args.cache_root,
    )
    store = RawGenerationStore(rows_by_seed)
    panel = comparison._comparison_panel(args.variant, variant_model)
    dense.validate_seed_independence(
        store,
        (panel,),
        include_denovo_auxiliary=False,
    )
    metrics, references = dense.table1_metrics(store, (panel,))
    section = comparison._format_section(panel, metrics, references)
    output = (
        "# Motif-Extension Metrics Without Retention Filtering\n\n"
        "Utility and diversity are recomputed from every raw generated "
        "molecule; motif retention is not used as a filter.\n\n"
        + section
        + "\n"
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output_path.with_suffix(
        args.output_path.suffix + ".tmp"
    )
    if temp_path.exists():
        raise FileExistsError(f"stale output temp file: {temp_path}")
    with temp_path.open("x") as handle:
        handle.write(output)
    os.replace(temp_path, args.output_path)
    print(args.output_path)


if __name__ == "__main__":
    main()
