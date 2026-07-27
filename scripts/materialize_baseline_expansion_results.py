#!/usr/bin/env python3
"""Validate and materialize baseline-expansion sweep summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import yaml


DEFAULT_RUN_ROOT = Path(
    "/public/home/xinwuye/ai4s-tool-joint-train/runs/baseline_expansion_sweep"
)
SEEDS = (42, 43, 44, 45, 46)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"No rows in task manifest: {path}")
    return rows


def _expected_counts(run_root: Path) -> dict[tuple[str, int], int]:
    spec_root = run_root / "specs"
    denovo_rows = _read_tsv(spec_root / "denovo/tasks.tsv")
    counts: dict[tuple[str, int], int] = {}
    for seed in SEEDS:
        expected = 0
        for row in denovo_rows:
            if int(row["seed"]) != seed:
                continue
            config_path = Path(row["config_path"])
            if not config_path.is_file():
                raise FileNotFoundError(config_path)
            config = yaml.safe_load(config_path.read_text())
            points = config.get("randomness_temperature_pairs")
            if not isinstance(points, list) or not points:
                raise ValueError(
                    f"Missing randomness_temperature_pairs in {config_path}"
                )
            expected += len(points)
        if expected == 0:
            raise ValueError(f"No de novo sweep points for seed {seed}")
        counts[("denovo", seed)] = expected
        counts[("mmgenmol", seed)] = len(
            _read_tsv(spec_root / f"mmgenmol/seed{seed}.tsv")
        )
        counts[("progen2", seed)] = len(
            _read_tsv(spec_root / f"progen2/seed{seed}_tasks.tsv")
        )
    return counts


def _source_path(run_root: Path, domain: str, seed: int) -> Path:
    names = {
        "denovo": "denovo_dense.json",
        "mmgenmol": "mmgenmol_dense.json",
        "progen2": "progen2_dense_sweep.json",
    }
    return run_root / domain / f"seed{seed}" / "aggregate" / names[domain]


def _validated_row_count(path: Path, domain: str) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty aggregate: {path}")
    value = json.loads(path.read_text())
    if domain == "progen2":
        if not isinstance(value, dict) or not isinstance(value.get("results"), list):
            raise TypeError(f"Expected a dict with a results list in {path}")
        rows = value["results"]
    else:
        if not isinstance(value, list):
            raise TypeError(f"Expected a list in {path}")
        rows = value
    return len(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(run_root: Path, output_root: Path) -> Path:
    complete_path = run_root / "COMPLETE"
    if not complete_path.is_file():
        raise FileNotFoundError(
            f"Sweep controller has not produced its completion marker: {complete_path}"
        )
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to mix materialized results in existing path: {output_root}"
        )

    expected_counts = _expected_counts(run_root)
    records = []
    temporary_root = output_root.with_name(output_root.name + ".tmp")
    if temporary_root.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary_root}")
    temporary_root.mkdir(parents=True)
    try:
        for domain in ("denovo", "mmgenmol", "progen2"):
            for seed in SEEDS:
                source = _source_path(run_root, domain, seed)
                actual_count = _validated_row_count(source, domain)
                expected_count = expected_counts[(domain, seed)]
                if actual_count != expected_count:
                    raise ValueError(
                        f"{source} has {actual_count} rows; expected {expected_count}"
                    )
                destination = temporary_root / domain / f"seed{seed}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                if source.stat().st_size != destination.stat().st_size:
                    raise RuntimeError(f"Size mismatch after copying {source}")
                source_sha256 = _sha256(source)
                if _sha256(destination) != source_sha256:
                    raise RuntimeError(f"SHA-256 mismatch after copying {source}")
                records.append(
                    {
                        "domain": domain,
                        "seed": seed,
                        "rows": actual_count,
                        "source": str(source),
                        "destination": str(
                            output_root / domain / f"seed{seed}.json"
                        ),
                        "bytes": source.stat().st_size,
                        "sha256": source_sha256,
                    }
                )
        manifest_path = temporary_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "run_root": str(run_root),
                    "completion_marker": complete_path.read_text().strip(),
                    "files": records,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        temporary_root.replace(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return output_root / "manifest.json"


def main() -> None:
    args = _parse_args()
    run_root: Path = args.run_root
    output_root = args.output_root or run_root / "materialized-results"
    print(materialize(run_root, output_root))


if __name__ == "__main__":
    main()
