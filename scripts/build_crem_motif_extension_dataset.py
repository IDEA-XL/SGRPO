#!/usr/bin/env python3
"""Build a strict single-attachment motif-extension training set from CReM."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from genmol.rl.motif import (  # noqa: E402
    MotifRecord,
    canonicalize_single_attachment_fragment,
    load_test_motif_records,
)


V1_RESERVED_RADIUS_COLUMNS = {
    "env_id",
    "core_smi_id",
    "core_num_atoms",
    "dist2",
    "is_ring_closure",
}


def _table_columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
    rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    return tuple(str(row[1]) for row in rows)


def _quoted_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _iter_core_frequencies(
    connection: sqlite3.Connection,
    *,
    set_name: str | None,
):
    user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    radius_columns = set(_table_columns(connection, "radius3"))
    if not radius_columns:
        raise ValueError("CReM database has no radius3 table")

    if user_version == 0:
        required = {"core_smi", "freq"}
        missing = required - radius_columns
        if missing:
            raise ValueError(
                "CReM v0 radius3 is missing required columns: "
                f"{sorted(missing)}"
            )
        if set_name is not None:
            raise ValueError("--set-name is not valid for a CReM v0 database")
        rows = connection.execute(
            """
            SELECT core_smi, SUM(freq)
            FROM radius3
            GROUP BY core_smi
            """
        )
        for core_smi, frequency in rows:
            yield str(core_smi), int(frequency)
        return

    if user_version != 1:
        raise ValueError(
            f"unsupported CReM PRAGMA user_version={user_version}; expected 0 or 1"
        )
    fragment_columns = set(_table_columns(connection, "frags"))
    required_fragments = {"core_smi_id", "core_smi"}
    missing_fragments = required_fragments - fragment_columns
    if missing_fragments:
        raise ValueError(
            "CReM v1 frags is missing required columns: "
            f"{sorted(missing_fragments)}"
        )
    frequency_columns = sorted(radius_columns - V1_RESERVED_RADIUS_COLUMNS)
    if set_name is None:
        if len(frequency_columns) != 1:
            raise ValueError(
                "CReM v1 database must have exactly one frequency set or "
                "--set-name must be provided; available sets: "
                f"{frequency_columns}"
            )
        set_name = frequency_columns[0]
    if set_name not in frequency_columns:
        raise ValueError(
            f"frequency set {set_name!r} not found; available sets: "
            f"{frequency_columns}"
        )
    frequency_identifier = _quoted_identifier(set_name)
    ring_filter = (
        "WHERE r.is_ring_closure = 0"
        if "is_ring_closure" in radius_columns
        else ""
    )
    rows = connection.execute(
        f"""
        SELECT f.core_smi, SUM(r.{frequency_identifier})
        FROM radius3 AS r
        JOIN frags AS f ON r.core_smi_id = f.core_smi_id
        {ring_filter}
        GROUP BY f.core_smi
        """
    )
    for core_smi, frequency in rows:
        yield str(core_smi), int(frequency)


def build_dataset(
    *,
    database_path: Path,
    test_fragments_path: Path,
    output_path: Path,
    metadata_path: Path,
    min_frequency: int,
    min_heavy_atoms: int,
    max_heavy_atoms: int,
    set_name: str | None,
) -> dict:
    if not database_path.is_file() or database_path.stat().st_size == 0:
        raise FileNotFoundError(f"CReM database not found or empty: {database_path}")
    if min_frequency <= 0:
        raise ValueError("min_frequency must be positive")
    if min_heavy_atoms <= 0:
        raise ValueError("min_heavy_atoms must be positive")
    if max_heavy_atoms < min_heavy_atoms:
        raise ValueError("max_heavy_atoms must be at least min_heavy_atoms")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_path}")
    if metadata_path.exists():
        raise FileExistsError(f"refusing to overwrite metadata: {metadata_path}")

    test_records = load_test_motif_records(test_fragments_path)
    test_smiles = {record.smiles for record in test_records}
    canonical_frequencies: dict[str, int] = defaultdict(int)
    counts = defaultdict(int)

    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
        for raw_smiles, frequency in _iter_core_frequencies(
            connection,
            set_name=set_name,
        ):
            counts["database_core_count"] += 1
            try:
                canonical = canonicalize_single_attachment_fragment(
                    raw_smiles,
                    min_heavy_atoms=min_heavy_atoms,
                    max_heavy_atoms=max_heavy_atoms,
                )
            except ValueError:
                counts["invalid_or_filtered_structure_count"] += 1
                continue
            canonical_frequencies[canonical] += int(frequency)

    selected = []
    for canonical, frequency in sorted(canonical_frequencies.items()):
        if frequency < min_frequency:
            counts["below_frequency_count"] += 1
            continue
        if canonical in test_smiles:
            counts["test_overlap_removed_count"] += 1
            continue
        selected.append(
            MotifRecord(
                motif_id=f"crem-{len(selected):07d}",
                smiles=canonical,
                frequency=frequency,
            )
        )
    if not selected:
        raise RuntimeError("CReM filtering produced an empty motif dataset")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as handle:
        for record in selected:
            handle.write(
                json.dumps(record.__dict__, sort_keys=True) + "\n"
            )

    database_hash = hashlib.sha256()
    with database_path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            database_hash.update(chunk)
    output_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    metadata = {
        "database_path": str(database_path),
        "database_sha256": database_hash.hexdigest(),
        "test_fragments_path": str(test_fragments_path),
        "output_path": str(output_path),
        "output_sha256": output_hash,
        "set_name": set_name,
        "radius": 3,
        "min_frequency": min_frequency,
        "min_heavy_atoms": min_heavy_atoms,
        "max_heavy_atoms": max_heavy_atoms,
        "test_motif_count": len(test_records),
        "selected_motif_count": len(selected),
        **{key: int(value) for key, value in sorted(counts.items())},
    }
    with metadata_path.open("x") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument(
        "--test-fragments",
        type=Path,
        default=REPO_ROOT / "data/fragments.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--min-frequency", type=int, default=5)
    parser.add_argument("--min-heavy-atoms", type=int, default=5)
    parser.add_argument("--max-heavy-atoms", type=int, default=12)
    parser.add_argument("--set-name")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    metadata = build_dataset(
        database_path=args.database,
        test_fragments_path=args.test_fragments,
        output_path=args.output,
        metadata_path=args.metadata,
        min_frequency=args.min_frequency,
        min_heavy_atoms=args.min_heavy_atoms,
        max_heavy_atoms=args.max_heavy_atoms,
        set_name=args.set_name,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
