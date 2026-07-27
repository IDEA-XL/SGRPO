from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class MotifRecord:
    motif_id: str
    smiles: str
    frequency: int | None = None


def canonicalize_single_attachment_fragment(
    smiles: str,
    *,
    min_heavy_atoms: int | None = None,
    max_heavy_atoms: int | None = None,
) -> str:
    from rdkit import Chem

    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("fragment SMILES must be a non-empty string")
    molecule = Chem.MolFromSmiles(smiles, sanitize=True)
    if molecule is None:
        raise ValueError(f"fragment is not a valid sanitized molecule: {smiles!r}")

    dummy_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) != 1:
        raise ValueError(
            "fragment must contain exactly one attachment-point dummy atom, "
            f"found {len(dummy_atoms)}: {smiles!r}"
        )
    dummy = dummy_atoms[0]
    if dummy.GetDegree() != 1:
        raise ValueError(
            "the attachment-point dummy atom must have exactly one neighbor: "
            f"{smiles!r}"
        )

    heavy_atom_count = sum(
        1 for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1
    )
    if min_heavy_atoms is not None and heavy_atom_count < min_heavy_atoms:
        raise ValueError(
            f"fragment has {heavy_atom_count} heavy atoms, below {min_heavy_atoms}"
        )
    if max_heavy_atoms is not None and heavy_atom_count > max_heavy_atoms:
        raise ValueError(
            f"fragment has {heavy_atom_count} heavy atoms, above {max_heavy_atoms}"
        )

    dummy.SetAtomMapNum(0)
    dummy.SetIsotope(1)
    canonical = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    reparsed = Chem.MolFromSmiles(canonical, sanitize=True)
    if reparsed is None:
        raise ValueError(
            f"canonicalized fragment failed RDKit sanitization: {canonical!r}"
        )
    return canonical


def attachment_free_query(fragment_smiles: str):
    from rdkit import Chem

    canonical = canonicalize_single_attachment_fragment(fragment_smiles)
    molecule = Chem.MolFromSmiles(canonical, sanitize=True)
    dummy_index = next(
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0
    )
    editable = Chem.RWMol(molecule)
    editable.RemoveAtom(dummy_index)
    query = editable.GetMol()
    # The attachment-free object is a query fragment, not a standalone
    # molecule. Re-sanitizing can incorrectly reject aromatic attachment
    # atoms whose missing valence is supplied by the generated extension.
    query.UpdatePropertyCache(strict=False)
    if query.GetNumHeavyAtoms() == 0:
        raise ValueError(
            f"fragment has no non-attachment heavy atoms: {fragment_smiles!r}"
        )
    return query


def molecule_contains_fragment(
    molecule_smiles: str | None,
    fragment_smiles: str,
) -> bool:
    return molecule_contains_query(
        molecule_smiles,
        attachment_free_query(fragment_smiles),
    )


def molecule_contains_query(molecule_smiles: str | None, query) -> bool:
    from rdkit import Chem

    if not molecule_smiles:
        return False
    try:
        molecule = Chem.MolFromSmiles(molecule_smiles, sanitize=True)
    except Exception:
        return False
    if molecule is None:
        return False
    return bool(molecule.HasSubstructMatch(query, useChirality=False))


def load_motif_records(path: str | Path) -> tuple[MotifRecord, ...]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"motif dataset not found: {source}")

    records: list[MotifRecord] = []
    seen_ids: set[str] = set()
    seen_smiles: set[str] = set()
    with source.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid motif JSON at {source}:{line_number}"
                ) from exc
            unknown = set(payload) - {"motif_id", "smiles", "frequency"}
            if unknown:
                raise ValueError(
                    f"unknown motif fields at {source}:{line_number}: "
                    f"{sorted(unknown)}"
                )
            if "motif_id" not in payload or "smiles" not in payload:
                raise ValueError(
                    f"motif_id and smiles are required at {source}:{line_number}"
                )
            motif_id = str(payload["motif_id"]).strip()
            if not motif_id:
                raise ValueError(f"empty motif_id at {source}:{line_number}")
            canonical = canonicalize_single_attachment_fragment(
                str(payload["smiles"])
            )
            frequency = payload.get("frequency")
            if frequency is not None:
                if isinstance(frequency, bool) or int(frequency) != frequency:
                    raise ValueError(
                        f"frequency must be an integer at {source}:{line_number}"
                    )
                frequency = int(frequency)
                if frequency < 0:
                    raise ValueError(
                        f"frequency must be non-negative at {source}:{line_number}"
                    )
            if motif_id in seen_ids:
                raise ValueError(f"duplicate motif_id in {source}: {motif_id!r}")
            if canonical in seen_smiles:
                raise ValueError(
                    f"duplicate canonical motif in {source}: {canonical!r}"
                )
            seen_ids.add(motif_id)
            seen_smiles.add(canonical)
            records.append(
                MotifRecord(
                    motif_id=motif_id,
                    smiles=canonical,
                    frequency=frequency,
                )
            )
    if not records:
        raise ValueError(f"motif dataset is empty: {source}")
    return tuple(records)


def load_test_motif_records(
    fragments_csv_path: str | Path,
) -> tuple[MotifRecord, ...]:
    source = Path(fragments_csv_path)
    if not source.is_file():
        raise FileNotFoundError(f"GenMol fragment test CSV not found: {source}")
    records = []
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"name", "motif_extension"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{source} must contain columns {sorted(required)}"
            )
        for row_number, row in enumerate(reader, start=2):
            motif_id = str(row["name"]).strip()
            raw_smiles = str(row["motif_extension"]).strip()
            if not motif_id or not raw_smiles:
                raise ValueError(
                    f"empty name or motif_extension at {source}:{row_number}"
                )
            records.append(
                MotifRecord(
                    motif_id=motif_id,
                    smiles=canonicalize_single_attachment_fragment(raw_smiles),
                )
            )
    if len(records) != 10:
        raise ValueError(
            f"expected exactly 10 official motif-extension test motifs, got {len(records)}"
        )
    if len({record.smiles for record in records}) != len(records):
        raise ValueError("official motif-extension test motifs are not unique")
    return tuple(records)


def serialize_motif_records(records: Sequence[MotifRecord]) -> str:
    if not records:
        raise ValueError("motif records must be non-empty")
    return json.dumps([asdict(record) for record in records], sort_keys=True)


def deserialize_motif_records(payload: str) -> tuple[MotifRecord, ...]:
    raw = json.loads(payload)
    if not isinstance(raw, list) or not raw:
        raise ValueError("serialized motif records must be a non-empty list")
    return tuple(MotifRecord(**item) for item in raw)


def sample_group_motif_records(
    records: Sequence[MotifRecord],
    *,
    num_groups: int,
    supergroup_num_groups: int,
    seed: int,
) -> list[MotifRecord]:
    if not records:
        raise ValueError("motif records must be non-empty")
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if supergroup_num_groups <= 0:
        raise ValueError("supergroup_num_groups must be positive")
    if num_groups % supergroup_num_groups != 0:
        raise ValueError(
            "num_groups must be divisible by supergroup_num_groups: "
            f"{num_groups} vs {supergroup_num_groups}"
        )

    rng = random.Random(seed)
    sampled = []
    for _ in range(num_groups // supergroup_num_groups):
        record = records[rng.randrange(len(records))]
        sampled.extend([record] * supergroup_num_groups)
    return sampled


def expand_group_motif_records(
    group_records: Sequence[MotifRecord],
    group_size: int,
) -> list[MotifRecord]:
    if not group_records:
        raise ValueError("group motif records must be non-empty")
    if group_size <= 1:
        raise ValueError("group_size must be greater than 1")
    expanded = []
    for record in group_records:
        expanded.extend([record] * group_size)
    return expanded
