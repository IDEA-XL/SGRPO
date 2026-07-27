#!/usr/bin/env python3
"""Evaluate one motif-extension model over the fixed paired sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

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


def _finite_mean(values, *, label):
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not finite:
        raise RuntimeError(f"no finite values available for {label}")
    return sum(finite) / len(finite)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_spec(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"evaluation spec not found: {path}")
    payload = json.loads(path.read_text())
    required = {
        "experiment",
        "display_name",
        "checkpoint_path",
        "seed",
        "output_dir",
    }
    missing = required - set(payload)
    unknown = set(payload) - required
    if missing or unknown:
        raise ValueError(
            f"invalid evaluation spec fields; missing={sorted(missing)} "
            f"unknown={sorted(unknown)}"
        )
    if isinstance(payload["seed"], bool) or int(payload["seed"]) != payload["seed"]:
        raise ValueError("seed must be an integer")
    payload["seed"] = int(payload["seed"])
    return payload


def evaluate(spec: dict) -> None:
    import torch

    from genmol.diversity import (
        MORGAN_INTERNAL_DIVERSITY,
        compute_molecular_diversity,
    )
    from genmol.rl.motif import (
        attachment_free_query,
        load_test_motif_records,
        molecule_contains_query,
    )
    from genmol.rl.policy import GenMolCpGRPOPolicy
    from genmol.rl.reward import MolecularReward
    from genmol.rl.specs import sample_conditioned_group_specs

    checkpoint_path = Path(spec["checkpoint_path"])
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"checkpoint not found or empty: {checkpoint_path}"
        )
    output_dir = Path(spec["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "rows.jsonl"
    summary_path = output_dir / "summary.json"
    rows_temp_path = output_dir / "rows.jsonl.tmp"
    summary_temp_path = output_dir / "summary.json.tmp"

    motifs = load_test_motif_records(REPO_ROOT / "data/fragments.csv")
    queries = {
        motif.smiles: attachment_free_query(motif.smiles)
        for motif in motifs
    }
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("motif-extension sweep requires CUDA")
    policy = GenMolCpGRPOPolicy(
        checkpoint_path=str(checkpoint_path),
        device=device,
        bf16=True,
        trainable=False,
    )
    reward_model = MolecularReward(always_compute_metrics=True)
    samples_per_motif = 100
    generation_batch_size = 1000
    min_add_len = 18
    gamma = 0.3
    guidance_weight = 2.0
    results = []
    total_row_count = 0

    try:
        with rows_temp_path.open("x") as rows_handle:
            for point_index, (randomness, temperature) in enumerate(
                MOLECULE_SWEEP
            ):
                expanded_motifs = [
                    motif
                    for motif in motifs
                    for _ in range(samples_per_motif)
                ]
                base_lengths = [
                    policy.motif_base_sequence_length(motif.smiles)
                    for motif in expanded_motifs
                ]
                # Reuse the seed across paired sweep points, matching the
                # existing GenMol sweep's common-random-number evaluation.
                point_seed = spec["seed"]
                specs = sample_conditioned_group_specs(
                    base_sequence_lengths=base_lengths,
                    generation_temperature=temperature,
                    randomness=randomness,
                    min_add_len=min_add_len,
                    seed=point_seed,
                )
                rollout = policy.rollout_motif_extension(
                    specs=specs,
                    fragments=[motif.smiles for motif in expanded_motifs],
                    generation_batch_size=generation_batch_size,
                    seed=point_seed + 1,
                    gamma=gamma,
                    guidance_weight=guidance_weight,
                )
                retained_flags = [
                    molecule_contains_query(
                        smiles,
                        queries[motif.smiles],
                    )
                    for smiles, motif in zip(
                        rollout.smiles,
                        expanded_motifs,
                    )
                ]
                task_smiles = [
                    smiles if retained else None
                    for smiles, retained in zip(
                        rollout.smiles,
                        retained_flags,
                    )
                ]
                records = reward_model.score(task_smiles)
                expected_count = len(motifs) * samples_per_motif
                if len(records) != expected_count:
                    raise RuntimeError(
                        f"reward record count mismatch: {len(records)} vs "
                        f"{expected_count}"
                    )

                per_motif_diversities = []
                per_motif_retention = []
                for motif_index, motif in enumerate(motifs):
                    start = motif_index * samples_per_motif
                    end = start + samples_per_motif
                    motif_records = records[start:end]
                    motif_retained = retained_flags[start:end]
                    per_motif_diversities.append(
                        compute_molecular_diversity(
                            [record.smiles for record in motif_records],
                            metric=MORGAN_INTERNAL_DIVERSITY,
                        )
                    )
                    per_motif_retention.append(
                        sum(motif_retained) / samples_per_motif
                    )
                    for sample_index, (
                        raw_smiles,
                        retained,
                        record,
                    ) in enumerate(
                        zip(
                            rollout.smiles[start:end],
                            motif_retained,
                            motif_records,
                        )
                    ):
                        row = {
                            "experiment": spec["experiment"],
                            "display_name": spec["display_name"],
                            "checkpoint_path": str(checkpoint_path),
                            "seed": spec["seed"],
                            "point_index": point_index,
                            "randomness": randomness,
                            "generation_temperature": temperature,
                            "motif_index": motif_index,
                            "motif_id": motif.motif_id,
                            "motif_smiles": motif.smiles,
                            "sample_index": sample_index,
                            "raw_smiles": raw_smiles,
                            "motif_retained": bool(retained),
                            "smiles": record.smiles,
                            "reward": float(record.reward),
                            "is_valid": bool(record.is_valid),
                            "alert_hit": bool(record.alert_hit),
                            "qed": record.qed,
                            "sa": record.sa,
                            "sa_score": record.sa_score,
                            "soft_reward": record.soft_reward,
                        }
                        rows_handle.write(
                            json.dumps(row, sort_keys=True) + "\n"
                        )
                        total_row_count += 1

                raw_valid_flags = [
                    smiles is not None for smiles in rollout.smiles
                ]
                results.append(
                    {
                        "experiment": spec["experiment"],
                        "display_name": spec["display_name"],
                        "checkpoint_path": str(checkpoint_path),
                        "seed": spec["seed"],
                        "point_index": point_index,
                        "sweep_axis": "randomness_temperature_pair",
                        "sweep_value": float(point_index + 1),
                        "sweep_label": (
                            f"r={randomness:.1f},t={temperature:g}"
                        ),
                        "randomness": randomness,
                        "generation_temperature": temperature,
                        "num_motifs": len(motifs),
                        "samples_per_motif": samples_per_motif,
                        "num_samples": expected_count,
                        "soft_reward_mean": _finite_mean(
                            [record.soft_reward for record in records],
                            label="soft reward",
                        ),
                        "qed_mean": _finite_mean(
                            [record.qed for record in records],
                            label="QED",
                        ),
                        "sa_mean": _finite_mean(
                            [record.sa for record in records],
                            label="SA",
                        ),
                        "sa_score_mean": _finite_mean(
                            [record.sa_score for record in records],
                            label="SA score",
                        ),
                        "diversity": float(
                            sum(per_motif_diversities)
                            / len(per_motif_diversities)
                        ),
                        "per_motif_diversity": per_motif_diversities,
                        "raw_valid_fraction": float(
                            sum(raw_valid_flags) / expected_count
                        ),
                        "motif_retention_fraction": float(
                            sum(retained_flags) / expected_count
                        ),
                        "per_motif_retention_fraction": per_motif_retention,
                        "task_valid_fraction": float(
                            sum(record.is_valid for record in records)
                            / expected_count
                        ),
                        "alert_hit_fraction": float(
                            sum(record.alert_hit for record in records)
                            / expected_count
                        ),
                        "diversity_metric": MORGAN_INTERNAL_DIVERSITY,
                    }
                )
    finally:
        reward_model.close()

    expected_total_rows = (
        len(MOLECULE_SWEEP) * len(motifs) * samples_per_motif
    )
    if total_row_count != expected_total_rows:
        raise RuntimeError(
            f"row count mismatch: {total_row_count} vs {expected_total_rows}"
        )
    os.replace(rows_temp_path, rows_path)
    summary = {
        "metadata": {
            "experiment": spec["experiment"],
            "display_name": spec["display_name"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "seed": spec["seed"],
            "motif_count": len(motifs),
            "samples_per_motif": samples_per_motif,
            "sweep_points": [list(point) for point in MOLECULE_SWEEP],
            "min_add_len": min_add_len,
            "gamma": gamma,
            "guidance_weight": guidance_weight,
            "generation_batch_size": generation_batch_size,
            "row_count": total_row_count,
            "rows_sha256": _sha256(rows_path),
        },
        "results": results,
    }
    with summary_temp_path.open("x") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(summary_temp_path, summary_path)
    print(summary_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    evaluate(_load_spec(args.spec))


if __name__ == "__main__":
    main()
