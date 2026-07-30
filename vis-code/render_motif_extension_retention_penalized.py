#!/usr/bin/env python3
"""Render motif-extension metrics with non-retained utility set to -1."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


VIS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(VIS_ROOT))
import render_motif_extension_group_weight_comparisons as comparison  # noqa: E402
import render_motif_extension_results as motif  # noqa: E402
import render_rebuttal_dense_results as dense  # noqa: E402


NON_RETAINED_UTILITY = -1.0


class RetentionPenalizedStore(dense.ResultStore):
    def __init__(
        self,
        base_store: motif.MotifResultStore,
        variant_store: motif.MotifResultStore,
        variant_model: dense.ModelSpec,
    ):
        self._rows_by_seed = {
            seed: [
                *self._penalized_rows(
                    base_store,
                    comparison.BASE_MODELS,
                    seed,
                ),
                *self._penalized_rows(
                    variant_store,
                    (variant_model,),
                    seed,
                ),
            ]
            for seed in dense.SEEDS
        }

    @staticmethod
    def _penalized_rows(
        source_store: motif.MotifResultStore,
        models: tuple[dense.ModelSpec, ...],
        seed: int,
    ) -> list[dict]:
        conditioned_rows = {
            (row["experiment"], int(row["point_index"])): row
            for row in source_store.rows("denovo", seed)
            if row["experiment"] in {
                model.source_id for model in models
            }
        }
        output = []
        for model in models:
            rows_path = (
                source_store.results_root
                / "results"
                / model.source_id
                / f"seed{seed}"
                / "rows.jsonl"
            )
            raw_rows = [
                json.loads(line)
                for line in rows_path.read_text().splitlines()
            ]
            if len(raw_rows) != 10_000:
                raise ValueError(f"expected 10,000 rows in {rows_path}")
            rows_by_point: dict[int, list[dict]] = {}
            for row in raw_rows:
                if row.get("experiment") != model.source_id:
                    raise ValueError(
                        f"experiment mismatch in {rows_path}"
                    )
                if int(row.get("seed")) != seed:
                    raise ValueError(f"seed mismatch in {rows_path}")
                rows_by_point.setdefault(
                    int(row["point_index"]),
                    [],
                ).append(row)
            if set(rows_by_point) != set(range(10)):
                raise ValueError(
                    f"incomplete sweep coverage in {rows_path}"
                )

            for point_index, (randomness, temperature) in enumerate(
                dense.MOLECULE_SWEEP
            ):
                point_rows = rows_by_point[point_index]
                if len(point_rows) != 1_000:
                    raise ValueError(
                        f"expected 1,000 rows for {model.source_id}, "
                        f"seed={seed}, point={point_index}"
                    )
                utilities = []
                for row in point_rows:
                    if not row["motif_retained"]:
                        utilities.append(NON_RETAINED_UTILITY)
                        continue
                    soft_reward = row.get("soft_reward")
                    if soft_reward is None or not math.isfinite(
                        float(soft_reward)
                    ):
                        raise ValueError(
                            "retained molecule has no finite soft reward "
                            f"in {rows_path}"
                        )
                    utilities.append(float(soft_reward))

                conditioned = conditioned_rows[
                    (model.source_id, point_index)
                ]
                if dense._molecule_key(
                    conditioned["randomness"],
                    conditioned["generation_temperature"],
                ) != dense._molecule_key(randomness, temperature):
                    raise ValueError(
                        f"sweep coordinate mismatch for "
                        f"{model.source_id}, seed={seed}, "
                        f"point={point_index}"
                    )
                output.append(
                    {
                        **conditioned,
                        "soft_reward_mean": (
                            sum(utilities) / len(utilities)
                        ),
                        "utility_aggregation": (
                            "all_attempts_non_retained_minus_one"
                        ),
                        "non_retained_utility": NON_RETAINED_UTILITY,
                    }
                )
        return output

    def rows(self, kind: str, seed: int) -> list[dict]:
        if kind != "denovo":
            raise ValueError(f"unsupported result kind: {kind}")
        return self._rows_by_seed[seed]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run-root", type=Path, required=True)
    parser.add_argument("--variant-run-root", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=tuple(comparison.VARIANT_MODELS),
        required=True,
    )
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    variant_model = comparison.VARIANT_MODELS[args.variant]
    base_store = motif.MotifResultStore(
        args.base_run_root,
        panel=comparison.BASE_PANEL,
    )
    base_store.validate_raw_seed_independence()
    variant_panel = comparison._single_model_panel(
        args.variant,
        variant_model,
    )
    variant_store = motif.MotifResultStore(
        args.variant_run_root,
        panel=variant_panel,
    )
    variant_store.validate_raw_seed_independence()

    store = RetentionPenalizedStore(
        base_store,
        variant_store,
        variant_model,
    )
    panel = comparison._comparison_panel(args.variant, variant_model)
    dense.validate_seed_independence(
        store,
        (panel,),
        include_denovo_auxiliary=False,
    )
    metrics, references = dense.table1_metrics(store, (panel,))
    section = comparison._format_section(panel, metrics, references)
    output = (
        "# Motif-Extension Retention-Penalized Metrics\n\n"
        "Each motif-retained molecule keeps its standard soft Utility; "
        "each non-retained generation has Utility = -1. Diversity retains "
        "the standard conditional definition.\n\n"
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
