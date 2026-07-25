#!/usr/bin/env python3
"""Render Table 1 and Figure 2 tables with the added GRPO baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import render_rebuttal_dense_results as dense


PROGEN2_TEMPERATURES = tuple(round(index / 10.0, 1) for index in range(1, 13))
COLOR_DMB = "#6B5B95"
COLOR_ENTROPY = "#C85A3E"

PANELS = (
    dense.PanelSpec(
        key="denovo",
        title="De Novo Molecule Design",
        source_kind="denovo",
        models=(
            dense.ModelSpec("original_genmol_v2", "Original", dense.COLOR_ORIGINAL, "D"),
            dense.ModelSpec("genmol_denovo_grpo_2000", "GRPO", dense.COLOR_GRPO, "^"),
            dense.ModelSpec(
                "genmol_denovo_grpo_hbd_2000",
                "Memory-Assisted GRPO",
                dense.COLOR_MEMORY,
                "s",
            ),
            dense.ModelSpec(
                "denovo_dmb_2000",
                "Diverse Mini-Batch GRPO",
                COLOR_DMB,
                "P",
            ),
            dense.ModelSpec(
                "denovo_entropy_2000",
                "Entropy-Regularized GRPO",
                COLOR_ENTROPY,
                "X",
            ),
            dense.ModelSpec(
                "genmol_denovo_sgrpo_rewardsum_loo_2000",
                "SGRPO",
                dense.COLOR_SGRPO,
                "o",
            ),
        ),
    ),
    dense.PanelSpec(
        key="mmgenmol",
        title="Pocket-Based Design",
        source_kind="mmgenmol",
        models=(
            dense.ModelSpec("original_5500", "Original", dense.COLOR_ORIGINAL, "D"),
            dense.ModelSpec("grpo_unidock_1000", "GRPO", dense.COLOR_GRPO, "^"),
            dense.ModelSpec(
                "mmgenmol_dmb_1000",
                "Diverse Mini-Batch GRPO",
                COLOR_DMB,
                "P",
            ),
            dense.ModelSpec(
                "mmgenmol_entropy_1000",
                "Entropy-Regularized GRPO",
                COLOR_ENTROPY,
                "X",
            ),
            dense.ModelSpec(
                "sgrpo_unidock_rewardsum_loo_1000",
                "SGRPO",
                dense.COLOR_SGRPO,
                "o",
            ),
        ),
    ),
    dense.PanelSpec(
        key="progen2",
        title="De Novo Protein Design",
        source_kind="progen2",
        models=(
            dense.ModelSpec("original", "Original", dense.COLOR_ORIGINAL, "D"),
            dense.ModelSpec("grpo_step100", "GRPO", dense.COLOR_GRPO, "^"),
            dense.ModelSpec(
                "grpo_hbd_step100",
                "Memory-Assisted GRPO",
                dense.COLOR_MEMORY,
                "s",
            ),
            dense.ModelSpec(
                "dmb_grpo_step100",
                "Diverse Mini-Batch GRPO",
                COLOR_DMB,
                "P",
            ),
            dense.ModelSpec(
                "entropy_grpo_step100",
                "Entropy-Regularized GRPO",
                COLOR_ENTROPY,
                "X",
            ),
            dense.ModelSpec(
                "sgrpo_gw08_rewardsum_loo_step100",
                "SGRPO",
                dense.COLOR_SGRPO,
                "o",
            ),
        ),
    ),
)


class MergedResultStore(dense.ResultStore):
    def __init__(self, base_root: Path, expansion_root: Path):
        self.results_root = base_root
        self.expansion_root = expansion_root
        self._cache: dict[tuple[str, int], list[dict]] = {}
        missing = [
            path
            for root in (base_root, expansion_root)
            for kind in ("denovo", "mmgenmol", "progen2")
            for seed in dense.SEEDS
            if not (path := root / kind / f"seed{seed}.json").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing sweep summaries:\n" + "\n".join(str(path) for path in missing)
            )

    @staticmethod
    def _decode_rows(path: Path, kind: str) -> list[dict]:
        value = json.loads(path.read_text())
        if kind == "progen2":
            if not isinstance(value, dict) or not isinstance(value.get("results"), list):
                raise TypeError(f"Expected dict with results list in {path}")
            return value["results"]
        if not isinstance(value, list):
            raise TypeError(f"Expected list in {path}")
        return value

    def rows(self, kind: str, seed: int) -> list[dict]:
        cache_key = (kind, seed)
        if cache_key not in self._cache:
            base_path = self.results_root / kind / f"seed{seed}.json"
            expansion_path = self.expansion_root / kind / f"seed{seed}.json"
            self._cache[cache_key] = [
                *self._decode_rows(base_path, kind),
                *self._decode_rows(expansion_path, kind),
            ]
        return self._cache[cache_key]


def _append_table1(
    lines: list[str],
    metrics: dict[str, dict[str, tuple[float, float]]],
) -> None:
    lines.extend(
        [
            "## Table 1: Frontier Metrics",
            "",
            "Each cell is the mean ± 95% confidence interval over five independent "
            "sweep runs. The HV reference point is recomputed per run from all methods "
            "listed in the corresponding task panel.",
            "",
        ]
    )
    for panel in PANELS:
        lines.extend(
            [
                f"### {panel.title}",
                "",
                "| Metric | " + " | ".join(model.label for model in panel.models) + " |",
                "|---|" + "|".join("---:" for _ in panel.models) + "|",
            ]
        )
        for metric, direction in (("HV", "↑"), ("DIP", "↓"), ("R2", "↓")):
            values = [
                dense._format_interval(
                    *metrics[panel.key][f"{model.label}:{metric}"]
                )
                for model in panel.models
            ]
            lines.append(f"| {metric} {direction} | " + " | ".join(values) + " |")
        lines.append("")


def _append_reference_points(
    lines: list[str],
    references: dict[str, list[dense.Point]],
) -> None:
    lines.extend(
        [
            "### Per-Run HV Reference Points",
            "",
            "| Task | Seed | Utility reference | Diversity reference |",
            "|---|---:|---:|---:|",
        ]
    )
    for panel in PANELS:
        for seed, point in zip(dense.SEEDS, references[panel.key]):
            lines.append(
                f"| {panel.title} | {seed} | {point.utility:.4f} | "
                f"{point.diversity:.4f} |"
            )
    lines.append("")


def _append_figure2(lines: list[str], store: MergedResultStore) -> None:
    lines.extend(
        [
            "## Figure 2: Utility-Diversity Operating Points",
            "",
            "Each row reports the five-run mean and 95% confidence interval for one "
            "decoding point.",
            "",
        ]
    )
    for panel in PANELS:
        lines.extend(
            [
                f"### {panel.title}",
                "",
                "| Model | Sweep point | Utility mean | Utility 95% CI | "
                "Diversity mean | Diversity 95% CI |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for model in panel.models:
            for point in dense.aggregate_series(store, panel, model):
                lines.append(
                    f"| {model.label} | {point.point_label} | "
                    f"{point.utility_mean:.4f} | {point.utility_ci95:.4f} | "
                    f"{point.diversity_mean:.4f} | {point.diversity_ci95:.4f} |"
                )
        lines.append("")


def _write_markdown(
    output_path: Path,
    store: MergedResultStore,
    metrics: dict[str, dict[str, tuple[float, float]]],
    references: dict[str, list[dense.Point]],
) -> None:
    molecule_points = ", ".join(
        f"({randomness:g}, {temperature:g})"
        for randomness, temperature in dense.MOLECULE_SWEEP
    )
    protein_points = ", ".join(f"{temperature:g}" for temperature in PROGEN2_TEMPERATURES)
    lines = [
        "# Expanded Sweeping Evaluation Results",
        "",
        "These results use five independent sweep runs. No paper or Overleaf source "
        "is modified.",
        "",
        "## Evaluation Protocol",
        "",
        "| Task family | Sweep points | Runs | Samples per model and point |",
        "|---|---|---:|---:|",
        f"| GenMol de novo | `{molecule_points}` | 5 | 1,000 |",
        f"| GenMol-P / mmGenMol | `{molecule_points}` | 5 | 1,600 |",
        f"| ProGen2 | `{protein_points}` | 5 | 512 |",
        "",
    ]
    _append_table1(lines, metrics)
    _append_reference_points(lines, references)
    _append_figure2(lines, store)
    dense._append_figure3_table(lines, store)
    group_result = dense._hyperparameter_hv(store, dense.GROUP_SIZE_SPECS)
    weight_result = dense._hyperparameter_hv(store, dense.DIVERSITY_WEIGHT_SPECS)
    dense._append_figure5_tables(lines, group_result, weight_result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    rebuttal_root = project_root / "nips26/rebuttal"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-results-root",
        type=Path,
        default=rebuttal_root / "sweep-results",
    )
    parser.add_argument(
        "--expansion-results-root",
        type=Path,
        default=rebuttal_root / "baseline-expansion-results",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=rebuttal_root / "expanded-sweep-results.md",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dense.PROGEN2_TEMPERATURES = PROGEN2_TEMPERATURES
    store = MergedResultStore(args.base_results_root, args.expansion_results_root)
    dense.validate_seed_independence(store, PANELS)
    metrics, references = dense.table1_metrics(store, PANELS)
    _write_markdown(args.output_path, store, metrics, references)
    print(args.output_path)


if __name__ == "__main__":
    main()
