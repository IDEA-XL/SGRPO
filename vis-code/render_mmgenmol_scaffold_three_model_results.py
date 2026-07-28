#!/usr/bin/env python3
"""Render three-model mmGenMol scaffold-diversity Table 1 and Figure 2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

import render_rebuttal_dense_results as dense


SCAFFOLD_DIVERSITY_METRIC = "relative_scaffold_diversity"
SECTION_BEGIN = "<!-- BEGIN MMGENMOL THREE-MODEL SCAFFOLD REANALYSIS RESULTS -->"
SECTION_END = "<!-- END MMGENMOL THREE-MODEL SCAFFOLD REANALYSIS RESULTS -->"

PANEL = dense.PanelSpec(
    key="mmgenmol",
    title="Pocket-Based Design",
    source_kind="mmgenmol",
    models=(
        dense.ModelSpec("original_5500", "Original", dense.COLOR_ORIGINAL, "D"),
        dense.ModelSpec("grpo_unidock_1000", "GRPO", dense.COLOR_GRPO, "^"),
        dense.ModelSpec(
            "sgrpo_unidock_rewardsum_loo_1000",
            "SGRPO",
            dense.COLOR_SGRPO,
            "o",
        ),
    ),
)


class ScaffoldResultStore(dense.ResultStore):
    def rows(self, kind: str, seed: int) -> list[dict]:
        rows = super().rows(kind, seed)
        relevant = {model.source_id for model in PANEL.models}
        selected = [row for row in rows if row.get("model_name") in relevant]
        expected = len(PANEL.models) * len(dense.MOLECULE_SWEEP)
        if len(selected) != expected:
            raise ValueError(
                f"Seed {seed} has {len(selected)} three-model scaffold rows; "
                f"expected {expected}"
            )
        mismatched = [
            row
            for row in selected
            if row.get("diversity_metric") != SCAFFOLD_DIVERSITY_METRIC
        ]
        if mismatched:
            raise ValueError(
                f"Found non-scaffold rows for seed {seed}: "
                f"{sorted({row.get('diversity_metric') for row in mismatched})}"
            )
        return rows


def _validate_manifest(results_root: Path) -> dict:
    path = results_root / "manifest.json"
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing reanalysis manifest: {path}")
    manifest = json.loads(path.read_text())
    expected_models = [model.source_id for model in PANEL.models]
    checks = {
        "profile": "mmgenmol_scaffold_three_model_reanalysis",
        "diversity_metric": SCAFFOLD_DIVERSITY_METRIC,
        "seeds": list(dense.SEEDS),
        "sweep_points": [list(point) for point in dense.MOLECULE_SWEEP],
        "models": expected_models,
        "num_pockets": 100,
        "samples_per_pocket": 16,
        "samples_per_model_point": 1600,
        "total_generated_rows": 240_000,
        "total_docking_rows": 240_000,
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Manifest field {key!r} is {manifest.get(key)!r}; "
                f"expected {expected!r}"
            )
    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != 150:
        raise ValueError("Manifest does not contain all 150 model-point sources")
    if any(
        int(record.get("generated_rows", -1)) != 1600
        or int(record.get("docking_rows", -1)) != 1600
        or int(record.get("pockets", -1)) != 100
        or len(str(record.get("generated_sha256", ""))) != 64
        or len(str(record.get("docking_sha256", ""))) != 64
        for record in sources
    ):
        raise ValueError("Manifest contains an invalid source-file audit record")
    return manifest


def _plot_figure2(store: ScaffoldResultStore, output_path: Path) -> None:
    dense.configure_style(20)
    figure, axis = plt.subplots(figsize=(10.8, 7.4))
    figure.patch.set_facecolor("white")
    dense._draw_main_panel(axis, store, PANEL)
    figure.supxlabel("Utility", y=0.22)
    figure.supylabel("Mean Per-Pocket Scaffold Diversity", x=0.025)
    figure.legend(
        handles=dense._main_legend_handles((PANEL,)),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.15,
        handletextpad=0.5,
    )
    figure.subplots_adjust(left=0.15, right=0.99, top=0.90, bottom=0.36)
    figure.savefig(output_path)
    plt.close(figure)


def _result_section(
    store: ScaffoldResultStore,
    metrics: dict[str, dict[str, tuple[float, float]]],
    references: dict[str, list[dense.Point]],
    figure_link: str,
) -> str:
    molecule_points = ", ".join(
        f"({randomness:g}, {temperature:g})"
        for randomness, temperature in dense.MOLECULE_SWEEP
    )
    lines = [
        "## GenMol-P/mmGenMol: Three-Model Scaffold-Diversity Reanalysis",
        "",
        "This is an additional evaluation result and does not replace any primary, "
        "100-ligand, or previously reported scaffold experiment. The exact saved "
        "per-sample generations and Utility values from the primary five-run sweep "
        "are reused. Diversity is recomputed within each pocket as the number of "
        "unique canonical Bemis-Murcko scaffolds divided by the number of final-valid "
        "molecules, then averaged over the 100 pockets.",
        "",
        "| Sweep points | Runs | Pockets | Samples per pocket | "
        "Samples per model and point | Total reused samples |",
        "|---|---:|---:|---:|---:|---:|",
        f"| `{molecule_points}` | 5 | 100 | 16 | 1,600 | 240,000 |",
        "",
        "### Table 1: Frontier Metrics",
        "",
        "| Metric | " + " | ".join(model.label for model in PANEL.models) + " |",
        "|---|" + "|".join("---:" for _ in PANEL.models) + "|",
    ]
    for metric, direction in (("HV", "↑"), ("DIP", "↓"), ("R2", "↓")):
        values = [
            dense._format_interval(*metrics[PANEL.key][f"{model.label}:{metric}"])
            for model in PANEL.models
        ]
        lines.append(f"| {metric} {direction} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "### Per-Run HV Reference Points",
            "",
            "| Seed | Utility reference | Scaffold-diversity reference |",
            "|---:|---:|---:|",
        ]
    )
    for seed, point in zip(dense.SEEDS, references[PANEL.key]):
        lines.append(f"| {seed} | {point.utility:.4f} | {point.diversity:.4f} |")

    lines.extend(
        [
            "",
            "### Figure 2: Utility-Scaffold-Diversity Operating Points",
            "",
            f"[PDF]({figure_link})",
            "",
            "| Model | Sweep point | Utility mean | Utility 95% CI | "
            "Mean per-pocket scaffold diversity | Scaffold-diversity 95% CI |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for model in PANEL.models:
        for point in dense.aggregate_series(store, PANEL, model):
            lines.append(
                f"| {model.label} | {point.point_label} | "
                f"{point.utility_mean:.4f} | {point.utility_ci95:.4f} | "
                f"{point.diversity_mean:.4f} | {point.diversity_ci95:.4f} |"
            )
    return "\n".join(lines) + "\n"


def _upsert_section(path: Path, section: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Expanded results Markdown does not exist: {path}")
    text = path.read_text()
    begin_count = text.count(SECTION_BEGIN)
    end_count = text.count(SECTION_END)
    if begin_count != end_count or begin_count > 1:
        raise RuntimeError(
            f"Malformed mmGenMol scaffold markers in {path}: "
            f"begin={begin_count}, end={end_count}"
        )
    wrapped = f"{SECTION_BEGIN}\n{section.rstrip()}\n{SECTION_END}\n"
    if begin_count == 1:
        prefix, remainder = text.split(SECTION_BEGIN, 1)
        _, suffix = remainder.split(SECTION_END, 1)
        updated = prefix.rstrip() + "\n\n" + wrapped + suffix.lstrip()
    else:
        updated = text.rstrip() + "\n\n" + wrapped
    path.write_text(updated)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expanded-results-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _validate_manifest(args.results_root)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    store = ScaffoldResultStore(args.results_root, ("mmgenmol",))
    dense.validate_seed_independence(
        store,
        (PANEL,),
        include_denovo_auxiliary=False,
    )
    metrics, references = dense.table1_metrics(store, (PANEL,))

    figure_path = output_dir / "figure2-mmgenmol-scaffold-three-model.pdf"
    standalone_path = output_dir / "mmgenmol-scaffold-three-model-results.md"
    figure_link = os.path.relpath(
        figure_path,
        start=args.expanded_results_path.parent,
    ).replace(os.sep, "/")
    _plot_figure2(store, figure_path)
    section = _result_section(store, metrics, references, figure_link)
    standalone_path.write_text(
        "# mmGenMol Three-Model Scaffold-Diversity Reanalysis\n\n" + section
    )
    _upsert_section(args.expanded_results_path, section)
    print(figure_path)
    print(standalone_path)
    print(args.expanded_results_path)


if __name__ == "__main__":
    main()
