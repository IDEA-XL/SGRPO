#!/usr/bin/env python3
"""Render five-run de novo scaffold-diversity Table 1 and Figure 2 results."""

from __future__ import annotations

import argparse
from pathlib import Path

import render_rebuttal_dense_results as dense
import matplotlib.pyplot as plt


SCAFFOLD_DIVERSITY_METRIC = "relative_scaffold_diversity"
COLOR_DMB = "#6B5B95"
COLOR_ENTROPY = "#C85A3E"
SECTION_BEGIN = "<!-- BEGIN DENOVO SCAFFOLD DIVERSITY RESULTS -->"
SECTION_END = "<!-- END DENOVO SCAFFOLD DIVERSITY RESULTS -->"

PANEL = dense.PanelSpec(
    key="denovo",
    title="De Novo Molecule Design",
    source_kind="denovo",
    models=(
        dense.ModelSpec("scaffold_original_genmol_v2", "Original", dense.COLOR_ORIGINAL, "D"),
        dense.ModelSpec("scaffold_grpo_2000", "GRPO", dense.COLOR_GRPO, "^"),
        dense.ModelSpec(
            "scaffold_hbd_2000",
            "Memory-Assisted GRPO",
            dense.COLOR_MEMORY,
            "s",
        ),
        dense.ModelSpec(
            "scaffold_dmb_2000",
            "Diverse Mini-Batch GRPO",
            COLOR_DMB,
            "P",
        ),
        dense.ModelSpec(
            "scaffold_entropy_2000",
            "Entropy-Regularized GRPO",
            COLOR_ENTROPY,
            "X",
        ),
        dense.ModelSpec("scaffold_sgrpo_2000", "SGRPO", dense.COLOR_SGRPO, "o"),
    ),
)


class ScaffoldResultStore(dense.ResultStore):
    def _path(self, kind: str, seed: int) -> Path:
        if kind != "denovo":
            raise ValueError(f"Scaffold result store only supports denovo, got {kind}")
        return (
            self.results_root
            / "denovo"
            / f"seed{seed}"
            / "aggregate"
            / "denovo_dense.json"
        )

    def rows(self, kind: str, seed: int) -> list[dict]:
        rows = super().rows(kind, seed)
        relevant = {
            model.source_id for model in PANEL.models
        }
        mismatched = [
            row
            for row in rows
            if row.get("experiment") in relevant
            and row.get("diversity_metric") != SCAFFOLD_DIVERSITY_METRIC
        ]
        if mismatched:
            raise ValueError(
                f"Found non-scaffold rows for seed {seed}: "
                f"{sorted({row.get('diversity_metric') for row in mismatched})}"
            )
        return rows


def _plot_figure2(store: ScaffoldResultStore, output_path: Path) -> None:
    dense.configure_style(20)
    figure, axis = plt.subplots(figsize=(10.8, 7.2))
    figure.patch.set_facecolor("white")
    dense._draw_main_panel(axis, store, PANEL)
    figure.supxlabel("Utility", y=0.22)
    figure.supylabel("Scaffold Diversity", x=0.035)
    figure.legend(
        handles=dense._main_legend_handles((PANEL,)),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.15,
        handletextpad=0.5,
    )
    figure.subplots_adjust(left=0.13, right=0.99, top=0.91, bottom=0.34)
    figure.savefig(output_path)
    plt.close(figure)


def _result_section(
    store: ScaffoldResultStore,
    metrics: dict[str, dict[str, tuple[float, float]]],
    references: dict[str, list[dense.Point]],
    figure_name: str,
) -> str:
    molecule_points = ", ".join(
        f"({randomness:g}, {temperature:g})"
        for randomness, temperature in dense.MOLECULE_SWEEP
    )
    lines = [
        "## GenMol De Novo: Scaffold Diversity",
        "",
        "All six models are evaluated with relative Bemis-Murcko scaffold diversity. "
        "Each operating point reports the mean and 95% confidence interval over "
        "five independent generation runs.",
        "",
        "| Sweep points | Runs | Samples per model and point |",
        "|---|---:|---:|",
        f"| `{molecule_points}` | 5 | 1,000 |",
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
            f"[PDF]({figure_name})",
            "",
            "| Model | Sweep point | Utility mean | Utility 95% CI | "
            "Scaffold diversity mean | Scaffold diversity 95% CI |",
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
            f"Malformed scaffold result markers in {path}: "
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
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expanded-results-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    store = ScaffoldResultStore(args.run_root, ("denovo",))
    dense.validate_seed_independence(
        store,
        (PANEL,),
        include_denovo_auxiliary=False,
    )
    metrics, references = dense.table1_metrics(store, (PANEL,))

    figure_path = output_dir / "figure2-scaffold-diversity.pdf"
    standalone_path = output_dir / "scaffold-diversity-results.md"
    _plot_figure2(store, figure_path)
    section = _result_section(store, metrics, references, figure_path.name)
    standalone_path.write_text("# Scaffold Diversity Results\n\n" + section)
    _upsert_section(args.expanded_results_path, section)
    print(figure_path)
    print(standalone_path)
    print(args.expanded_results_path)


if __name__ == "__main__":
    main()
