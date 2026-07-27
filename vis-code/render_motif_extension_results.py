#!/usr/bin/env python3
"""Validate and render motif-extension Table 1 and Figure 2 results."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VIS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(VIS_ROOT))
import render_rebuttal_dense_results as dense  # noqa: E402


SECTION_BEGIN = "<!-- BEGIN MOTIF EXTENSION RESULTS -->"
SECTION_END = "<!-- END MOTIF EXTENSION RESULTS -->"
COLOR_DMB = "#6B5B95"
COLOR_ENTROPY = "#C85A3E"
PANEL = dense.PanelSpec(
    key="motif_extension",
    title="Motif Extension",
    source_kind="denovo",
    models=(
        dense.ModelSpec(
            "motif_original_genmol_v2",
            "Original",
            dense.COLOR_ORIGINAL,
            "D",
        ),
        dense.ModelSpec("motif_grpo_2000", "GRPO", dense.COLOR_GRPO, "^"),
        dense.ModelSpec(
            "motif_dmb_2000",
            "Diverse Mini-Batch GRPO",
            COLOR_DMB,
            "P",
        ),
        dense.ModelSpec(
            "motif_entropy_2000",
            "Entropy-Regularized GRPO",
            COLOR_ENTROPY,
            "X",
        ),
        dense.ModelSpec("motif_sgrpo_2000", "SGRPO", dense.COLOR_SGRPO, "o"),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class MotifResultStore(dense.ResultStore):
    def __init__(self, run_root: Path):
        self.results_root = run_root
        self._cache = {}
        self._row_hashes = {}

    def rows(self, kind: str, seed: int) -> list[dict]:
        if kind != "denovo":
            raise ValueError(f"unsupported motif result kind: {kind}")
        if seed in self._cache:
            return self._cache[seed]
        rows = []
        for model in PANEL.models:
            result_dir = (
                self.results_root
                / "results"
                / model.source_id
                / f"seed{seed}"
            )
            summary_path = result_dir / "summary.json"
            rows_path = result_dir / "rows.jsonl"
            if not summary_path.is_file() or not rows_path.is_file():
                raise FileNotFoundError(
                    f"missing motif result files in {result_dir}"
                )
            summary = json.loads(summary_path.read_text())
            metadata = summary.get("metadata")
            result_rows = summary.get("results")
            if not isinstance(metadata, dict) or not isinstance(
                result_rows,
                list,
            ):
                raise TypeError(f"invalid summary structure: {summary_path}")
            if metadata.get("experiment") != model.source_id:
                raise ValueError(
                    f"experiment mismatch in {summary_path}: "
                    f"{metadata.get('experiment')!r}"
                )
            if int(metadata.get("seed")) != seed:
                raise ValueError(f"seed mismatch in {summary_path}")
            if int(metadata.get("row_count")) != 10_000:
                raise ValueError(f"unexpected row count in {summary_path}")
            if metadata.get("rows_sha256") != _sha256(rows_path):
                raise ValueError(f"row SHA256 mismatch in {summary_path}")
            expected_sweep = [list(point) for point in dense.MOLECULE_SWEEP]
            if metadata.get("sweep_points") != expected_sweep:
                raise ValueError(f"unexpected sweep grid in {summary_path}")
            if int(metadata.get("motif_count")) != 10:
                raise ValueError(f"unexpected motif count in {summary_path}")
            if int(metadata.get("samples_per_motif")) != 100:
                raise ValueError(
                    f"unexpected samples-per-motif in {summary_path}"
                )
            self._row_hashes[(model.source_id, seed)] = metadata[
                "rows_sha256"
            ]
            counts = Counter()
            sample_indices = {}
            line_count = 0
            with rows_path.open() as handle:
                for line_number, line in enumerate(handle, start=1):
                    row = json.loads(line)
                    if row.get("experiment") != model.source_id:
                        raise ValueError(
                            f"experiment mismatch at {rows_path}:{line_number}"
                        )
                    if int(row.get("seed")) != seed:
                        raise ValueError(
                            f"seed mismatch at {rows_path}:{line_number}"
                        )
                    key = (
                        int(row["point_index"]),
                        int(row["motif_index"]),
                    )
                    point_index, motif_index = key
                    if point_index not in range(10) or motif_index not in range(10):
                        raise ValueError(
                            f"out-of-range point/motif index at "
                            f"{rows_path}:{line_number}"
                        )
                    randomness, temperature = dense.MOLECULE_SWEEP[
                        point_index
                    ]
                    if dense._molecule_key(
                        row["randomness"],
                        row["generation_temperature"],
                    ) != dense._molecule_key(randomness, temperature):
                        raise ValueError(
                            f"sweep coordinate mismatch at "
                            f"{rows_path}:{line_number}"
                        )
                    sample_index = int(row["sample_index"])
                    if sample_index not in range(100):
                        raise ValueError(
                            f"out-of-range sample index at "
                            f"{rows_path}:{line_number}"
                        )
                    counts[key] += 1
                    sample_indices.setdefault(key, set()).add(sample_index)
                    line_count += 1
            expected_keys = {
                (point_index, motif_index)
                for point_index in range(10)
                for motif_index in range(10)
            }
            if line_count != 10_000 or set(counts) != expected_keys:
                raise ValueError(
                    f"incomplete row coverage in {rows_path}: "
                    f"lines={line_count} keys={len(counts)}"
                )
            if any(count != 100 for count in counts.values()):
                raise ValueError(
                    f"each point/motif cell must contain 100 rows: {rows_path}"
                )
            if any(
                indices != set(range(100))
                for indices in sample_indices.values()
            ):
                raise ValueError(
                    f"duplicate or missing sample indices in {rows_path}"
                )
            if len(result_rows) != 10:
                raise ValueError(
                    f"expected 10 summary rows in {summary_path}"
                )
            for point_index, row in enumerate(result_rows):
                if row.get("experiment") != model.source_id:
                    raise ValueError(
                        f"summary experiment mismatch in {summary_path}"
                    )
                if int(row.get("seed")) != seed:
                    raise ValueError(
                        f"summary seed mismatch in {summary_path}"
                    )
                if int(row.get("point_index")) != point_index:
                    raise ValueError(
                        f"summary point order mismatch in {summary_path}"
                    )
            rows.extend(result_rows)
        self._cache[seed] = rows
        return rows

    def validate_raw_seed_independence(self) -> None:
        for seed in dense.SEEDS:
            self.rows("denovo", seed)
        for model in PANEL.models:
            hashes = [
                self._row_hashes[(model.source_id, seed)]
                for seed in dense.SEEDS
            ]
            if len(set(hashes)) != len(hashes):
                raise ValueError(
                    "duplicated raw seed-level motif outputs for "
                    f"{model.source_id}"
                )


def _aggregate_auxiliary(
    store: MotifResultStore,
    model: dense.ModelSpec,
    field: str,
) -> dict[str, tuple[float, float]]:
    values_by_label = {
        dense._molecule_label(randomness, temperature): []
        for randomness, temperature in dense.MOLECULE_SWEEP
    }
    for seed in dense.SEEDS:
        rows = [
            row
            for row in store.rows("denovo", seed)
            if row["experiment"] == model.source_id
        ]
        rows_by_point = {
            dense._molecule_key(
                row["randomness"],
                row["generation_temperature"],
            ): row
            for row in rows
        }
        if len(rows_by_point) != len(dense.MOLECULE_SWEEP):
            raise ValueError(
                f"incomplete auxiliary rows for {model.source_id}, seed={seed}"
            )
        for randomness, temperature in dense.MOLECULE_SWEEP:
            key = dense._molecule_key(randomness, temperature)
            label = dense._molecule_label(randomness, temperature)
            values_by_label[label].append(float(rows_by_point[key][field]))
    return {
        label: dense.mean_ci95(values)
        for label, values in values_by_label.items()
    }


def _plot_figure2(store: MotifResultStore, output_path: Path) -> None:
    dense.configure_style(20)
    figure, axis = plt.subplots(figsize=(11.5, 7.3))
    figure.patch.set_facecolor("white")
    dense._draw_main_panel(axis, store, PANEL)
    figure.supxlabel("Utility", y=0.22)
    figure.supylabel("Diversity", x=0.035)
    figure.legend(
        handles=dense._main_legend_handles((PANEL,)),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.15,
        handletextpad=0.5,
    )
    figure.subplots_adjust(left=0.13, right=0.99, top=0.91, bottom=0.35)
    figure.savefig(output_path)
    plt.close(figure)


def _section(store, metrics, references, figure_name):
    points = ", ".join(
        f"({randomness:g}, {temperature:g})"
        for randomness, temperature in dense.MOLECULE_SWEEP
    )
    lines = [
        "## GenMol Motif Extension",
        "",
        "| Sweep points | Runs | Test motifs | Generations per motif | Samples per point |",
        "|---|---:|---:|---:|---:|",
        f"| `{points}` | 5 | 10 | 100 | 1,000 |",
        "",
        "Utility is the valid, motif-retaining mean de novo soft reward. "
        "Diversity is computed independently within each test motif and then "
        "averaged over the 10 motifs.",
        "",
        "### Table 1: Frontier Metrics",
        "",
        "| Metric | " + " | ".join(model.label for model in PANEL.models) + " |",
        "|---|" + "|".join("---:" for _ in PANEL.models) + "|",
    ]
    for metric, direction in (("HV", "↑"), ("DIP", "↓"), ("R2", "↓")):
        values = [
            dense._format_interval(
                *metrics[PANEL.key][f"{model.label}:{metric}"]
            )
            for model in PANEL.models
        ]
        lines.append(f"| {metric} {direction} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "### Per-Run HV Reference Points",
            "",
            "| Seed | Utility reference | Diversity reference |",
            "|---:|---:|---:|",
        ]
    )
    for seed, reference in zip(dense.SEEDS, references[PANEL.key]):
        lines.append(
            f"| {seed} | {reference.utility:.4f} | "
            f"{reference.diversity:.4f} |"
        )
    lines.extend(
        [
            "",
            "### Figure 2: Utility-Diversity Operating Points",
            "",
            f"[PDF]({figure_name})",
            "",
            "| Model | Sweep point | Utility mean | Utility 95% CI | "
            "Diversity mean | Diversity 95% CI | Raw validity mean | "
            "Raw validity 95% CI | Motif retention mean | "
            "Motif retention 95% CI | Task-valid mean | "
            "Task-valid 95% CI |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in PANEL.models:
        raw_validity = _aggregate_auxiliary(
            store,
            model,
            "raw_valid_fraction",
        )
        retention = _aggregate_auxiliary(
            store,
            model,
            "motif_retention_fraction",
        )
        task_validity = _aggregate_auxiliary(
            store,
            model,
            "task_valid_fraction",
        )
        for point in dense.aggregate_series(store, PANEL, model):
            raw_validity_mean, raw_validity_ci = raw_validity[
                point.point_label
            ]
            retention_mean, retention_ci = retention[point.point_label]
            task_validity_mean, task_validity_ci = task_validity[
                point.point_label
            ]
            lines.append(
                f"| {model.label} | {point.point_label} | "
                f"{point.utility_mean:.4f} | {point.utility_ci95:.4f} | "
                f"{point.diversity_mean:.4f} | "
                f"{point.diversity_ci95:.4f} | "
                f"{raw_validity_mean:.4f} | {raw_validity_ci:.4f} | "
                f"{retention_mean:.4f} | {retention_ci:.4f} | "
                f"{task_validity_mean:.4f} | {task_validity_ci:.4f} |"
            )
    return "\n".join(lines) + "\n"


def _upsert(path: Path, section: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"expanded result Markdown not found: {path}")
    text = path.read_text()
    begin_count = text.count(SECTION_BEGIN)
    end_count = text.count(SECTION_END)
    if begin_count != end_count or begin_count > 1:
        raise RuntimeError(
            f"malformed motif result markers in {path}: "
            f"begin={begin_count} end={end_count}"
        )
    wrapped = f"{SECTION_BEGIN}\n{section.rstrip()}\n{SECTION_END}\n"
    if begin_count:
        prefix, remainder = text.split(SECTION_BEGIN, 1)
        _, suffix = remainder.split(SECTION_END, 1)
        updated = prefix.rstrip() + "\n\n" + wrapped + suffix.lstrip()
    else:
        updated = text.rstrip() + "\n\n" + wrapped
    path.write_text(updated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expanded-results-path", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = MotifResultStore(args.run_root)
    store.validate_raw_seed_independence()
    dense.validate_seed_independence(
        store,
        (PANEL,),
        include_denovo_auxiliary=False,
    )
    metrics, references = dense.table1_metrics(store, (PANEL,))
    figure_path = args.output_dir / "figure2-motif-extension.pdf"
    _plot_figure2(store, figure_path)
    section = _section(store, metrics, references, figure_path.name)
    standalone = args.output_dir / "motif-extension-results.md"
    standalone.write_text("# Motif-Extension Results\n\n" + section)
    _upsert(args.expanded_results_path, section)
    print(figure_path)
    print(standalone)
    print(args.expanded_results_path)


if __name__ == "__main__":
    main()
