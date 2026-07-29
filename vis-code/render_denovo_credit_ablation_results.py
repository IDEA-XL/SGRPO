#!/usr/bin/env python3
"""Render the de novo credit-ablation Table 1 and Figure 2 results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

import render_rebuttal_dense_results as dense


SECTION_BEGIN = "<!-- BEGIN DENOVO CREDIT ABLATION RESULTS -->"
SECTION_END = "<!-- END DENOVO CREDIT ABLATION RESULTS -->"
EXPECTED_DIVERSITY_METRIC = "morgan_internal_diversity"
SGRPO_MODEL_ID = "genmol_denovo_sgrpo_rewardsum_loo_2000"
GRPO_MODEL_ID = "genmol_denovo_grpo_2000"
EXTRA_MODEL_IDS = (
    "denovo_raw_loo_diversity_2000",
    "denovo_mean_baseline_2000",
    "denovo_mean_baseline_std_2000",
)
BASE_MODEL_IDS = (SGRPO_MODEL_ID, GRPO_MODEL_ID)

SGRPO_MODEL = dense.ModelSpec(
    SGRPO_MODEL_ID,
    "SGRPO",
    dense.COLOR_SGRPO,
    "o",
)
GRPO_MODEL = dense.ModelSpec(
    GRPO_MODEL_ID,
    "GRPO",
    dense.COLOR_GRPO,
    "^",
)
RAW_LOO_MODEL = dense.ModelSpec(
    "denovo_raw_loo_diversity_2000",
    "Raw-LOO Diversity",
    "#2A9D8F",
    "D",
)
MEAN_BASELINE_MODEL = dense.ModelSpec(
    "denovo_mean_baseline_2000",
    "Mean Baseline",
    "#C85A3E",
    "s",
)
MEAN_BASELINE_STD_MODEL = dense.ModelSpec(
    "denovo_mean_baseline_std_2000",
    "Mean Baseline + Std",
    "#6B5B95",
    "^",
)

PANELS = (
    dense.PanelSpec(
        key="raw_loo_comparison",
        title="Raw-LOO Diversity Comparison",
        source_kind="denovo",
        models=(
            SGRPO_MODEL,
            RAW_LOO_MODEL,
        ),
    ),
    dense.PanelSpec(
        key="mean_baseline_comparison",
        title="Mean-Baseline Comparison",
        source_kind="denovo",
        models=(
            GRPO_MODEL,
            MEAN_BASELINE_MODEL,
            MEAN_BASELINE_STD_MODEL,
        ),
    ),
)
ALL_MODEL_IDS = {*BASE_MODEL_IDS, *EXTRA_MODEL_IDS}


class CreditAblationStore(dense.ResultStore):
    def __init__(self, results_root: Path, base_results_root: Path):
        self.results_root = results_root
        self.base_results_root = base_results_root
        self._cache: dict[tuple[str, int], list[dict]] = {}
        missing = [
            path
            for root in (results_root, base_results_root)
            for seed in dense.SEEDS
            if not (path := root / "denovo" / f"seed{seed}.json").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing credit-ablation or base sweep summaries:\n"
                + "\n".join(str(path) for path in missing)
            )

    def rows(self, kind: str, seed: int) -> list[dict]:
        if kind != "denovo":
            raise ValueError(f"Unsupported credit-ablation source kind: {kind}")
        cache_key = (kind, seed)
        if cache_key not in self._cache:
            extra_path = self.results_root / kind / f"seed{seed}.json"
            base_path = self.base_results_root / kind / f"seed{seed}.json"
            extra_rows = json.loads(extra_path.read_text())
            base_rows = json.loads(base_path.read_text())
            if not isinstance(extra_rows, list) or not isinstance(base_rows, list):
                raise TypeError("De novo sweep summaries must be JSON lists")
            base_model_rows = [
                row
                for row in base_rows
                if row.get("experiment") in BASE_MODEL_IDS
            ]
            legacy_metrics = {
                row.get("diversity_metric") for row in base_model_rows
            }
            if not legacy_metrics.issubset({None, EXPECTED_DIVERSITY_METRIC}):
                raise ValueError(
                    f"Seed {seed} base rows contain an unexpected diversity metric: "
                    f"{legacy_metrics}"
                )
            if any(
                row.get("diversity_metric") != EXPECTED_DIVERSITY_METRIC
                for row in extra_rows
            ):
                raise ValueError(
                    f"Seed {seed} credit-ablation rows contain a non-Morgan metric"
                )
            self._cache[cache_key] = [*base_model_rows, *extra_rows]
        rows = self._cache[cache_key]
        expected = len(ALL_MODEL_IDS) * len(dense.MOLECULE_SWEEP)
        if len(rows) != expected:
            raise ValueError(
                f"Seed {seed} has {len(rows)} credit-ablation rows; "
                f"expected {expected}"
            )
        actual_models = {row.get("experiment") for row in rows}
        if actual_models != ALL_MODEL_IDS:
            raise ValueError(
                f"Seed {seed} model mismatch: "
                f"expected={ALL_MODEL_IDS}, actual={actual_models}"
            )
        return rows


def _validate_manifest(results_root: Path) -> dict:
    path = results_root / "manifest.json"
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing materialization manifest: {path}")
    manifest = json.loads(path.read_text())
    checks = {
        "profile": "denovo_credit_ablation_materialized",
        "diversity_metric": EXPECTED_DIVERSITY_METRIC,
        "seeds": list(dense.SEEDS),
        "sweep_points": [list(point) for point in dense.MOLECULE_SWEEP],
        "num_samples_per_model_point": 1000,
        "generation_batch_size": 2048,
        "total_expected_raw_rows": 150_000,
        "total_validated_raw_rows": 150_000,
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Manifest field {key!r} is {manifest.get(key)!r}; "
                f"expected {expected!r}"
            )
    models = manifest.get("models")
    if not isinstance(models, list) or [
        model.get("name") for model in models
    ] != list(EXTRA_MODEL_IDS):
        raise ValueError("Manifest model order or identity is invalid")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or len(sources) != 15:
        raise ValueError("Manifest must contain all 15 model-seed sources")
    if any(
        int(record.get("raw_rows", -1)) != 10_000
        or len(str(record.get("raw_rows_sha256", ""))) != 64
        or len(str(record.get("summary_sha256", ""))) != 64
        for record in sources
    ):
        raise ValueError("Manifest contains an invalid source audit record")
    return manifest


def _validate_sgrpo_audit(results_root: Path) -> dict:
    path = results_root / "sgrpo-base-morgan-audit.json"
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing SGRPO Morgan-diversity audit: {path}")
    audit = json.loads(path.read_text())
    checks = {
        "profile": "denovo_sgrpo_base_morgan_audit",
        "model": SGRPO_MODEL_ID,
        "diversity_metric": EXPECTED_DIVERSITY_METRIC,
        "seeds": list(dense.SEEDS),
        "sweep_points": [list(point) for point in dense.MOLECULE_SWEEP],
        "samples_per_model_point": 1000,
        "total_validated_raw_rows": 50_000,
    }
    for key, expected in checks.items():
        if audit.get(key) != expected:
            raise ValueError(
                f"SGRPO audit field {key!r} is {audit.get(key)!r}; "
                f"expected {expected!r}"
            )
    sources = audit.get("sources")
    if not isinstance(sources, list) or len(sources) != len(dense.SEEDS):
        raise ValueError("SGRPO audit must contain all five seed sources")
    if any(
        int(record.get("raw_rows", -1)) != 10_000
        or len(str(record.get("raw_rows_sha256", ""))) != 64
        or len(str(record.get("summary_sha256", ""))) != 64
        or len(str(record.get("config_sha256", ""))) != 64
        for record in sources
    ):
        raise ValueError("SGRPO audit contains an invalid source record")
    return audit


def _plot_figure2(
    store: CreditAblationStore,
    panel: dense.PanelSpec,
    output_path: Path,
) -> None:
    dense.configure_style(20)
    figure, axis = plt.subplots(figsize=(11.8, 7.8))
    figure.patch.set_facecolor("white")
    dense._draw_main_panel(axis, store, panel)
    figure.supxlabel("Utility", y=0.205)
    figure.supylabel("Diversity", x=0.025)
    figure.legend(
        handles=dense._main_legend_handles((panel,)),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    figure.subplots_adjust(left=0.13, right=0.99, top=0.90, bottom=0.36)
    figure.savefig(output_path)
    plt.close(figure)


def _result_section(
    store: CreditAblationStore,
    metrics: dict[str, dict[str, tuple[float, float]]],
    references: dict[str, list[dense.Point]],
    figure_links: dict[str, str],
) -> str:
    molecule_points = ", ".join(
        f"({randomness:g}, {temperature:g})"
        for randomness, temperature in dense.MOLECULE_SWEEP
    )
    lines = [
        "## GenMol De Novo: Credit-Baseline Ablation",
        "",
        "This is an additional five-run evaluation and does not replace any existing "
        "result section. The paper SGRPO and GRPO sweeps and the three new "
        "credit-baseline checkpoints use the standard GenMol De Novo protocol: "
        "QED/SA soft Utility, Morgan-fingerprint internal Diversity, and the ten paired "
        "randomness-temperature sweep points. Results are divided into two independent "
        "comparison groups; no four-model joint metric is reported.",
        "",
        "| Sweep points | Runs | Samples per model and point |",
        "|---|---:|---:|",
        f"| `{molecule_points}` | 5 | 1,000 |",
    ]

    for group_index, panel in enumerate(PANELS, start=1):
        lines.extend(
            [
                "",
                f"### Group {group_index}: {panel.title}",
                "",
                "The per-run HV reference point is determined only from the models "
                "listed in this group.",
                "",
                "#### Table 1: Frontier Metrics",
                "",
                "| Metric | "
                + " | ".join(model.label for model in panel.models)
                + " |",
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
            lines.append(
                f"| {metric} {direction} | " + " | ".join(values) + " |"
            )

        lines.extend(
            [
                "",
                "#### Per-Run HV Reference Points",
                "",
                "| Seed | Utility reference | Diversity reference |",
                "|---:|---:|---:|",
            ]
        )
        for seed, point in zip(dense.SEEDS, references[panel.key]):
            lines.append(
                f"| {seed} | {point.utility:.4f} | {point.diversity:.4f} |"
            )

        lines.extend(
            [
                "",
                "#### Figure 2: Utility-Diversity Operating Points",
                "",
                f"[PDF]({figure_links[panel.key]})",
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
    return "\n".join(lines) + "\n"


def _figure_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "raw_loo_comparison": output_dir
        / "figure2-denovo-raw-loo-comparison.pdf",
        "mean_baseline_comparison": output_dir
        / "figure2-denovo-mean-baseline-comparison.pdf",
    }


def _upsert_section(path: Path, section: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Expanded results Markdown does not exist: {path}")
    text = path.read_text()
    begin_count = text.count(SECTION_BEGIN)
    end_count = text.count(SECTION_END)
    if begin_count != end_count or begin_count > 1:
        raise RuntimeError(
            f"Malformed credit-ablation markers in {path}: "
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
    parser.add_argument("--base-results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expanded-results-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _validate_manifest(args.results_root)
    _validate_sgrpo_audit(args.results_root)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    store = CreditAblationStore(
        args.results_root,
        args.base_results_root,
    )
    dense.validate_seed_independence(
        store,
        PANELS,
        include_denovo_auxiliary=False,
    )
    metrics, references = dense.table1_metrics(store, PANELS)

    figure_paths = _figure_paths(output_dir)
    standalone_path = output_dir / "denovo-credit-ablation-results.md"
    figure_links = {
        key: os.path.relpath(
            path,
            start=args.expanded_results_path.parent,
        ).replace(os.sep, "/")
        for key, path in figure_paths.items()
    }
    for panel in PANELS:
        _plot_figure2(store, panel, figure_paths[panel.key])
    section = _result_section(store, metrics, references, figure_links)
    standalone_path.write_text(
        "# GenMol De Novo Credit-Baseline Ablation\n\n" + section
    )
    _upsert_section(args.expanded_results_path, section)
    legacy_figure_path = output_dir / "figure2-denovo-credit-ablation.pdf"
    if legacy_figure_path.is_file():
        legacy_figure_path.unlink()
    for path in figure_paths.values():
        print(path)
    print(standalone_path)
    print(args.expanded_results_path)


if __name__ == "__main__":
    main()
