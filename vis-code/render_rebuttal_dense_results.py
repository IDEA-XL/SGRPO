#!/usr/bin/env python3
"""Render expanded-sweep rebuttal figures and Markdown tables."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import FormatStrFormatter


SEEDS = (42, 43, 44, 45, 46)
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
PROGEN2_TEMPERATURES = tuple(round(0.1 + 0.05 * index, 2) for index in range(23))
T_CRITICAL_95_DF4 = 2.7764451052
R2_WEIGHTS = tuple((index / 100.0, 1.0 - index / 100.0) for index in range(101))

COLOR_ORIGINAL = "#7A9E3A"
COLOR_GRPO = "#2A9D8F"
COLOR_SGRPO = "#1F4E79"
COLOR_MEMORY = "#B85C38"
COLOR_NO_LOO = "#D97706"
COLOR_GROUP_SIZE = "#8E63B6"
COLOR_DIVERSITY_WEIGHT = "#D1495B"

UNIFORM_FONT_SIZE = 24
SMALL_PANEL_FONT_SIZE = 20
NON_PARETO_ALPHA = 0.25
SHADE_ALPHA = 0.10
ERRORBAR_LINEWIDTH = 3.10
PLOT_MARKER_AREA = 92.0 / 9.0
REFERENCE_LINE_COLOR = "#B5B5B5"
REFERENCE_MARKER_COLOR = "#606060"
REFERENCE_LINESTYLE = (0, (1.2, 2.6))
REFERENCE_LINEWIDTH = 1.15
FRONTIER_LINESTYLE = (0, (4.5, 3.0))

APTOS_DISPLAY_FONT_PATH = (
    Path.home()
    / "Library"
    / "Group Containers"
    / "UBF8T346G9.Office"
    / "FontCache"
    / "4"
    / "CloudFonts"
    / "Aptos Display"
    / "32677218994.ttf"
)


@dataclass(frozen=True)
class ModelSpec:
    source_id: str
    label: str
    color: str
    marker: str


@dataclass(frozen=True)
class PanelSpec:
    key: str
    title: str
    source_kind: str
    models: tuple[ModelSpec, ...]


@dataclass(frozen=True)
class Point:
    utility: float
    diversity: float


@dataclass(frozen=True)
class RunPoint:
    point_label: str
    utility: float
    diversity: float


@dataclass(frozen=True)
class AggregatePoint:
    point_label: str
    utility_mean: float
    utility_ci95: float
    diversity_mean: float
    diversity_ci95: float
    utility_runs: tuple[float, ...]
    diversity_runs: tuple[float, ...]


@dataclass
class PlotPoint:
    utility: float
    utility_ci95: float
    diversity: float
    diversity_ci95: float
    model: ModelSpec
    sweep_rank: int
    model_pareto: bool = False


@dataclass(frozen=True)
class CurveSpec:
    x_value: float
    experiment: str


MAIN_PANELS = (
    PanelSpec(
        key="denovo",
        title="De Novo Molecule Design",
        source_kind="denovo",
        models=(
            ModelSpec("original_genmol_v2", "Original", COLOR_ORIGINAL, "D"),
            ModelSpec("genmol_denovo_grpo_2000", "GRPO", COLOR_GRPO, "^"),
            ModelSpec(
                "genmol_denovo_sgrpo_rewardsum_loo_2000",
                "SGRPO",
                COLOR_SGRPO,
                "o",
            ),
            ModelSpec(
                "genmol_denovo_grpo_hbd_2000",
                "Memory-Assisted GRPO",
                COLOR_MEMORY,
                "s",
            ),
        ),
    ),
    PanelSpec(
        key="mmgenmol",
        title="Pocket-Based Design",
        source_kind="mmgenmol",
        models=(
            ModelSpec("original_5500", "Original", COLOR_ORIGINAL, "D"),
            ModelSpec("grpo_unidock_1000", "GRPO", COLOR_GRPO, "^"),
            ModelSpec(
                "sgrpo_unidock_rewardsum_loo_1000",
                "SGRPO",
                COLOR_SGRPO,
                "o",
            ),
        ),
    ),
    PanelSpec(
        key="progen2",
        title="De Novo Protein Design",
        source_kind="progen2",
        models=(
            ModelSpec("original", "Original", COLOR_ORIGINAL, "D"),
            ModelSpec("grpo_step100", "GRPO", COLOR_GRPO, "^"),
            ModelSpec(
                "sgrpo_gw08_rewardsum_loo_step100",
                "SGRPO",
                COLOR_SGRPO,
                "o",
            ),
            ModelSpec(
                "grpo_hbd_step100",
                "Memory-Assisted GRPO",
                COLOR_MEMORY,
                "s",
            ),
        ),
    ),
)
REBUTTAL_PANELS = MAIN_PANELS[:2]

ABLATION_MODELS = (
    ModelSpec(
        "genmol_denovo_sgrpo_rewardsum_loo_2000",
        "SGRPO",
        COLOR_SGRPO,
        "o",
    ),
    ModelSpec(
        "genmol_denovo_sgrpo_rewardsum_2000",
        "SGRPO w.o. LOO",
        COLOR_NO_LOO,
        "s",
    ),
    ModelSpec("genmol_denovo_grpo_2000", "GRPO", COLOR_GRPO, "^"),
)

GROUP_SIZE_SPECS = (
    CurveSpec(1, "genmol_denovo_grpo_2000"),
    CurveSpec(4, "genmol_denovo_sgrpo_ng4_rewardsum_loo_2000"),
    CurveSpec(16, "genmol_denovo_sgrpo_ng16_rewardsum_loo_2000"),
    CurveSpec(32, "genmol_denovo_sgrpo_ng32_rewardsum_loo_2000"),
    CurveSpec(64, "genmol_denovo_sgrpo_rewardsum_loo_2000"),
)

DIVERSITY_WEIGHT_SPECS = (
    CurveSpec(0.0, "genmol_denovo_grpo_2000"),
    CurveSpec(0.1, "genmol_denovo_sgrpo_gw01_rewardsum_loo_2000"),
    CurveSpec(0.3, "genmol_denovo_sgrpo_gw03_rewardsum_loo_2000"),
    CurveSpec(0.5, "genmol_denovo_sgrpo_gw05_rewardsum_loo_2000"),
    CurveSpec(0.7, "genmol_denovo_sgrpo_gw07_rewardsum_loo_2000"),
    CurveSpec(0.9, "genmol_denovo_sgrpo_rewardsum_loo_2000"),
)


class ResultStore:
    def __init__(self, results_root: Path, source_kinds: Iterable[str]):
        self.results_root = results_root
        self._cache: dict[tuple[str, int], list[dict]] = {}
        kinds = tuple(source_kinds)
        if not kinds:
            raise ValueError("At least one result source is required")
        missing = [
            self._path(kind, seed)
            for kind in kinds
            for seed in SEEDS
            if not self._path(kind, seed).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing dense sweep summaries:\n" + "\n".join(str(path) for path in missing)
            )

    def _path(self, kind: str, seed: int) -> Path:
        return self.results_root / kind / f"seed{seed}.json"

    def rows(self, kind: str, seed: int) -> list[dict]:
        cache_key = (kind, seed)
        if cache_key not in self._cache:
            path = self._path(kind, seed)
            value = json.loads(path.read_text())
            if kind == "progen2":
                if not isinstance(value, dict) or not isinstance(value.get("results"), list):
                    raise TypeError(f"Expected dict with results list in {path}")
                rows = value["results"]
            else:
                if not isinstance(value, list):
                    raise TypeError(f"Expected list in {path}")
                rows = value
            self._cache[cache_key] = rows
        return self._cache[cache_key]

    def series(self, panel: PanelSpec, model: ModelSpec, seed: int) -> list[RunPoint]:
        rows = self.rows(panel.source_kind, seed)
        if panel.source_kind == "denovo":
            subset = [row for row in rows if row["experiment"] == model.source_id]
            row_by_point = {
                _molecule_key(row["randomness"], row["generation_temperature"]): row
                for row in subset
            }
            _validate_exact_sweep_keys(
                row_by_point,
                subset,
                {_molecule_key(*point) for point in MOLECULE_SWEEP},
                context=f"de novo seed={seed} model={model.source_id}",
            )
            return [
                RunPoint(
                    point_label=_molecule_label(randomness, temperature),
                    utility=_finite(row_by_point[_molecule_key(randomness, temperature)]["soft_reward_mean"]),
                    diversity=_finite(row_by_point[_molecule_key(randomness, temperature)]["diversity"]),
                )
                for randomness, temperature in MOLECULE_SWEEP
            ]
        if panel.source_kind == "mmgenmol":
            subset = [row for row in rows if row["model_name"] == model.source_id]
            row_by_point = {
                _molecule_key(row["randomness"], row["temperature"]): row for row in subset
            }
            _validate_exact_sweep_keys(
                row_by_point,
                subset,
                {_molecule_key(*point) for point in MOLECULE_SWEEP},
                context=f"mmGenMol seed={seed} model={model.source_id}",
            )
            series = []
            for randomness, temperature in MOLECULE_SWEEP:
                row = row_by_point[_molecule_key(randomness, temperature)]
                utility = (
                    _finite(row["soft_reward_mean"])
                    if model.source_id != "original_5500"
                    else _mmgenmol_original_utility(row)
                )
                series.append(
                    RunPoint(
                        point_label=_molecule_label(randomness, temperature),
                        utility=utility,
                        diversity=_finite(row["diversity"]),
                    )
                )
            return series
        if panel.source_kind == "progen2":
            subset = [row for row in rows if row["experiment"] == model.source_id]
            row_by_temperature = {round(float(row["temperature"]), 4): row for row in subset}
            _validate_exact_sweep_keys(
                row_by_temperature,
                subset,
                {round(temperature, 4) for temperature in PROGEN2_TEMPERATURES},
                context=f"ProGen2 seed={seed} model={model.source_id}",
            )
            return [
                RunPoint(
                    point_label=f"T={temperature:.2f}",
                    utility=_finite(row_by_temperature[round(temperature, 4)]["soft_reward_mean"]),
                    diversity=_finite(row_by_temperature[round(temperature, 4)]["diversity"]),
                )
                for temperature in PROGEN2_TEMPERATURES
            ]
        raise ValueError(f"Unsupported panel source: {panel.source_kind}")

    def denovo_series(self, experiment: str, seed: int) -> list[RunPoint]:
        panel = PanelSpec(
            key="denovo",
            title="",
            source_kind="denovo",
            models=(ModelSpec(experiment, experiment, "", "o"),),
        )
        return self.series(panel, panel.models[0], seed)


def _finite(value: object) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"Expected finite value, got {value!r}")
    return converted


def _validate_exact_sweep_keys(
    row_by_key: dict,
    source_rows: list[dict],
    expected_keys: set,
    *,
    context: str,
) -> None:
    actual_keys = set(row_by_key)
    if len(source_rows) != len(row_by_key):
        raise ValueError(f"Duplicate sweep rows for {context}")
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            f"Sweep grid mismatch for {context}: missing={missing}, unexpected={unexpected}"
        )


def _molecule_key(randomness: object, temperature: object) -> tuple[float, float]:
    return round(float(randomness), 4), round(float(temperature), 4)


def _molecule_label(randomness: float, temperature: float) -> str:
    return f"(R={randomness:.2f}, T={temperature:.2f})"


def _mmgenmol_original_utility(row: dict) -> float:
    vina_score = max(0.0, min(-_finite(row["vina_dock_mean"]) / 10.0, 1.0))
    return 0.3 * _finite(row["qed_mean"]) + 0.2 * _finite(row["sa_score_mean"]) + 0.5 * vina_score


def mean_ci95(values: Iterable[float]) -> tuple[float, float]:
    frozen = tuple(_finite(value) for value in values)
    if len(frozen) != len(SEEDS):
        raise ValueError(f"Expected {len(SEEDS)} runs, got {len(frozen)}")
    mean = sum(frozen) / len(frozen)
    variance = sum((value - mean) ** 2 for value in frozen) / (len(frozen) - 1)
    standard_error = math.sqrt(variance / len(frozen))
    return mean, T_CRITICAL_95_DF4 * standard_error


def aggregate_series(store: ResultStore, panel: PanelSpec, model: ModelSpec) -> list[AggregatePoint]:
    per_seed = [store.series(panel, model, seed) for seed in SEEDS]
    expected_length = len(per_seed[0])
    if any(len(series) != expected_length for series in per_seed):
        raise ValueError(f"Inconsistent sweep lengths for {panel.title} / {model.label}")
    aggregate = []
    for point_index in range(expected_length):
        aligned = [series[point_index] for series in per_seed]
        labels = {point.point_label for point in aligned}
        if len(labels) != 1:
            raise ValueError(f"Misaligned sweep point for {panel.title} / {model.label}")
        utilities = tuple(point.utility for point in aligned)
        diversities = tuple(point.diversity for point in aligned)
        utility_mean, utility_ci95 = mean_ci95(utilities)
        diversity_mean, diversity_ci95 = mean_ci95(diversities)
        aggregate.append(
            AggregatePoint(
                point_label=aligned[0].point_label,
                utility_mean=utility_mean,
                utility_ci95=utility_ci95,
                diversity_mean=diversity_mean,
                diversity_ci95=diversity_ci95,
                utility_runs=utilities,
                diversity_runs=diversities,
            )
        )
    return aggregate


def validate_seed_independence(
    store: ResultStore, panels: tuple[PanelSpec, ...]
) -> None:
    panel_models = [
        (panel, model) for panel in panels for model in panel.models
    ]
    denovo_panel = next(
        (panel for panel in panels if panel.source_kind == "denovo"),
        None,
    )
    if denovo_panel is not None:
        denovo_experiments = {
            model.source_id for model in ABLATION_MODELS
        } | {
            spec.experiment for spec in (*GROUP_SIZE_SPECS, *DIVERSITY_WEIGHT_SPECS)
        }
        known_denovo = {model.source_id for model in denovo_panel.models}
        panel_models.extend(
            (
                denovo_panel,
                ModelSpec(experiment, experiment, "", "o"),
            )
            for experiment in sorted(denovo_experiments - known_denovo)
        )
    for panel, model in panel_models:
        signatures = []
        for seed in SEEDS:
            series = store.series(panel, model, seed)
            signatures.append(
                tuple(
                    (round(point.utility, 12), round(point.diversity, 12))
                    for point in series
                )
            )
        if len(set(signatures)) != len(SEEDS):
            raise ValueError(
                f"Non-independent or duplicated seed-level sweep summaries for "
                f"{panel.source_kind} / {model.source_id}"
            )


def strictly_dominates(left: Point, right: Point) -> bool:
    return (
        left.utility >= right.utility
        and left.diversity >= right.diversity
        and (left.utility > right.utility or left.diversity > right.diversity)
    )


def non_dominated(points: Iterable[Point]) -> list[Point]:
    frozen = list(points)
    if not frozen:
        raise ValueError("Expected at least one point")
    survivors = [
        point
        for point in frozen
        if not any(strictly_dominates(other, point) for other in frozen if other is not point)
    ]
    unique = sorted({(point.utility, point.diversity) for point in survivors})
    frontier = [Point(utility, diversity) for utility, diversity in unique]
    for left, right in zip(frontier, frontier[1:]):
        if right.diversity > left.diversity + 1e-12:
            raise ValueError("Non-dominated frontier is not monotone")
    return frontier


def compute_hv(points: Iterable[Point], reference: Point) -> float:
    frontier = non_dominated(points)
    hv = 0.0
    previous_utility = reference.utility
    for point in frontier:
        if point.utility < reference.utility or point.diversity < reference.diversity:
            raise ValueError("Reference point is not weakly worse than every frontier point")
        hv += (point.utility - previous_utility) * (point.diversity - reference.diversity)
        previous_utility = point.utility
    return hv


def compute_dip(points: Iterable[Point]) -> float:
    frozen = list(points)
    return min(math.hypot(1.0 - point.utility, 1.0 - point.diversity) for point in frozen)


def compute_r2(points: Iterable[Point]) -> float:
    frozen = list(points)
    losses = []
    for utility_weight, diversity_weight in R2_WEIGHTS:
        losses.append(
            min(
                max(
                    utility_weight * (1.0 - point.utility),
                    diversity_weight * (1.0 - point.diversity),
                )
                for point in frozen
            )
        )
    return sum(losses) / len(losses)


def table1_metrics(
    store: ResultStore,
    panels: tuple[PanelSpec, ...],
) -> tuple[dict[str, dict[str, tuple[float, float]]], dict[str, list[Point]]]:
    output: dict[str, dict[str, tuple[float, float]]] = {}
    references: dict[str, list[Point]] = {}
    for panel in panels:
        per_model_values = {
            model.label: {"HV": [], "DIP": [], "R2": []} for model in panel.models
        }
        panel_references = []
        for seed in SEEDS:
            series_by_model = {
                model.label: [
                    Point(point.utility, point.diversity)
                    for point in store.series(panel, model, seed)
                ]
                for model in panel.models
            }
            all_points = [
                point for model_points in series_by_model.values() for point in model_points
            ]
            reference = Point(
                utility=min(point.utility for point in all_points),
                diversity=min(point.diversity for point in all_points),
            )
            panel_references.append(reference)
            for model in panel.models:
                points = series_by_model[model.label]
                per_model_values[model.label]["HV"].append(compute_hv(points, reference))
                per_model_values[model.label]["DIP"].append(compute_dip(points))
                per_model_values[model.label]["R2"].append(compute_r2(points))
        output[panel.key] = {
            f"{model.label}:{metric}": mean_ci95(per_model_values[model.label][metric])
            for model in panel.models
            for metric in ("HV", "DIP", "R2")
        }
        references[panel.key] = panel_references
    return output, references


def configure_style(font_size: int) -> None:
    font_family = "DejaVu Sans"
    if APTOS_DISPLAY_FONT_PATH.exists():
        font_manager.fontManager.addfont(str(APTOS_DISPLAY_FONT_PATH))
        font_family = font_manager.FontProperties(fname=str(APTOS_DISPLAY_FONT_PATH)).get_name()
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.2,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _plot_points_for_panel(
    store: ResultStore, panel: PanelSpec
) -> dict[str, list[PlotPoint]]:
    output = {}
    for model in panel.models:
        aggregate = list(reversed(aggregate_series(store, panel, model)))
        output[model.label] = [
            PlotPoint(
                utility=point.utility_mean,
                utility_ci95=point.utility_ci95,
                diversity=point.diversity_mean,
                diversity_ci95=point.diversity_ci95,
                model=model,
                sweep_rank=rank,
            )
            for rank, point in enumerate(aggregate)
        ]
    return output


def _mark_model_pareto(points_by_model: dict[str, list[PlotPoint]]) -> None:
    for points in points_by_model.values():
        for point in points:
            candidate = Point(point.utility, point.diversity)
            point.model_pareto = not any(
                strictly_dominates(Point(other.utility, other.diversity), candidate)
                for other in points
                if other is not point
            )


def _model_frontier(points: list[PlotPoint]) -> list[PlotPoint]:
    frontier = [point for point in points if point.model_pareto]
    frontier.sort(key=lambda point: (point.utility, -point.diversity))
    if not frontier:
        raise ValueError("Expected a non-empty model frontier")
    return frontier


def _panel_limits(
    points: Iterable[PlotPoint],
) -> tuple[float, float, float, float]:
    frozen = list(points)
    utility_low = min(point.utility - point.utility_ci95 for point in frozen)
    utility_high = max(point.utility + point.utility_ci95 for point in frozen)
    diversity_low = min(point.diversity - point.diversity_ci95 for point in frozen)
    diversity_high = max(point.diversity + point.diversity_ci95 for point in frozen)
    utility_span = utility_high - utility_low
    diversity_span = diversity_high - diversity_low
    utility_pad = max(utility_span * 0.07, 0.015)
    diversity_pad = max(diversity_span * 0.10, 0.02)
    return (
        utility_low - utility_pad,
        utility_high + utility_pad * 0.55,
        max(0.0, diversity_low - diversity_pad),
        min(1.0, diversity_high + diversity_pad * 0.45),
    )


def _set_utility_ticks(
    axis: plt.Axes, source_kind: str, utility_min: float, utility_max: float
) -> None:
    step = 0.1 if source_kind == "progen2" else 0.05
    current = math.ceil((utility_min - 1e-12) / step) * step
    ticks = []
    while current <= utility_max + 1e-12:
        ticks.append(round(current, 2))
        current += step
    if ticks:
        axis.set_xticks(ticks)


def _draw_main_panel(axis: plt.Axes, store: ResultStore, panel: PanelSpec) -> None:
    points_by_model = _plot_points_for_panel(store, panel)
    _mark_model_pareto(points_by_model)
    all_points = [point for points in points_by_model.values() for point in points]
    frontiers = {label: _model_frontier(points) for label, points in points_by_model.items()}
    utility_min, utility_max, diversity_min, diversity_max = _panel_limits(all_points)
    reference_utility = min(point.utility for point in all_points)
    reference_diversity = min(point.diversity for point in all_points)
    min_utility_point = min(all_points, key=lambda point: (point.utility, point.diversity))
    min_diversity_point = min(all_points, key=lambda point: (point.diversity, point.utility))

    axis.set_facecolor("white")
    axis.grid(False)
    axis.set_xlim(utility_min, utility_max)
    axis.set_ylim(diversity_min, diversity_max)
    axis.set_title(panel.title, pad=12)
    _set_utility_ticks(axis, panel.source_kind, utility_min, utility_max)
    axis.plot(
        [reference_utility, min_diversity_point.utility],
        [reference_diversity, reference_diversity],
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINEWIDTH,
        linestyle=REFERENCE_LINESTYLE,
        zorder=1.0,
    )
    axis.plot(
        [reference_utility, reference_utility],
        [reference_diversity, min_utility_point.diversity],
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINEWIDTH,
        linestyle=REFERENCE_LINESTYLE,
        zorder=1.0,
    )
    axis.scatter(
        [reference_utility],
        [reference_diversity],
        s=120,
        marker="x",
        color=REFERENCE_MARKER_COLOR,
        linewidths=2.0,
        zorder=1.1,
    )

    for model in panel.models:
        points = sorted(points_by_model[model.label], key=lambda point: point.sweep_rank)
        frontier = frontiers[model.label]
        polygon_points = [(utility_min, diversity_min), (utility_min, frontier[0].diversity)]
        polygon_points.extend((point.utility, point.diversity) for point in frontier)
        polygon_points.append((frontier[-1].utility, diversity_min))
        axis.add_patch(
            Polygon(
                polygon_points,
                closed=True,
                facecolor=model.color,
                edgecolor="none",
                alpha=SHADE_ALPHA,
                zorder=0.0,
            )
        )
        for left, right in zip(points, points[1:]):
            alpha = 1.0 if left.model_pareto and right.model_pareto else NON_PARETO_ALPHA
            axis.plot(
                [left.utility, right.utility],
                [left.diversity, right.diversity],
                color=model.color,
                linewidth=2.7,
                alpha=alpha,
                solid_capstyle="round",
                zorder=2.0,
            )
        for left, right in zip(frontier, frontier[1:]):
            axis.plot(
                [left.utility, right.utility],
                [left.diversity, right.diversity],
                color="white",
                linewidth=4.4,
                solid_capstyle="round",
                zorder=2.2,
            )
            axis.plot(
                [left.utility, right.utility],
                [left.diversity, right.diversity],
                color=model.color,
                linewidth=2.4,
                linestyle=FRONTIER_LINESTYLE,
                solid_capstyle="round",
                zorder=2.3,
            )
        for point in points:
            alpha = 1.0 if point.model_pareto else NON_PARETO_ALPHA
            axis.errorbar(
                point.utility,
                point.diversity,
                xerr=point.utility_ci95,
                yerr=point.diversity_ci95,
                fmt="none",
                ecolor=model.color,
                elinewidth=ERRORBAR_LINEWIDTH,
                alpha=alpha,
                capsize=0.0,
                zorder=2.7,
            )
            axis.scatter(
                [point.utility],
                [point.diversity],
                s=PLOT_MARKER_AREA,
                marker=model.marker,
                facecolor=model.color,
                edgecolor="white",
                linewidths=1.0,
                alpha=alpha,
                zorder=3.0,
            )


def _main_legend_handles(panels: tuple[PanelSpec, ...]) -> list[Line2D]:
    unique = {}
    for panel in panels:
        for model in panel.models:
            unique.setdefault(model.label, model)
    handles = []
    for label in ("Original", "GRPO", "SGRPO", "Memory-Assisted GRPO"):
        model = unique.get(label)
        if model is None:
            continue
        handles.append(
            Line2D(
                [0],
                [0],
                color=model.color,
                linewidth=2.8,
                marker=model.marker,
                markersize=9.5,
                markerfacecolor=model.color,
                markeredgecolor="white",
                markeredgewidth=1.1,
                label=label,
            )
        )
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                color=REFERENCE_MARKER_COLOR,
                marker="x",
                markersize=11,
                linestyle="None",
                markeredgewidth=2.0,
                label="Ref. Point of HV",
            ),
            Line2D(
                [0],
                [0],
                color="#4A4A4A",
                linewidth=2.4,
                linestyle=FRONTIER_LINESTYLE,
                label="Frontier",
            ),
        ]
    )
    return handles


def plot_figure2(
    store: ResultStore,
    output_path: Path,
    panels: tuple[PanelSpec, ...],
) -> None:
    configure_style(UNIFORM_FONT_SIZE)
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(6.2 * len(panels), 6.1),
        squeeze=False,
    )
    figure.patch.set_facecolor("white")
    for axis, panel in zip(axes[0], panels):
        _draw_main_panel(axis, store, panel)
    figure.supxlabel("Utility", y=0.086)
    figure.supylabel("Diversity", x=0.017)
    figure.legend(
        handles=_main_legend_handles(panels),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=6,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.25,
        handletextpad=0.55,
    )
    figure.subplots_adjust(left=0.105, right=0.995, top=0.88, bottom=0.23, wspace=0.18)
    figure.savefig(output_path)
    plt.close(figure)


def plot_figure3(store: ResultStore, output_path: Path) -> None:
    configure_style(SMALL_PANEL_FONT_SIZE)
    panel = MAIN_PANELS[0]
    figure, axis = plt.subplots(figsize=(7.4, 5.8))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("#FAFAF8")
    all_utility = []
    all_diversity = []
    handles = []
    for model in ABLATION_MODELS:
        aggregate = aggregate_series(store, panel, model)
        utilities = [point.utility_mean for point in aggregate]
        diversities = [point.diversity_mean for point in aggregate]
        all_utility.extend(utilities)
        all_diversity.extend(diversities)
        axis.plot(
            utilities,
            diversities,
            color=model.color,
            linewidth=2.8,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=2.8,
        )
        for point in aggregate:
            axis.errorbar(
                point.utility_mean,
                point.diversity_mean,
                xerr=point.utility_ci95,
                yerr=point.diversity_ci95,
                fmt="none",
                ecolor=model.color,
                elinewidth=ERRORBAR_LINEWIDTH,
                capsize=0.0,
                zorder=2.9,
            )
        axis.scatter(
            utilities,
            diversities,
            s=PLOT_MARKER_AREA,
            marker=model.marker,
            facecolor=model.color,
            edgecolor="white",
            linewidths=1.0,
            zorder=3.2,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                color=model.color,
                linewidth=2.8,
                marker=model.marker,
                markersize=9.5,
                markerfacecolor=model.color,
                markeredgecolor="white",
                markeredgewidth=1.1,
                label=model.label,
            )
        )

    utility_span = max(all_utility) - min(all_utility)
    diversity_span = max(all_diversity) - min(all_diversity)
    axis.set_xlim(
        min(all_utility) - max(0.015, utility_span * 0.08),
        max(all_utility) + max(0.01, utility_span * 0.05),
    )
    axis.set_ylim(
        min(all_diversity) - max(0.05, diversity_span * 0.12),
        min(1.0, max(all_diversity) + max(0.05, diversity_span * 0.12)),
    )
    axis.set_xlabel("Utility")
    axis.set_ylabel("Diversity")
    axis.grid(True, color="#D8D8D8", linewidth=0.9, alpha=0.45)
    axis.legend(
        handles=handles,
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#D9D9D9",
        framealpha=0.92,
        borderpad=0.6,
        handlelength=2.2,
        handletextpad=0.6,
    )
    axis.text(
        0.03,
        0.98,
        "High Rand. & Temp.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        color="#555555",
        bbox={"facecolor": "#FAFAF8", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        zorder=5,
    )
    axis.text(
        0.96,
        0.02,
        "Low Rand. & Temp.",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color="#555555",
        bbox={"facecolor": "#FAFAF8", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
        zorder=5,
    )
    figure.savefig(output_path)
    plt.close(figure)


def _hyperparameter_hv(
    store: ResultStore, specs: tuple[CurveSpec, ...]
) -> tuple[list[float], list[float], list[float], Point]:
    all_points = [
        Point(point.utility, point.diversity)
        for seed in SEEDS
        for spec in specs
        for point in store.denovo_series(spec.experiment, seed)
    ]
    reference = Point(
        utility=min(point.utility for point in all_points),
        diversity=min(point.diversity for point in all_points),
    )
    x_values = []
    means = []
    ci95s = []
    for spec in specs:
        values = [
            compute_hv(
                [
                    Point(point.utility, point.diversity)
                    for point in store.denovo_series(spec.experiment, seed)
                ],
                reference,
            )
            for seed in SEEDS
        ]
        mean, ci95 = mean_ci95(values)
        x_values.append(spec.x_value)
        means.append(mean)
        ci95s.append(ci95)
    return x_values, means, ci95s, reference


def _plot_hv_curve(
    axis: plt.Axes,
    x_values: list[float],
    means: list[float],
    ci95s: list[float],
    color: str,
) -> None:
    axis.plot(
        x_values,
        means,
        color=color,
        linewidth=2.8,
        marker="o",
        markersize=7.8 / 3.0,
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3.0,
    )
    axis.errorbar(
        x_values,
        means,
        yerr=ci95s,
        fmt="none",
        ecolor=color,
        elinewidth=ERRORBAR_LINEWIDTH,
        capsize=0.0,
        zorder=2.9,
    )


def _annotate_grpo(
    axis: plt.Axes,
    x_value: float,
    y_value: float,
    text_x: float,
    text_y: float,
) -> None:
    axis.annotate(
        "GRPO",
        xy=(x_value, y_value),
        xytext=(text_x, text_y),
        ha="center",
        va="bottom",
        color="#333333",
        arrowprops={
            "arrowstyle": "->",
            "color": "#555555",
            "lw": 1.5,
            "shrinkA": 2.0,
            "shrinkB": 4.0,
        },
        zorder=5,
    )


def plot_figure5(
    store: ResultStore, output_path: Path
) -> tuple[tuple[list[float], list[float], list[float], Point], ...]:
    configure_style(22)
    group_result = _hyperparameter_hv(store, GROUP_SIZE_SPECS)
    weight_result = _hyperparameter_hv(store, DIVERSITY_WEIGHT_SPECS)
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 3.8))
    figure.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("#FAFAF8")
        axis.grid(True, color="#D8D8D8", linewidth=0.9, alpha=0.45)
        axis.set_box_aspect(0.5)

    group_x, group_mean, group_ci95, _ = group_result
    weight_x, weight_mean, weight_ci95, _ = weight_result
    _plot_hv_curve(axes[0], group_x, group_mean, group_ci95, COLOR_GROUP_SIZE)
    _plot_hv_curve(axes[1], weight_x, weight_mean, weight_ci95, COLOR_DIVERSITY_WEIGHT)
    axes[0].set_xlabel("Group Size")
    axes[0].set_ylabel("Hypervolume")
    axes[0].set_xticks(group_x)
    axes[1].set_xlabel("Weight of Diversity Rewards")
    axes[1].set_ylabel("Hypervolume")
    axes[1].set_xticks(weight_x)
    axes[1].set_xticklabels(["0", "0.1", "0.3", "0.5", "0.7", "0.9"])
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    _annotate_grpo(
        axes[0],
        group_x[0],
        group_mean[0],
        text_x=10.7,
        text_y=group_mean[0] + 0.0006,
    )
    _annotate_grpo(
        axes[1],
        weight_x[0],
        weight_mean[0],
        text_x=0.13,
        text_y=weight_mean[0] + 0.0011,
    )
    figure.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.98, wspace=0.20)
    figure.savefig(output_path)
    plt.close(figure)
    return group_result, weight_result


def _format_interval(mean: float, ci95: float) -> str:
    return f"{mean:.4f} ± {ci95:.4f}"


def _format_point(value: float) -> str:
    return f"{value:.4f}"


def _append_main_metric_tables(
    lines: list[str],
    metrics: dict[str, dict[str, tuple[float, float]]],
    panels: tuple[PanelSpec, ...],
) -> None:
    lines.extend(
        [
            "## Table 1: Frontier Metrics",
            "",
            "Each cell is the five-run mean ± 95% confidence interval.",
            "",
        ]
    )
    for panel in panels:
        model_by_label = {model.label: model for model in panel.models}
        ordered_labels = ["Original", "GRPO"]
        if "Memory-Assisted GRPO" in model_by_label:
            ordered_labels.append("Memory-Assisted GRPO")
        ordered_labels.append("SGRPO")
        ordered_models = [model_by_label[label] for label in ordered_labels]
        header_labels = [
            "Mem-GRPO" if model.label == "Memory-Assisted GRPO" else model.label
            for model in ordered_models
        ]
        lines.extend(
            [
                f"### {panel.title}",
                "",
                "| Metric | " + " | ".join(header_labels) + " |",
                "|---|" + "|".join("---:" for _ in ordered_models) + "|",
            ]
        )
        for metric in ("HV ↑", "DIP ↓", "R2 ↓"):
            key = metric.split()[0]
            values = [
                _format_interval(*metrics[panel.key][f"{model.label}:{key}"])
                for model in ordered_models
            ]
            lines.append(f"| {metric} | " + " | ".join(values) + " |")
        lines.append("")


def _append_reference_table(
    lines: list[str],
    references: dict[str, list[Point]],
    panels: tuple[PanelSpec, ...],
) -> None:
    lines.extend(
        [
            "### Per-Run HV Reference Points",
            "",
            "| Task | Seed | Utility reference | Diversity reference |",
            "|---|---:|---:|---:|",
        ]
    )
    panel_by_key = {panel.key: panel for panel in panels}
    for panel_key, points in references.items():
        for seed, point in zip(SEEDS, points):
            lines.append(
                f"| {panel_by_key[panel_key].title} | {seed} | "
                f"{point.utility:.4f} | {point.diversity:.4f} |"
            )
    lines.append("")


def _append_figure2_tables(
    lines: list[str],
    store: ResultStore,
    panels: tuple[PanelSpec, ...],
) -> None:
    lines.extend(
        [
            "## Figure 2: Utility-Diversity Operating Points",
            "",
            "[PDF](figure2.pdf)",
            "",
        ]
    )
    for panel in panels:
        lines.extend(
            [
                f"### {panel.title}",
                "",
                "| Model | Sweep point | Utility mean | Utility 95% CI | Diversity mean | Diversity 95% CI |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for model in panel.models:
            for point in aggregate_series(store, panel, model):
                lines.append(
                    f"| {model.label} | {point.point_label} | "
                    f"{_format_point(point.utility_mean)} | {_format_point(point.utility_ci95)} | "
                    f"{_format_point(point.diversity_mean)} | {_format_point(point.diversity_ci95)} |"
                )
        lines.append("")


def _append_figure3_table(lines: list[str], store: ResultStore) -> None:
    panel = MAIN_PANELS[0]
    lines.extend(
        [
            "## Figure 3: Ablation",
            "",
            "[PDF](figure3.pdf)",
            "",
            "| Model | Sweep point | Utility mean | Utility 95% CI | Diversity mean | Diversity 95% CI |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for model in ABLATION_MODELS:
        for point in aggregate_series(store, panel, model):
            lines.append(
                f"| {model.label} | {point.point_label} | "
                f"{_format_point(point.utility_mean)} | {_format_point(point.utility_ci95)} | "
                f"{_format_point(point.diversity_mean)} | {_format_point(point.diversity_ci95)} |"
            )
    lines.append("")


def _append_figure5_tables(
    lines: list[str],
    group_result: tuple[list[float], list[float], list[float], Point],
    weight_result: tuple[list[float], list[float], list[float], Point],
) -> None:
    lines.extend(
        [
            "## Figure 5: Hyperparameter Analysis",
            "",
            "[PDF](figure5.pdf)",
            "",
            "### Group Size",
            "",
            "| Group size | HV mean | HV 95% CI |",
            "|---:|---:|---:|",
        ]
    )
    group_x, group_mean, group_ci95, group_reference = group_result
    for x_value, mean, ci95 in zip(group_x, group_mean, group_ci95):
        lines.append(f"| {x_value:g} | {mean:.4f} | {ci95:.4f} |")
    lines.extend(
        [
            "",
            f"Global panel reference point: `({_format_point(group_reference.utility)}, "
            f"{_format_point(group_reference.diversity)})`.",
            "",
            "### Weight of Diversity Rewards",
            "",
            "| Weight | HV mean | HV 95% CI |",
            "|---:|---:|---:|",
        ]
    )
    weight_x, weight_mean, weight_ci95, weight_reference = weight_result
    for x_value, mean, ci95 in zip(weight_x, weight_mean, weight_ci95):
        lines.append(f"| {x_value:g} | {mean:.4f} | {ci95:.4f} |")
    lines.extend(
        [
            "",
            f"Global panel reference point: `({_format_point(weight_reference.utility)}, "
            f"{_format_point(weight_reference.diversity)})`.",
            "",
        ]
    )


def write_markdown(
    output_path: Path,
    store: ResultStore,
    metrics: dict[str, dict[str, tuple[float, float]]],
    references: dict[str, list[Point]],
    group_result: tuple[list[float], list[float], list[float], Point],
    weight_result: tuple[list[float], list[float], list[float], Point],
    panels: tuple[PanelSpec, ...],
) -> None:
    molecule_points = ", ".join(
        f"({randomness:g}, {temperature:g})"
        for randomness, temperature in MOLECULE_SWEEP
    )
    lines = [
        "# Expanded Sweeping Evaluation Results",
        "",
        "These are new five-run results generated with the expanded sweep grids. "
        "No paper or Overleaf source is modified.",
        "",
        "## Evaluation Protocol",
        "",
        "| Task family | Sweep points | Runs | Samples per model and point |",
        "|---|---|---:|---:|",
        f"| GenMol de novo | `{molecule_points}` | 5 | 1,000 |",
        f"| GenMol-P / mmGenMol | `{molecule_points}` | 5 | 1,600 |",
        "",
    ]
    _append_main_metric_tables(lines, metrics, panels)
    _append_reference_table(lines, references, panels)
    _append_figure2_tables(lines, store, panels)
    _append_figure3_table(lines, store)
    _append_figure5_tables(lines, group_result, weight_result)
    output_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_output_dir = project_root / "nips26/rebuttal"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_output_dir / "sweep-results",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = REBUTTAL_PANELS
    store = ResultStore(
        args.results_root,
        tuple(panel.source_kind for panel in panels),
    )
    validate_seed_independence(store, panels)
    metrics, references = table1_metrics(store, panels)
    figure2_path = output_dir / "figure2.pdf"
    figure3_path = output_dir / "figure3.pdf"
    figure5_path = output_dir / "figure5.pdf"
    markdown_path = output_dir / "expanded-sweep-results.md"
    plot_figure2(store, figure2_path, panels)
    plot_figure3(store, figure3_path)
    group_result, weight_result = plot_figure5(store, figure5_path)
    write_markdown(
        markdown_path,
        store,
        metrics,
        references,
        group_result,
        weight_result,
        panels,
    )
    print(figure2_path)
    print(figure3_path)
    print(figure5_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
