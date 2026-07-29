#!/usr/bin/env python3
"""Validate and render motif-extension Table 1 and Figure 2 results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VIS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = VIS_ROOT.parent
sys.path.insert(0, str(VIS_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
import render_rebuttal_dense_results as dense  # noqa: E402
from genmol.diversity import (  # noqa: E402
    MORGAN_INTERNAL_DIVERSITY,
    compute_molecular_diversity,
)
from genmol.rl.motif import load_test_motif_records  # noqa: E402


COLOR_DMB = "#6B5B95"
COLOR_ENTROPY = "#C85A3E"


def _panel_for_checkpoint_step(checkpoint_step: int) -> dense.PanelSpec:
    if checkpoint_step <= 0:
        raise ValueError("checkpoint_step must be positive")
    return dense.PanelSpec(
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
            dense.ModelSpec(
                f"motif_grpo_{checkpoint_step}",
                "GRPO",
                dense.COLOR_GRPO,
                "^",
            ),
            dense.ModelSpec(
                f"motif_dmb_{checkpoint_step}",
                "Diverse Mini-Batch GRPO",
                COLOR_DMB,
                "P",
            ),
            dense.ModelSpec(
                f"motif_entropy_{checkpoint_step}",
                "Entropy-Regularized GRPO",
                COLOR_ENTROPY,
                "X",
            ),
            dense.ModelSpec(
                f"motif_sgrpo_{checkpoint_step}",
                "SGRPO",
                dense.COLOR_SGRPO,
                "o",
            ),
        ),
    )


PANEL = _panel_for_checkpoint_step(2000)


def _section_markers(checkpoint_step: int) -> tuple[str, str]:
    suffix = "" if checkpoint_step == 2000 else f" CHECKPOINT {checkpoint_step}"
    return (
        f"<!-- BEGIN MOTIF EXTENSION{suffix} RESULTS -->",
        f"<!-- END MOTIF EXTENSION{suffix} RESULTS -->",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_mean(values: list[object], *, context: str) -> float:
    finite = []
    for value in values:
        if value is None:
            continue
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite value for {context}: {value!r}")
        finite.append(parsed)
    if not finite:
        raise ValueError(f"no finite values for {context}")
    return sum(finite) / len(finite)


def _assert_close(
    actual: object,
    expected: float,
    *,
    context: str,
) -> None:
    parsed = float(actual)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite aggregate for {context}: {actual!r}")
    if not math.isclose(parsed, expected, rel_tol=1.0e-10, abs_tol=1.0e-12):
        raise ValueError(
            f"raw-to-summary mismatch for {context}: "
            f"summary={parsed:.17g} raw={expected:.17g}"
        )


def _recompute_point_summary(
    rows_by_motif: dict[int, list[dict]],
    *,
    context: str,
) -> dict[str, object]:
    if set(rows_by_motif) != set(range(10)):
        raise ValueError(f"incomplete motif coverage for {context}")
    ordered_rows = []
    per_motif_diversity = []
    per_motif_retention = []
    for motif_index in range(10):
        motif_rows = sorted(
            rows_by_motif[motif_index],
            key=lambda row: int(row["sample_index"]),
        )
        if len(motif_rows) != 100:
            raise ValueError(
                f"expected 100 rows for {context}, motif={motif_index}"
            )
        ordered_rows.extend(motif_rows)
        per_motif_diversity.append(
            compute_molecular_diversity(
                [row["smiles"] for row in motif_rows],
                metric=MORGAN_INTERNAL_DIVERSITY,
            )
        )
        per_motif_retention.append(
            sum(row["motif_retained"] for row in motif_rows) / 100
        )

    return {
        "soft_reward_mean": _finite_mean(
            [row["soft_reward"] for row in ordered_rows],
            context=f"{context} soft_reward",
        ),
        "qed_mean": _finite_mean(
            [row["qed"] for row in ordered_rows],
            context=f"{context} qed",
        ),
        "sa_mean": _finite_mean(
            [row["sa"] for row in ordered_rows],
            context=f"{context} sa",
        ),
        "sa_score_mean": _finite_mean(
            [row["sa_score"] for row in ordered_rows],
            context=f"{context} sa_score",
        ),
        "diversity": sum(per_motif_diversity) / len(per_motif_diversity),
        "per_motif_diversity": per_motif_diversity,
        "raw_valid_fraction": (
            sum(row["raw_smiles"] is not None for row in ordered_rows)
            / len(ordered_rows)
        ),
        "motif_retention_fraction": (
            sum(row["motif_retained"] for row in ordered_rows)
            / len(ordered_rows)
        ),
        "per_motif_retention_fraction": per_motif_retention,
        "task_valid_fraction": (
            sum(row["is_valid"] for row in ordered_rows) / len(ordered_rows)
        ),
        "alert_hit_fraction": (
            sum(row["alert_hit"] for row in ordered_rows) / len(ordered_rows)
        ),
    }


def _validate_point_summary(
    summary_row: dict,
    rows_by_motif: dict[int, list[dict]],
    *,
    model: dense.ModelSpec,
    seed: int,
    point_index: int,
    checkpoint_path: str,
) -> None:
    context = f"{model.source_id}, seed={seed}, point={point_index}"
    randomness, temperature = dense.MOLECULE_SWEEP[point_index]
    expected_exact = {
        "experiment": model.source_id,
        "display_name": model.label,
        "checkpoint_path": checkpoint_path,
        "seed": seed,
        "point_index": point_index,
        "sweep_axis": "randomness_temperature_pair",
        "sweep_label": f"r={randomness:.1f},t={temperature:g}",
        "num_motifs": 10,
        "samples_per_motif": 100,
        "num_samples": 1_000,
        "diversity_metric": MORGAN_INTERNAL_DIVERSITY,
    }
    for field, expected in expected_exact.items():
        if summary_row.get(field) != expected:
            raise ValueError(
                f"unexpected {field} for {context}: "
                f"{summary_row.get(field)!r} vs {expected!r}"
            )
    _assert_close(
        summary_row.get("randomness"),
        randomness,
        context=f"{context} randomness",
    )
    _assert_close(
        summary_row.get("generation_temperature"),
        temperature,
        context=f"{context} generation_temperature",
    )
    _assert_close(
        summary_row.get("sweep_value"),
        float(point_index + 1),
        context=f"{context} sweep_value",
    )

    recomputed = _recompute_point_summary(
        rows_by_motif,
        context=context,
    )
    scalar_fields = (
        "soft_reward_mean",
        "qed_mean",
        "sa_mean",
        "sa_score_mean",
        "diversity",
        "raw_valid_fraction",
        "motif_retention_fraction",
        "task_valid_fraction",
        "alert_hit_fraction",
    )
    for field in scalar_fields:
        _assert_close(
            summary_row.get(field),
            float(recomputed[field]),
            context=f"{context} {field}",
        )
    for field in (
        "per_motif_diversity",
        "per_motif_retention_fraction",
    ):
        actual = summary_row.get(field)
        expected = recomputed[field]
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(
                f"unexpected {field} shape for {context}: "
                f"{type(actual).__name__}"
            )
        for motif_index, (actual_value, expected_value) in enumerate(
            zip(actual, expected)
        ):
            _assert_close(
                actual_value,
                float(expected_value),
                context=f"{context} {field}[{motif_index}]",
            )


class MotifResultStore(dense.ResultStore):
    def __init__(
        self,
        run_root: Path,
        *,
        panel: dense.PanelSpec | None = None,
    ):
        self.results_root = run_root
        self.panel = PANEL if panel is None else panel
        self._cache = {}
        self._row_hashes = {}
        self._test_motifs = load_test_motif_records(
            REPO_ROOT / "data/fragments.csv"
        )
        manifest_path = run_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"missing motif sweep manifest: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text())
        expected_task_count = len(self.panel.models) * len(dense.SEEDS)
        if manifest.get("task_count") != expected_task_count:
            raise ValueError(
                "motif sweep manifest task count mismatch: "
                f"{manifest.get('task_count')!r} vs {expected_task_count}"
            )
        if manifest.get("seeds") != list(dense.SEEDS):
            raise ValueError("motif sweep manifest has unexpected seeds")
        tasks = manifest.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != expected_task_count:
            raise ValueError("motif sweep manifest has invalid tasks")
        self._tasks = {}
        for task in tasks:
            key = (task.get("experiment"), int(task.get("seed")))
            if key in self._tasks:
                raise ValueError(f"duplicate motif sweep manifest task: {key}")
            self._tasks[key] = task
        expected_keys = {
            (model.source_id, seed)
            for model in self.panel.models
            for seed in dense.SEEDS
        }
        if set(self._tasks) != expected_keys:
            raise ValueError("motif sweep manifest task coverage is incomplete")

    def rows(self, kind: str, seed: int) -> list[dict]:
        if kind != "denovo":
            raise ValueError(f"unsupported motif result kind: {kind}")
        if seed in self._cache:
            return self._cache[seed]
        rows = []
        for model in self.panel.models:
            task = self._tasks[(model.source_id, seed)]
            if task.get("display_name") != model.label:
                raise ValueError(
                    f"manifest display name mismatch for {model.source_id}"
                )
            checkpoint_path = task.get("checkpoint_path")
            if not isinstance(checkpoint_path, str) or not checkpoint_path:
                raise ValueError(
                    f"manifest checkpoint path is invalid for {model.source_id}"
                )
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
            if metadata.get("display_name") != model.label:
                raise ValueError(f"display-name mismatch in {summary_path}")
            if metadata.get("checkpoint_path") != checkpoint_path:
                raise ValueError(f"checkpoint mismatch in {summary_path}")
            if int(metadata.get("checkpoint_size_bytes", 0)) <= 0:
                raise ValueError(
                    f"invalid checkpoint size in {summary_path}"
                )
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
            expected_generation_metadata = {
                "min_add_len": 18,
                "gamma": 0.3,
                "guidance_weight": 2.0,
                "generation_batch_size": 1_000,
            }
            for field, expected in expected_generation_metadata.items():
                if metadata.get(field) != expected:
                    raise ValueError(
                        f"unexpected {field} in {summary_path}: "
                        f"{metadata.get(field)!r}"
                    )
            self._row_hashes[(model.source_id, seed)] = metadata[
                "rows_sha256"
            ]
            counts = Counter()
            sample_indices = {}
            raw_rows_by_point_motif = {}
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
                    if row.get("display_name") != model.label:
                        raise ValueError(
                            f"display-name mismatch at "
                            f"{rows_path}:{line_number}"
                        )
                    if row.get("checkpoint_path") != checkpoint_path:
                        raise ValueError(
                            f"checkpoint mismatch at "
                            f"{rows_path}:{line_number}"
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
                    expected_motif = self._test_motifs[motif_index]
                    if (
                        row.get("motif_id") != expected_motif.motif_id
                        or row.get("motif_smiles") != expected_motif.smiles
                    ):
                        raise ValueError(
                            f"official motif mismatch at "
                            f"{rows_path}:{line_number}"
                        )
                    for field in (
                        "motif_retained",
                        "is_valid",
                        "alert_hit",
                    ):
                        if not isinstance(row.get(field), bool):
                            raise TypeError(
                                f"{field} must be boolean at "
                                f"{rows_path}:{line_number}"
                            )
                    if row["motif_retained"] and row["raw_smiles"] is None:
                        raise ValueError(
                            f"retained motif has no raw molecule at "
                            f"{rows_path}:{line_number}"
                        )
                    if row["is_valid"] and row["smiles"] is None:
                        raise ValueError(
                            f"task-valid record has no scored molecule at "
                            f"{rows_path}:{line_number}"
                        )
                    if row["is_valid"] and not row["motif_retained"]:
                        raise ValueError(
                            f"task-valid molecule did not retain motif at "
                            f"{rows_path}:{line_number}"
                        )
                    for field in ("qed", "sa", "sa_score", "soft_reward"):
                        value = row.get(field)
                        if value is not None and not math.isfinite(float(value)):
                            raise ValueError(
                                f"non-finite {field} at "
                                f"{rows_path}:{line_number}"
                            )
                    counts[key] += 1
                    sample_indices.setdefault(key, set()).add(sample_index)
                    raw_rows_by_point_motif.setdefault(key, []).append(row)
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
                _validate_point_summary(
                    row,
                    {
                        motif_index: raw_rows_by_point_motif[
                            (point_index, motif_index)
                        ]
                        for motif_index in range(10)
                    },
                    model=model,
                    seed=seed,
                    point_index=point_index,
                    checkpoint_path=checkpoint_path,
                )
            rows.extend(result_rows)
        self._cache[seed] = rows
        return rows

    def validate_raw_seed_independence(self) -> None:
        for seed in dense.SEEDS:
            self.rows("denovo", seed)
        for model in self.panel.models:
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


def _section(store, metrics, references, figure_name, checkpoint_step):
    points = ", ".join(
        f"({randomness:g}, {temperature:g})"
        for randomness, temperature in dense.MOLECULE_SWEEP
    )
    lines = [
        (
            "## GenMol Motif Extension"
            if checkpoint_step == 2000
            else f"## GenMol Motif Extension: Checkpoint {checkpoint_step}"
        ),
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


def _upsert(path: Path, section: str, checkpoint_step: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"expanded result Markdown not found: {path}")
    section_begin, section_end = _section_markers(checkpoint_step)
    text = path.read_text()
    begin_count = text.count(section_begin)
    end_count = text.count(section_end)
    if begin_count != end_count or begin_count > 1:
        raise RuntimeError(
            f"malformed motif result markers in {path}: "
            f"begin={begin_count} end={end_count}"
        )
    wrapped = f"{section_begin}\n{section.rstrip()}\n{section_end}\n"
    if begin_count:
        prefix, remainder = text.split(section_begin, 1)
        _, suffix = remainder.split(section_end, 1)
        updated = prefix.rstrip() + "\n\n" + wrapped + suffix.lstrip()
    else:
        updated = text.rstrip() + "\n\n" + wrapped
    path.write_text(updated)


def main() -> None:
    global PANEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expanded-results-path", type=Path)
    parser.add_argument("--checkpoint-step", type=int)
    args = parser.parse_args()
    manifest = json.loads((args.run_root / "manifest.json").read_text())
    manifest_step = int(manifest.get("checkpoint_step", 2000))
    checkpoint_step = (
        manifest_step if args.checkpoint_step is None else args.checkpoint_step
    )
    if checkpoint_step != manifest_step:
        raise ValueError(
            "checkpoint step does not match manifest: "
            f"argument={checkpoint_step} manifest={manifest_step}"
        )
    PANEL = _panel_for_checkpoint_step(checkpoint_step)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = MotifResultStore(args.run_root)
    store.validate_raw_seed_independence()
    dense.validate_seed_independence(
        store,
        (PANEL,),
        include_denovo_auxiliary=False,
    )
    metrics, references = dense.table1_metrics(store, (PANEL,))
    name_suffix = "" if checkpoint_step == 2000 else f"-checkpoint-{checkpoint_step}"
    figure_path = args.output_dir / f"figure2-motif-extension{name_suffix}.pdf"
    _plot_figure2(store, figure_path)
    section = _section(
        store,
        metrics,
        references,
        figure_path.name,
        checkpoint_step,
    )
    standalone = (
        args.output_dir / f"motif-extension-results{name_suffix}.md"
    )
    standalone.write_text("# Motif-Extension Results\n\n" + section)
    if args.expanded_results_path is not None:
        _upsert(args.expanded_results_path, section, checkpoint_step)
    print(figure_path)
    print(standalone)
    if args.expanded_results_path is not None:
        print(args.expanded_results_path)


if __name__ == "__main__":
    main()
