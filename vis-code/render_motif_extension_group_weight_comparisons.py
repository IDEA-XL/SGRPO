#!/usr/bin/env python3
"""Render separate checkpoint-1000 motif SGRPO weight comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


VIS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(VIS_ROOT))
import render_motif_extension_results as motif  # noqa: E402
import render_rebuttal_dense_results as dense  # noqa: E402


CHECKPOINT_STEP = 1000
BASE_PANEL = motif._panel_for_checkpoint_step(CHECKPOINT_STEP)
BASE_MODELS = BASE_PANEL.models[:4]
VARIANT_MODELS = {
    "gw05": dense.ModelSpec(
        "motif_sgrpo_gw05_1000",
        "SGRPO (w=0.5)",
        dense.COLOR_SGRPO,
        "o",
    ),
    "gw01": dense.ModelSpec(
        "motif_sgrpo_gw01_1000",
        "SGRPO (w=0.1)",
        dense.COLOR_SGRPO,
        "o",
    ),
}


class CombinedStore(dense.ResultStore):
    def __init__(
        self,
        base_store: motif.MotifResultStore,
        variant_store: motif.MotifResultStore,
    ):
        self.base_store = base_store
        self.variant_store = variant_store
        self._cache = {}
        self._base_ids = {model.source_id for model in BASE_MODELS}

    def rows(self, kind: str, seed: int) -> list[dict]:
        if kind != "denovo":
            raise ValueError(f"unsupported result kind: {kind}")
        if seed not in self._cache:
            base_rows = [
                row
                for row in self.base_store.rows(kind, seed)
                if row["experiment"] in self._base_ids
            ]
            variant_rows = self.variant_store.rows(kind, seed)
            self._cache[seed] = base_rows + variant_rows
        return self._cache[seed]


def _single_model_panel(
    key: str,
    model: dense.ModelSpec,
) -> dense.PanelSpec:
    return dense.PanelSpec(
        key=key,
        title="Motif Extension",
        source_kind="denovo",
        models=(model,),
    )


def _comparison_panel(
    key: str,
    model: dense.ModelSpec,
) -> dense.PanelSpec:
    return dense.PanelSpec(
        key=key,
        title="Motif Extension",
        source_kind="denovo",
        models=BASE_MODELS + (model,),
    )


def _format_section(
    panel: dense.PanelSpec,
    metrics: dict[str, dict[str, tuple[float, float]]],
    references: dict[str, list[dense.Point]],
) -> str:
    reference_text = ", ".join(
        f"seed{seed}=({point.utility:.4f}, {point.diversity:.4f})"
        for seed, point in zip(dense.SEEDS, references[panel.key])
    )
    lines = [
        f"## {panel.models[-1].label}",
        "",
        f"Per-seed HV reference points: {reference_text}.",
        "",
        "| Metric | "
        + " | ".join(model.label for model in panel.models)
        + " |",
        "|---|" + "|".join("---:" for _ in panel.models) + "|",
    ]
    for metric, direction in (("HV", "↑"), ("DIP", "↓"), ("R2", "↓")):
        values = []
        for model in panel.models:
            mean, ci95 = metrics[panel.key][f"{model.label}:{metric}"]
            values.append(f"{mean:.4f} ± {ci95:.4f}")
        lines.append(
            f"| {metric} {direction} | " + " | ".join(values) + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run-root", type=Path, required=True)
    parser.add_argument("--gw05-run-root", type=Path, required=True)
    parser.add_argument("--gw01-run-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    base_store = motif.MotifResultStore(
        args.base_run_root,
        panel=BASE_PANEL,
    )
    base_store.validate_raw_seed_independence()

    sections = []
    for key, run_root in (
        ("gw05", args.gw05_run_root),
        ("gw01", args.gw01_run_root),
    ):
        model = VARIANT_MODELS[key]
        validation_panel = _single_model_panel(key, model)
        variant_store = motif.MotifResultStore(
            run_root,
            panel=validation_panel,
        )
        variant_store.validate_raw_seed_independence()
        panel = _comparison_panel(key, model)
        combined_store = CombinedStore(base_store, variant_store)
        dense.validate_seed_independence(
            combined_store,
            (panel,),
            include_denovo_auxiliary=False,
        )
        metrics, references = dense.table1_metrics(
            combined_store,
            (panel,),
        )
        sections.append(_format_section(panel, metrics, references))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        "# Motif-Extension SGRPO Group-Weight Comparisons\n\n"
        + "\n\n".join(sections)
        + "\n"
    )
    print(args.output_path)


if __name__ == "__main__":
    main()
