#!/usr/bin/env python3
"""Append validated mmGenMol 100-per-pocket tables to the rebuttal results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from render_rebuttal_dense_results import (  # noqa: E402
    MAIN_PANELS,
    SEEDS,
    ResultStore,
    aggregate_series,
    table1_metrics,
    validate_seed_independence,
)


BLOCK_START = '<!-- BEGIN MMGENMOL 100X RESULTS -->'
BLOCK_END = '<!-- END MMGENMOL 100X RESULTS -->'


def _format_interval(mean: float, ci95: float) -> str:
    return f'{mean:.4f} ± {ci95:.4f}'


def _validate_protocol(results_root: Path) -> None:
    expected_models = {
        'original_5500',
        'grpo_unidock_1000',
        'sgrpo_unidock_rewardsum_loo_1000',
    }
    for seed in SEEDS:
        path = results_root / 'mmgenmol' / f'seed{seed}.json'
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or len(rows) != 30:
            raise ValueError(f'Expected 30 rows in {path}')
        if {str(row['model_name']) for row in rows} != expected_models:
            raise ValueError(f'Unexpected model set in {path}')
        for row in rows:
            if int(row['num_rows']) != 10000:
                raise ValueError(f"Expected num_rows=10000 in {path}: {row['task_id']}")
            if int(row['num_pockets']) != 100:
                raise ValueError(f"Expected num_pockets=100 in {path}: {row['task_id']}")
            if int(row['samples_per_pocket']) != 100:
                raise ValueError(
                    f"Expected samples_per_pocket=100 in {path}: {row['task_id']}"
                )


def _render_block(results_root: Path) -> str:
    _validate_protocol(results_root)
    panel = MAIN_PANELS[1]
    store = ResultStore(results_root, ('mmgenmol',))
    validate_seed_independence(store, (panel,))
    metrics, references = table1_metrics(store, (panel,))
    metric_values = metrics[panel.key]

    lines = [
        BLOCK_START,
        '## Pocket-Based Design with 100 Generations per Pocket',
        '',
        '| Sweep points | Runs | Pockets per point | Generations per pocket | Samples per point |',
        '|---|---:|---:|---:|---:|',
        '| `(0.1, 0.5), (0.2, 0.65), (0.3, 0.8), (0.4, 0.95), (0.5, 1.1), (0.6, 1.25), (0.7, 1.4), (0.8, 1.55), (0.9, 1.7), (1.0, 2.0)` | 5 | 100 | 100 | 10,000 |',
        '',
        '### Table 1: Frontier Metrics',
        '',
        'Each cell is the five-run mean ± 95% confidence interval.',
        '',
        '| Metric | Original | GRPO | SGRPO |',
        '|---|---:|---:|---:|',
    ]
    for metric, direction in (('HV', '↑'), ('DIP', '↓'), ('R2', '↓')):
        values = [
            _format_interval(*metric_values[f'{model.label}:{metric}'])
            for model in panel.models
        ]
        lines.append(f"| {metric} {direction} | " + ' | '.join(values) + ' |')

    lines.extend(
        [
            '',
            '### Per-Run HV Reference Points',
            '',
            '| Seed | Utility reference | Diversity reference |',
            '|---:|---:|---:|',
        ]
    )
    for seed, reference in zip(SEEDS, references[panel.key]):
        lines.append(
            f'| {seed} | {reference.utility:.4f} | {reference.diversity:.4f} |'
        )

    lines.extend(
        [
            '',
            '### Figure 2: Utility-Diversity Operating Points',
            '',
            '| Model | Sweep point | Utility mean | Utility 95% CI | Diversity mean | Diversity 95% CI |',
            '|---|---|---:|---:|---:|---:|',
        ]
    )
    for model in panel.models:
        for point in aggregate_series(store, panel, model):
            lines.append(
                f'| {model.label} | {point.point_label} | '
                f'{point.utility_mean:.4f} | {point.utility_ci95:.4f} | '
                f'{point.diversity_mean:.4f} | {point.diversity_ci95:.4f} |'
            )
    lines.extend(['', BLOCK_END])
    return '\n'.join(lines)


def _replace_block(markdown: str, block: str) -> str:
    start_index = markdown.find(BLOCK_START)
    end_index = markdown.find(BLOCK_END)
    if (start_index == -1) != (end_index == -1):
        raise ValueError('Markdown contains only one mmGenMol 100x block marker')
    if start_index == -1:
        return markdown.rstrip() + '\n\n' + block + '\n'
    end_index += len(BLOCK_END)
    return markdown[:start_index].rstrip() + '\n\n' + block + markdown[end_index:]


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results-root',
        type=Path,
        default=project_root / 'nips26/rebuttal/sweep-results-mmgenmol-100x',
    )
    parser.add_argument(
        '--markdown-path',
        type=Path,
        default=project_root / 'nips26/rebuttal/expanded-sweep-results.md',
    )
    args = parser.parse_args()
    if not args.markdown_path.is_file():
        raise FileNotFoundError(args.markdown_path)
    block = _render_block(args.results_root)
    updated = _replace_block(args.markdown_path.read_text(), block)
    args.markdown_path.write_text(updated)
    print(args.markdown_path)


if __name__ == '__main__':
    main()
