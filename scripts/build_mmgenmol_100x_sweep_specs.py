#!/usr/bin/env python3
"""Build five-seed mmGenMol sweep specs with 100 generations per pocket."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from build_rebuttal_dense_sweep_specs import (
    MMGENMOL_EXPERIMENTS,
    MOLECULE_SWEEP,
    REPO_REMOTE_ROOT,
    SEEDS,
)


DEFAULT_RUN_ROOT = Path(
    '/public/home/xinwuye/ai4s-tool-joint-train/runs/rebuttal_mmgenmol_100x_sweep'
)


def _write_tasks(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
            delimiter='\t',
            lineterminator='\n',
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    spec_root = args.run_root / 'specs'
    if spec_root.exists():
        raise FileExistsError(f'Spec root already exists: {spec_root}')

    for seed in SEEDS:
        rows = []
        task_id = 0
        for experiment in MMGENMOL_EXPERIMENTS:
            for point_index, (randomness, temperature) in enumerate(
                MOLECULE_SWEEP,
                start=1,
            ):
                rows.append(
                    {
                        'task_id': task_id,
                        'model_name': experiment['name'],
                        'sweep_type': 'paired',
                        'sweep_value': point_index,
                        'randomness': randomness,
                        'temperature': temperature,
                        'checkpoint_path': experiment['checkpoint_path'],
                        'output_path': (
                            args.run_root
                            / 'mmgenmol'
                            / f'seed{seed}'
                            / 'generation'
                            / experiment['name']
                            / f'paired_{point_index}'
                            / 'generated.rows.jsonl'
                        ),
                    }
                )
                task_id += 1
        _write_tasks(spec_root / 'mmgenmol' / f'seed{seed}.tsv', rows)

    manifest = {
        'repo_root': str(REPO_REMOTE_ROOT),
        'run_root': str(args.run_root),
        'seeds': list(SEEDS),
        'models': [experiment['name'] for experiment in MMGENMOL_EXPERIMENTS],
        'sweep_points': [list(point) for point in MOLECULE_SWEEP],
        'tasks_per_seed': len(MMGENMOL_EXPERIMENTS) * len(MOLECULE_SWEEP),
        'num_pockets': 100,
        'samples_per_pocket': 100,
        'rows_per_task': 10000,
        'generation_batch_size': 10000,
        'docking_shards_per_task': 10,
        'docking_cpus_per_shard': 32,
        'docking_memory_gb_per_shard': 64,
    }
    manifest_path = spec_root / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(manifest_path)


if __name__ == '__main__':
    main()
