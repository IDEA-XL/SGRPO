#!/usr/bin/env python3
"""Drive the five-seed, 100-samples-per-pocket mmGenMol sweep DAG."""

from __future__ import annotations

import csv
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import drive_rebuttal_dense_sweeps as engine
from build_rebuttal_dense_sweep_specs import REPO_REMOTE_ROOT, SEEDS


RUN_ROOT = Path(
    os.environ.get(
        'MMGENMOL_100X_RUN_ROOT',
        '/public/home/xinwuye/ai4s-tool-joint-train/runs/rebuttal_mmgenmol_100x_sweep',
    )
)
SPEC_ROOT = RUN_ROOT / 'specs'
LOG_ROOT = RUN_ROOT / 'logs'
STATE_PATH = RUN_ROOT / 'controller_state.json'
LOCK_PATH = RUN_ROOT / 'controller.lock'
COMPLETE_PATH = RUN_ROOT / 'COMPLETE'
DOCKING_CACHE_DIR = Path(
    '/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/'
    'pocket_prefix_crossdocked_5500ckpt/docking_cache'
)
DOCKING_SHARDS = 10
ROWS_PER_POINT = 10000
ROWS_PER_SHARD = 1000


def _read_tasks(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    if len(rows) != 30:
        raise ValueError(f'Expected 30 tasks in {path}, found {len(rows)}')
    expected_ids = list(range(30))
    observed_ids = [int(row['task_id']) for row in rows]
    if observed_ids != expected_ids:
        raise ValueError(f'Unexpected task IDs in {path}: {observed_ids}')
    return rows


def _build_dag():
    groups = {}
    tasks = {}
    terminal_keys = []
    for seed in SEEDS:
        tasks_path = SPEC_ROOT / 'mmgenmol' / f'seed{seed}.tsv'
        rows = _read_tasks(tasks_path)
        generation_group = f'generation_{seed}'
        docking_group = f'docking_{seed}'
        merge_group = f'merge_{seed}'
        aggregate_group = f'aggregate_{seed}'
        docking_root = RUN_ROOT / 'mmgenmol' / f'seed{seed}' / 'docking'
        aggregate_root = RUN_ROOT / 'mmgenmol' / f'seed{seed}' / 'aggregate'

        engine._add_group(
            groups,
            engine.GroupSpec(
                name=generation_group,
                resource='gpu',
                script=REPO_REMOTE_ROOT
                / 'scripts/slurm/rebuttal_mmgenmol_100x_generate_1gpu.sbatch',
                job_name=f'm100g{seed}',
                output_pattern=LOG_ROOT / f'generation_seed{seed}_%A_%a.out',
                error_pattern=LOG_ROOT / f'generation_seed{seed}_%A_%a.err',
                time_limit='02:00:00',
                exports=(('TASKS_PATH', str(tasks_path)), ('SEED', str(seed))),
            ),
        )
        engine._add_group(
            groups,
            engine.GroupSpec(
                name=docking_group,
                resource='cpu',
                script=REPO_REMOTE_ROOT
                / 'scripts/slurm/rebuttal_mmgenmol_100x_dock_shard_32cpu.sbatch',
                job_name=f'm100d{seed}',
                output_pattern=LOG_ROOT / f'docking_seed{seed}_%A_%a.out',
                error_pattern=LOG_ROOT / f'docking_seed{seed}_%A_%a.err',
                time_limit='04:00:00',
                exports=(
                    ('TASKS_PATH', str(tasks_path)),
                    ('OUTPUT_ROOT', str(docking_root)),
                    ('DOCKING_CACHE_DIR', str(DOCKING_CACHE_DIR)),
                ),
            ),
        )
        engine._add_group(
            groups,
            engine.GroupSpec(
                name=merge_group,
                resource='cpu',
                script=REPO_REMOTE_ROOT
                / 'scripts/slurm/rebuttal_mmgenmol_100x_merge_docking_cpu.sbatch',
                job_name=f'm100m{seed}',
                output_pattern=LOG_ROOT / f'merge_seed{seed}_%A_%a.out',
                error_pattern=LOG_ROOT / f'merge_seed{seed}_%A_%a.err',
                time_limit='00:30:00',
                exports=(
                    ('TASKS_PATH', str(tasks_path)),
                    ('DOCKING_ROOT', str(docking_root)),
                ),
            ),
        )
        engine._add_group(
            groups,
            engine.GroupSpec(
                name=aggregate_group,
                resource='cpu',
                script=REPO_REMOTE_ROOT
                / 'scripts/slurm/rebuttal_mmgenmol_100x_aggregate_cpu.sbatch',
                job_name=f'm100a{seed}',
                output_pattern=LOG_ROOT / f'aggregate_seed{seed}_%j.out',
                error_pattern=LOG_ROOT / f'aggregate_seed{seed}_%j.err',
                time_limit='06:00:00',
                exports=(
                    ('TASKS_PATH', str(tasks_path)),
                    ('DOCKING_ROOT', str(docking_root)),
                    ('OUTPUT_DIR', str(aggregate_root)),
                ),
            ),
        )

        merge_keys = []
        for row in rows:
            point_id = int(row['task_id'])
            model_name = row['model_name']
            sweep_leaf = f"{row['sweep_type']}_{row['sweep_value']}"
            generation_path = Path(row['output_path'])
            point_docking_root = docking_root / model_name / sweep_leaf
            generation_key = f'mm100:{seed}:generation:{point_id}'
            engine._add_task(
                tasks,
                engine.TaskSpec(
                    key=generation_key,
                    group=generation_group,
                    array_id=point_id,
                    prerequisites=(),
                    validator=lambda path=generation_path: engine._jsonl_has_rows(
                        path,
                        ROWS_PER_POINT,
                    ),
                ),
            )

            docking_keys = []
            for shard_index in range(DOCKING_SHARDS):
                array_id = point_id * DOCKING_SHARDS + shard_index
                shard_root = point_docking_root / 'shards' / f'shard_{shard_index:02d}'
                records_path = shard_root / 'docking.records.jsonl'
                summary_path = shard_root / 'docking.summary.json'
                docking_key = f'mm100:{seed}:docking:{point_id}:{shard_index}'
                docking_keys.append(docking_key)
                engine._add_task(
                    tasks,
                    engine.TaskSpec(
                        key=docking_key,
                        group=docking_group,
                        array_id=array_id,
                        prerequisites=(generation_key,),
                        validator=lambda records=records_path, summary=summary_path: (
                            engine._jsonl_has_rows(records, ROWS_PER_SHARD)
                            and engine._json_is_nonempty_dict(summary)
                        ),
                    ),
                )

            merge_key = f'mm100:{seed}:merge:{point_id}'
            merge_keys.append(merge_key)
            engine._add_task(
                tasks,
                engine.TaskSpec(
                    key=merge_key,
                    group=merge_group,
                    array_id=point_id,
                    prerequisites=tuple(docking_keys),
                    validator=lambda records=point_docking_root
                    / 'docking.records.jsonl', summary=point_docking_root
                    / 'docking.summary.json': (
                        engine._jsonl_has_rows(records, ROWS_PER_POINT)
                        and engine._json_is_nonempty_dict(summary)
                    ),
                ),
            )

        aggregate_key = f'mm100:{seed}:aggregate'
        terminal_keys.append(aggregate_key)
        engine._add_task(
            tasks,
            engine.TaskSpec(
                key=aggregate_key,
                group=aggregate_group,
                array_id=None,
                prerequisites=tuple(merge_keys),
                validator=lambda path=aggregate_root
                / 'mmgenmol_100x.json': engine._json_has_length(path, 30),
            ),
        )
    return groups, tasks, tuple(terminal_keys)


def _preflight() -> None:
    manifest_path = SPEC_ROOT / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Missing sweep manifest: {manifest_path}')
    manifest = json.loads(manifest_path.read_text())
    expected = {
        'tasks_per_seed': 30,
        'num_pockets': 100,
        'samples_per_pocket': 100,
        'rows_per_task': ROWS_PER_POINT,
        'generation_batch_size': 10000,
        'docking_shards_per_task': DOCKING_SHARDS,
        'docking_cpus_per_shard': 32,
        'docking_memory_gb_per_shard': 64,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f'Unexpected manifest {key}: {manifest.get(key)} vs {value}')
    checkpoint_paths = []
    for seed in SEEDS:
        checkpoint_paths.extend(
            Path(row['checkpoint_path'])
            for row in _read_tasks(SPEC_ROOT / 'mmgenmol' / f'seed{seed}.tsv')
        )
    missing_checkpoints = sorted(
        {path for path in checkpoint_paths if not path.is_file() or path.stat().st_size == 0}
    )
    if missing_checkpoints:
        raise FileNotFoundError(
            'Missing checkpoints:\n' + '\n'.join(str(path) for path in missing_checkpoints)
        )
    cache_root = DOCKING_CACHE_DIR / 'vina_dock'
    pqr_count = len(list(cache_root.rglob('*.pqr')))
    pdbqt_count = len(list(cache_root.rglob('*.pdbqt')))
    if pqr_count != 100 or pdbqt_count != 100:
        raise RuntimeError(
            f'Expected a complete receptor cache, found {pqr_count} PQR and '
            f'{pdbqt_count} PDBQT files'
        )


def main() -> None:
    _preflight()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    engine.RUN_OUTPUT_ROOT = RUN_ROOT
    engine.SPEC_ROOT = SPEC_ROOT
    engine.LOG_ROOT = LOG_ROOT
    engine.STATE_PATH = STATE_PATH
    engine.LOCK_PATH = LOCK_PATH
    engine.COMPLETE_PATH = COMPLETE_PATH

    lock_handle = LOCK_PATH.open('w')
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError('Another mmGenMol 100x controller is already running') from exc

    groups, tasks, terminal_keys = _build_dag()
    state = engine._load_state(tasks)
    state['status'] = 'running'
    state['controller_restarted_at_epoch'] = time.time()
    engine._atomic_write_state(state)
    print(
        f'controller_start groups={len(groups)} tasks={len(tasks)} '
        f'gpu_submit_limit={engine.GPU_MAX_SUBMITTED_JOBS}',
        flush=True,
    )

    try:
        while True:
            active_jobs, gpu_submitted_count = engine._active_jobs()
            engine._refresh_task_states(state, tasks, active_jobs)
            engine._atomic_write_state(state)
            if all(
                state['tasks'].get(key, {}).get('status') == 'complete'
                for key in terminal_keys
            ):
                COMPLETE_PATH.write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z') + '\n')
                state['status'] = 'complete'
                state['completed_at_epoch'] = time.time()
                engine._atomic_write_state(state)
                print(f"controller_complete {engine._progress_summary(state, tasks)}", flush=True)
                return

            ready = engine._ready_tasks(state, tasks)
            engine._schedule_cpu_tasks(state, groups, ready)
            active_jobs, gpu_submitted_count = engine._active_jobs()
            ready = engine._ready_tasks(state, tasks)
            engine._schedule_gpu_tasks(
                state,
                groups,
                ready,
                gpu_submitted_count,
            )
            print(
                f'controller_poll gpu_submitted={gpu_submitted_count} '
                f'{engine._progress_summary(state, tasks)}',
                flush=True,
            )
            time.sleep(engine.POLL_SECONDS)
    except Exception as exc:
        state['status'] = 'failed'
        state['failure'] = repr(exc)
        state['failed_at_epoch'] = time.time()
        engine._atomic_write_state(state)
        raise


if __name__ == '__main__':
    main()
