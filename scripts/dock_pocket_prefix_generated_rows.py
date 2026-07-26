import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from multiprocessing import Pool

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.realpath('.'), 'src'))

from genmol.mm.crossdocked import load_crossdocked_manifest
from genmol.mm.docking import (
    CrossDockedDockingEvaluator,
    DockingRecord,
    SUPPORTED_DOCKING_MODES,
    summarize_docking_records,
)


_WORKER_ENTRY_BY_SOURCE_INDEX = None
_WORKER_EVALUATORS = None


def _read_jsonl(path):
    rows = []
    with open(path) as handle:
        for line_idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Failed to parse JSONL line {line_idx} in {path}') from exc
    if not rows:
        raise ValueError(f'No generated rows found in {path}')
    return rows


def _load_entry_map(manifest_path, split):
    entries, _ = load_crossdocked_manifest(manifest_path, split)
    entry_by_source_index = {}
    for entry in entries:
        source_index = int(entry['source_index'])
        if source_index in entry_by_source_index:
            raise ValueError(f'Duplicate source_index in manifest split {split!r}: {source_index}')
        entry_by_source_index[source_index] = entry
    if not entry_by_source_index:
        raise ValueError(f'Manifest split {split!r} is empty: {manifest_path}')
    return entry_by_source_index


def _validate_generated_rows(rows, entry_by_source_index, max_rows):
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError('max_rows must be positive when provided')
        rows = rows[:max_rows]
    validated = []
    for row_idx, row in enumerate(rows):
        if 'source_index' not in row:
            raise ValueError(f'Generated row {row_idx} is missing source_index')
        if 'smiles' not in row:
            raise ValueError(f'Generated row {row_idx} is missing smiles')
        source_index = int(row['source_index'])
        if source_index not in entry_by_source_index:
            raise ValueError(f'Generated row {row_idx} source_index not found in manifest split: {source_index}')
        validated.append((row_idx, row))
    if not validated:
        raise ValueError('No generated rows remain after validation')
    return validated


def _select_pocket_shard(indexed_rows, num_shards, shard_index):
    if num_shards <= 0:
        raise ValueError('num_shards must be positive')
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f'shard_index must be in [0, {num_shards}), got {shard_index}'
        )
    source_indices = sorted({int(row['source_index']) for _, row in indexed_rows})
    if num_shards > len(source_indices):
        raise ValueError(
            f'num_shards exceeds the number of pockets: {num_shards} vs {len(source_indices)}'
        )
    selected_source_indices = {
        source_index
        for source_rank, source_index in enumerate(source_indices)
        if source_rank % num_shards == shard_index
    }
    selected_rows = [
        (row_idx, row)
        for row_idx, row in indexed_rows
        if int(row['source_index']) in selected_source_indices
    ]
    if not selected_rows:
        raise ValueError(
            f'Pocket shard {shard_index}/{num_shards} selected no generated rows'
        )
    return selected_rows, sorted(selected_source_indices), source_indices


def _build_evaluator(args, mode):
    return CrossDockedDockingEvaluator(
        crossdocked_root=args.crossdocked_root,
        docking_mode=mode,
        qvina_path=args.qvina_path,
        cache_dir=os.path.join(args.docking_cache_dir, mode),
        exhaustiveness=args.docking_exhaustiveness,
        num_cpu_dock=args.docking_num_cpu,
        num_modes=args.docking_num_modes,
        timeout_gen3d=args.docking_timeout_gen3d,
        timeout_dock=args.docking_timeout_dock,
        box_size=args.docking_box_size,
    )


def _prewarm_receptor_cache(args, rows, entry_by_source_index):
    unique_source_indices = sorted({int(row['source_index']) for row in rows})
    for mode in args.docking_modes:
        evaluator = _build_evaluator(args, mode)
        try:
            start_time = time.perf_counter()
            for source_index in unique_source_indices:
                entry = entry_by_source_index[source_index]
                receptor_pdb_path = evaluator._resolve_receptor_pdb_path(entry)
                if mode == 'qvina':
                    evaluator._ensure_qvina_receptor_pdbqt(receptor_pdb_path)
                else:
                    evaluator._prepare_vina_receptor(receptor_pdb_path)
            elapsed = time.perf_counter() - start_time
            print(
                f'prewarmed mode={mode} unique_receptors={len(unique_source_indices)} elapsed_sec={elapsed:.2f}',
                flush=True,
            )
        finally:
            evaluator.close()


def _init_worker(entry_by_source_index, worker_args_dict):
    global _WORKER_ENTRY_BY_SOURCE_INDEX
    global _WORKER_EVALUATORS

    class WorkerArgs:
        pass

    worker_args = WorkerArgs()
    for key, value in worker_args_dict.items():
        setattr(worker_args, key, value)

    _WORKER_ENTRY_BY_SOURCE_INDEX = entry_by_source_index
    _WORKER_EVALUATORS = {
        mode: _build_evaluator(worker_args, mode)
        for mode in worker_args.docking_modes
    }


def _dock_one(task):
    row_idx = int(task['row_idx'])
    mode = task['mode']
    row = task['row']
    source_index = int(row['source_index'])
    smiles = row.get('smiles')
    entry = _WORKER_ENTRY_BY_SOURCE_INDEX[source_index]
    evaluator = _WORKER_EVALUATORS[mode]

    start_time = time.perf_counter()
    record = evaluator.score(entries=[entry], smiles_list=[smiles])[0]
    elapsed_sec = time.perf_counter() - start_time
    return {
        'task_index': int(task['task_index']),
        'row_idx': row_idx,
        'source_index': source_index,
        'mode': mode,
        'elapsed_sec': float(elapsed_sec),
        'smiles': smiles,
        'record': asdict(record),
    }


def _write_json(path, payload):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_rows_path', required=True)
    parser.add_argument('--manifest_path', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--crossdocked_root', required=True)
    parser.add_argument('--qvina_path', required=True)
    parser.add_argument('--docking_cache_dir', required=True)
    parser.add_argument('--output_rows_path', required=True)
    parser.add_argument('--output_summary_path', required=True)
    parser.add_argument('--num_workers', type=int, required=True)
    parser.add_argument('--max_rows', type=int, default=None)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--shard_index', type=int, default=0)
    parser.add_argument('--docking_modes', nargs='+', default=['vina_dock', 'qvina'])
    parser.add_argument('--docking_exhaustiveness', type=int, default=8)
    parser.add_argument('--docking_num_cpu', type=int, default=1)
    parser.add_argument('--docking_num_modes', type=int, default=10)
    parser.add_argument('--docking_timeout_gen3d', type=int, default=30)
    parser.add_argument('--docking_timeout_dock', type=int, default=100)
    parser.add_argument('--docking_box_size', nargs=3, type=float, default=[20.0, 20.0, 20.0])
    parser.add_argument('--progress_every', type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_workers <= 0:
        raise ValueError('num_workers must be positive')
    if args.progress_every <= 0:
        raise ValueError('progress_every must be positive')
    if args.docking_num_cpu != 1:
        raise ValueError('docking_num_cpu must be 1 for fixed-core worker-pool probes')
    if os.path.exists(args.output_rows_path):
        raise FileExistsError(f'output_rows_path already exists: {args.output_rows_path}')
    if os.path.exists(args.output_summary_path):
        raise FileExistsError(f'output_summary_path already exists: {args.output_summary_path}')
    if not os.path.exists(args.generated_rows_path):
        raise FileNotFoundError(f'generated_rows_path not found: {args.generated_rows_path}')
    for mode in args.docking_modes:
        if mode not in SUPPORTED_DOCKING_MODES:
            raise ValueError(f'docking mode must be one of {SUPPORTED_DOCKING_MODES}, got {mode!r}')
    if 'vina_score' in args.docking_modes and 'vina_dock' in args.docking_modes:
        raise ValueError('docking_modes cannot include both vina_score and vina_dock')

    rows = _read_jsonl(args.generated_rows_path)
    entry_by_source_index = _load_entry_map(args.manifest_path, args.split)
    indexed_rows = _validate_generated_rows(rows, entry_by_source_index, args.max_rows)
    global_num_rows = len(indexed_rows)
    indexed_rows, selected_source_indices, all_source_indices = _select_pocket_shard(
        indexed_rows,
        args.num_shards,
        args.shard_index,
    )
    selected_rows = [row for _, row in indexed_rows]
    _prewarm_receptor_cache(args, selected_rows, entry_by_source_index)

    tasks = []
    for row_idx, row in indexed_rows:
        for mode in args.docking_modes:
            tasks.append({'task_index': len(tasks), 'row_idx': row_idx, 'mode': mode, 'row': row})
    if not tasks:
        raise ValueError('No docking tasks were constructed')

    worker_args_dict = vars(args).copy()
    output_parent = os.path.dirname(args.output_rows_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    start_time = time.perf_counter()
    completed = 0
    with open(args.output_rows_path, 'w') as output_handle:
        with Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=(entry_by_source_index, worker_args_dict),
        ) as pool:
            chunksize = max(1, len(tasks) // max(1, args.num_workers * 8))
            for result in pool.imap_unordered(_dock_one, tasks, chunksize=chunksize):
                completed += 1
                output_handle.write(json.dumps(result, sort_keys=True) + '\n')
                if completed % args.progress_every == 0 or completed == len(tasks):
                    output_handle.flush()
                    elapsed = time.perf_counter() - start_time
                    print(
                        f'completed={completed}/{len(tasks)} elapsed_sec={elapsed:.2f} '
                        f'tasks_per_sec={completed / max(elapsed, 1e-9):.4f}',
                        flush=True,
                    )

    elapsed_sec = time.perf_counter() - start_time
    parsed_records_by_mode = {mode: [] for mode in args.docking_modes}
    with open(args.output_rows_path) as handle:
        for line in handle:
            row = json.loads(line)
            record_payload = row['record']
            parsed_records_by_mode[row['mode']].append(DockingRecord(**record_payload))

    summaries = {}
    for mode, records in parsed_records_by_mode.items():
        if not records:
            raise RuntimeError(f'No records produced for mode {mode!r}')
        summary = summarize_docking_records(records)
        if float(summary['docking_success_fraction']) <= 0.0:
            first_error = next((record.error for record in records if record.error), 'unknown docking failure')
            raise RuntimeError(f'Zero successful dockings for mode {mode!r}; first error: {first_error}')
        summaries[mode] = summary

    summary_payload = {
        'generated_rows_path': args.generated_rows_path,
        'global_num_rows': global_num_rows,
        'num_rows': len(indexed_rows),
        'num_shards': args.num_shards,
        'shard_index': args.shard_index,
        'num_pockets': len(selected_source_indices),
        'global_num_pockets': len(all_source_indices),
        'selected_source_indices': selected_source_indices,
        'docking_modes': args.docking_modes,
        'num_workers': args.num_workers,
        'docking_num_cpu': args.docking_num_cpu,
        'num_tasks': len(tasks),
        'elapsed_sec': float(elapsed_sec),
        'tasks_per_sec': float(len(tasks) / max(elapsed_sec, 1e-9)),
        'molecules_per_sec_two_modes_equivalent': float(len(indexed_rows) / max(elapsed_sec, 1e-9)),
        'summaries': summaries,
    }
    _write_json(args.output_summary_path, summary_payload)
    print(json.dumps(summary_payload, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
