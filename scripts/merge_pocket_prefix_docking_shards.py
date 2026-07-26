#!/usr/bin/env python3
"""Merge independently docked pocket shards into one validated result."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.realpath('.'), 'src'))

from genmol.mm.docking import DockingRecord, SUPPORTED_DOCKING_MODES, summarize_docking_records


def _read_json(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'Missing non-empty JSON file: {path}')
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f'Expected a JSON object in {path}')
    return payload


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'Missing non-empty JSONL file: {path}')
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSONL line {line_number} in {path}') from exc
            if not isinstance(row, dict):
                raise ValueError(f'Expected an object on JSONL line {line_number} in {path}')
            rows.append(row)
    if not rows:
        raise ValueError(f'No records found in {path}')
    return rows


def _atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + '.tmp')
    with temporary_path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    os.replace(temporary_path, path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + '.tmp')
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(temporary_path, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard_root', type=Path, required=True)
    parser.add_argument('--num_shards', type=int, required=True)
    parser.add_argument('--expected_num_rows', type=int, required=True)
    parser.add_argument('--output_rows_path', type=Path, required=True)
    parser.add_argument('--output_summary_path', type=Path, required=True)
    parser.add_argument('--docking_modes', nargs='+', default=['vina_dock'])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_shards <= 0:
        raise ValueError('num_shards must be positive')
    if args.expected_num_rows <= 0:
        raise ValueError('expected_num_rows must be positive')
    if args.output_rows_path.exists():
        raise FileExistsError(f'Output already exists: {args.output_rows_path}')
    if args.output_summary_path.exists():
        raise FileExistsError(f'Output already exists: {args.output_summary_path}')
    if len(set(args.docking_modes)) != len(args.docking_modes):
        raise ValueError(f'Duplicate docking modes: {args.docking_modes}')
    invalid_modes = sorted(set(args.docking_modes) - set(SUPPORTED_DOCKING_MODES))
    if invalid_modes:
        raise ValueError(f'Unsupported docking modes: {invalid_modes}')

    merged_rows = []
    shard_summaries = []
    seen_keys = set()
    row_to_shard = {}
    source_to_shard = {}
    for shard_index in range(args.num_shards):
        shard_dir = args.shard_root / f'shard_{shard_index:02d}'
        summary = _read_json(shard_dir / 'docking.summary.json')
        records = _read_jsonl(shard_dir / 'docking.records.jsonl')
        if int(summary.get('num_shards', -1)) != args.num_shards:
            raise ValueError(f'num_shards mismatch in {shard_dir}')
        if int(summary.get('shard_index', -1)) != shard_index:
            raise ValueError(f'shard_index mismatch in {shard_dir}')
        if int(summary.get('global_num_rows', -1)) != args.expected_num_rows:
            raise ValueError(f'global_num_rows mismatch in {shard_dir}')
        if int(summary.get('num_rows', -1)) * len(args.docking_modes) != len(records):
            raise ValueError(f'Record count does not match shard summary in {shard_dir}')
        if tuple(summary.get('docking_modes', ())) != tuple(args.docking_modes):
            raise ValueError(f'Docking modes mismatch in {shard_dir}')

        selected_sources = {int(value) for value in summary.get('selected_source_indices', ())}
        if not selected_sources:
            raise ValueError(f'No selected_source_indices in {shard_dir}')
        for source_index in selected_sources:
            previous_shard = source_to_shard.setdefault(source_index, shard_index)
            if previous_shard != shard_index:
                raise ValueError(
                    f'Pocket source_index={source_index} appears in shards '
                    f'{previous_shard} and {shard_index}'
                )

        for row in records:
            row_idx = int(row.get('row_idx', -1))
            source_index = int(row.get('source_index', -1))
            mode = str(row.get('mode'))
            if row_idx < 0 or row_idx >= args.expected_num_rows:
                raise ValueError(f'Out-of-range row_idx={row_idx} in {shard_dir}')
            if source_index not in selected_sources:
                raise ValueError(
                    f'Record source_index={source_index} is not assigned to {shard_dir}'
                )
            if mode not in args.docking_modes:
                raise ValueError(f'Unexpected docking mode={mode!r} in {shard_dir}')
            key = (row_idx, mode)
            if key in seen_keys:
                raise ValueError(f'Duplicate docking record key={key}')
            seen_keys.add(key)
            previous_shard = row_to_shard.setdefault(row_idx, shard_index)
            if previous_shard != shard_index:
                raise ValueError(
                    f'Generated row_idx={row_idx} appears in shards '
                    f'{previous_shard} and {shard_index}'
                )
            merged_rows.append(row)
        shard_summaries.append(summary)

    expected_keys = {
        (row_idx, mode)
        for row_idx in range(args.expected_num_rows)
        for mode in args.docking_modes
    }
    if seen_keys != expected_keys:
        missing = sorted(expected_keys - seen_keys)
        extra = sorted(seen_keys - expected_keys)
        raise ValueError(
            f'Merged docking coverage mismatch: missing={missing[:10]} extra={extra[:10]}'
        )
    merged_rows.sort(
        key=lambda row: (int(row['row_idx']), args.docking_modes.index(str(row['mode'])))
    )

    records_by_mode = defaultdict(list)
    for row in merged_rows:
        records_by_mode[str(row['mode'])].append(DockingRecord(**row['record']))
    summaries = {
        mode: summarize_docking_records(records_by_mode[mode])
        for mode in args.docking_modes
    }
    for mode, summary in summaries.items():
        if float(summary['docking_success_fraction']) <= 0.0:
            raise RuntimeError(f'Zero successful dockings after merge for mode={mode}')

    elapsed_sec = max(float(summary['elapsed_sec']) for summary in shard_summaries)
    summary_payload = {
        'generated_rows_path': shard_summaries[0]['generated_rows_path'],
        'global_num_rows': args.expected_num_rows,
        'num_rows': args.expected_num_rows,
        'num_shards': args.num_shards,
        'docking_modes': args.docking_modes,
        'num_tasks': len(merged_rows),
        'elapsed_sec': elapsed_sec,
        'tasks_per_sec': float(len(merged_rows) / max(elapsed_sec, 1e-9)),
        'summaries': summaries,
        'shards': [
            {
                'shard_index': int(summary['shard_index']),
                'num_rows': int(summary['num_rows']),
                'num_pockets': int(summary['num_pockets']),
                'elapsed_sec': float(summary['elapsed_sec']),
            }
            for summary in shard_summaries
        ],
    }
    _atomic_write_jsonl(args.output_rows_path, merged_rows)
    _atomic_write_json(args.output_summary_path, summary_payload)
    print(json.dumps(summary_payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
