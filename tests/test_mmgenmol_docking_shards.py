import json
import sys
from pathlib import Path

import pytest

from scripts.dock_pocket_prefix_generated_rows import _select_pocket_shard
from scripts.merge_pocket_prefix_docking_shards import main as merge_main


def _record(row_idx: int, source_index: int) -> dict:
    return {
        'task_index': row_idx,
        'row_idx': row_idx,
        'source_index': source_index,
        'mode': 'vina_dock',
        'elapsed_sec': 1.0,
        'smiles': 'CC',
        'record': {
            'mode': 'vina_dock',
            'is_success': True,
            'error': None,
            'receptor_pdb_path': '/tmp/receptor.pdb',
            'receptor_pdbqt_path': '/tmp/receptor.pdbqt',
            'ligand_sdf_path': '/tmp/ligand.sdf',
            'ligand_pdbqt_path': '/tmp/ligand.pdbqt',
            'center_x': 0.0,
            'center_y': 0.0,
            'center_z': 0.0,
            'size_x': 20.0,
            'size_y': 20.0,
            'size_z': 20.0,
            'score_only_affinity': -5.0,
            'minimize_affinity': -6.0,
            'dock_affinity': -7.0,
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_select_pocket_shard_preserves_global_row_indices():
    indexed_rows = [
        (row_idx, {'source_index': source_index})
        for row_idx, source_index in enumerate([10, 10, 20, 20, 30, 30, 40, 40])
    ]

    selected, selected_sources, all_sources = _select_pocket_shard(
        indexed_rows,
        num_shards=2,
        shard_index=1,
    )

    assert selected_sources == [20, 40]
    assert all_sources == [10, 20, 30, 40]
    assert [row_idx for row_idx, _ in selected] == [2, 3, 6, 7]


def test_select_pocket_shard_rejects_more_shards_than_pockets():
    with pytest.raises(ValueError, match='exceeds the number of pockets'):
        _select_pocket_shard(
            [(0, {'source_index': 10})],
            num_shards=2,
            shard_index=0,
        )


def test_merge_docking_shards_validates_and_restores_global_order(tmp_path, monkeypatch):
    shard_root = tmp_path / 'shards'
    shard_rows = (
        [_record(0, 10), _record(1, 10)],
        [_record(2, 20), _record(3, 20)],
    )
    for shard_index, rows in enumerate(shard_rows):
        shard_dir = shard_root / f'shard_{shard_index:02d}'
        _write_jsonl(shard_dir / 'docking.records.jsonl', list(reversed(rows)))
        (shard_dir / 'docking.summary.json').write_text(
            json.dumps(
                {
                    'generated_rows_path': '/tmp/generated.rows.jsonl',
                    'global_num_rows': 4,
                    'num_rows': 2,
                    'num_shards': 2,
                    'shard_index': shard_index,
                    'num_pockets': 1,
                    'selected_source_indices': [10 + 10 * shard_index],
                    'docking_modes': ['vina_dock'],
                    'elapsed_sec': float(shard_index + 1),
                }
            )
        )

    output_rows = tmp_path / 'docking.records.jsonl'
    output_summary = tmp_path / 'docking.summary.json'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'merge_pocket_prefix_docking_shards.py',
            '--shard_root',
            str(shard_root),
            '--num_shards',
            '2',
            '--expected_num_rows',
            '4',
            '--output_rows_path',
            str(output_rows),
            '--output_summary_path',
            str(output_summary),
            '--docking_modes',
            'vina_dock',
        ],
    )

    merge_main()

    merged = [json.loads(line) for line in output_rows.read_text().splitlines()]
    assert [row['row_idx'] for row in merged] == [0, 1, 2, 3]
    summary = json.loads(output_summary.read_text())
    assert summary['num_rows'] == 4
    assert summary['num_shards'] == 2
    assert summary['elapsed_sec'] == 2.0
    assert summary['summaries']['vina_dock']['docking_success_fraction'] == 1.0
