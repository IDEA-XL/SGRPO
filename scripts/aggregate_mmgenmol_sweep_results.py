import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

MARKER_CYCLE = ('o', '^', 's', 'D', 'P', 'X', 'v', '<', '>')

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.realpath('.'), 'src'))

MOLECULAR_REWARD_NAME_ORDER = (
    'qed',
    'sa_score',
    'drugclip_score',
    'unidock_score',
)
DEFAULT_MOLECULAR_REWARD_WEIGHTS = {
    'qed': 0.6,
    'sa_score': 0.4,
    'drugclip_score': 0.0,
    'unidock_score': 0.0,
}


def sa_to_score(sa_value):
    return max(0.0, min((6.0 - float(sa_value)) / 5.0, 1.0))


def unidock_affinity_to_score(unidock_affinity):
    return max(0.0, min((-float(unidock_affinity)) / 10.0, 1.0))


def normalize_molecular_reward_weights(config):
    if config is None:
        raw = dict(DEFAULT_MOLECULAR_REWARD_WEIGHTS)
    else:
        raw = dict(DEFAULT_MOLECULAR_REWARD_WEIGHTS)
        for reward_name in MOLECULAR_REWARD_NAME_ORDER:
            if reward_name in config and config[reward_name] is not None:
                raw[reward_name] = config[reward_name]
    normalized = {}
    for reward_name in MOLECULAR_REWARD_NAME_ORDER:
        value = raw[reward_name]
        try:
            weight = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'molecular reward weight for {reward_name!r} must be numeric, got {value!r}') from exc
        if weight < 0.0:
            raise ValueError(f'molecular reward weight for {reward_name!r} must be non-negative, got {weight}')
        normalized[reward_name] = weight
    return normalized


def compute_soft_reward(
    qed_value,
    sa_score_value,
    drugclip_score_value=None,
    unidock_score_value=None,
    reward_weights=None,
):
    normalized_weights = normalize_molecular_reward_weights(reward_weights)
    outputs = 0.0
    if normalized_weights['qed'] > 0.0:
        if qed_value is None:
            raise ValueError('qed_value is required when qed reward weight is positive')
        outputs += normalized_weights['qed'] * float(qed_value)
    if normalized_weights['sa_score'] > 0.0:
        if sa_score_value is None:
            raise ValueError('sa_score_value is required when sa_score reward weight is positive')
        outputs += normalized_weights['sa_score'] * float(sa_score_value)
    if normalized_weights['drugclip_score'] > 0.0:
        if drugclip_score_value is None:
            raise ValueError('drugclip_score_value is required when drugclip_score reward weight is positive')
        outputs += normalized_weights['drugclip_score'] * float(drugclip_score_value)
    if normalized_weights['unidock_score'] > 0.0:
        if unidock_score_value is None:
            raise ValueError('unidock_score_value is required when unidock_score reward weight is positive')
        outputs += normalized_weights['unidock_score'] * float(unidock_score_value)
    return float(outputs)


MODEL_REWARD_WEIGHTS = {
    'original_5500': {'qed': 0.6, 'sa_score': 0.4, 'drugclip_score': 0.0, 'unidock_score': 0.0},
    'grpo_1000': {'qed': 0.6, 'sa_score': 0.4, 'drugclip_score': 0.0, 'unidock_score': 0.0},
    'sgrpo_1000': {'qed': 0.6, 'sa_score': 0.4, 'drugclip_score': 0.0, 'unidock_score': 0.0},
    'grpo_divreg005_1000': {'qed': 0.6, 'sa_score': 0.4, 'drugclip_score': 0.0, 'unidock_score': 0.0},
    'grpo_unidock_500': {'qed': 0.3, 'sa_score': 0.2, 'drugclip_score': 0.0, 'unidock_score': 0.5},
    'grpo_unidock_1000': {'qed': 0.3, 'sa_score': 0.2, 'drugclip_score': 0.0, 'unidock_score': 0.5},
    'mmgenmol_dmb_1000': {'qed': 0.3, 'sa_score': 0.2, 'drugclip_score': 0.0, 'unidock_score': 0.5},
    'mmgenmol_entropy_1000': {'qed': 0.3, 'sa_score': 0.2, 'drugclip_score': 0.0, 'unidock_score': 0.5},
    'grpo_unidock_hbd_st07_sc04_1000': {'qed': 0.3, 'sa_score': 0.2, 'drugclip_score': 0.0, 'unidock_score': 0.5},
    'sgrpo_unidock_500': {'qed': 0.3, 'sa_score': 0.2, 'drugclip_score': 0.0, 'unidock_score': 0.5},
    'sgrpo_unidock_1000': {'qed': 0.3, 'sa_score': 0.2, 'drugclip_score': 0.0, 'unidock_score': 0.5},
    'sgrpo_unidock_rewardsum_loo_500': {
        'qed': 0.3,
        'sa_score': 0.2,
        'drugclip_score': 0.0,
        'unidock_score': 0.5,
    },
    'sgrpo_unidock_rewardsum_loo_1000': {
        'qed': 0.3,
        'sa_score': 0.2,
        'drugclip_score': 0.0,
        'unidock_score': 0.5,
    },
}

VINA_DERIVED_DOCKING_REWARD_LABEL = 'Vina-derived Docking Reward Proxy'
ORIGINAL_MODEL_NAME = 'original_5500'
WITH_UNIDOCK_PLOT_REWARD_WEIGHTS = {'qed': 0.3, 'sa_score': 0.2, 'drugclip_score': 0.0, 'unidock_score': 0.5}


def _reward_weights_for_model(model_name):
    try:
        return dict(MODEL_REWARD_WEIGHTS[model_name])
    except KeyError as exc:
        raise ValueError(f'No reward-weight mapping registered for model_name={model_name!r}') from exc


def _uses_unidock_reward(model_name):
    return normalize_molecular_reward_weights(_reward_weights_for_model(model_name))['unidock_score'] > 0.0


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
        raise ValueError(f'No rows found in {path}')
    return rows


def _write_json(path, payload):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_jsonl(path, rows):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, 'w') as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + '\n')


def _require_file(path, label):
    if not path:
        raise ValueError(f'{label} is required')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{label} not found: {path}')


def _build_pocket_entry_index(manifest_path, manifest_split):
    from genmol.mm.crossdocked import load_crossdocked_manifest

    entries, _ = load_crossdocked_manifest(manifest_path, manifest_split)
    by_source_index = {}
    for entry in entries:
        source_index = int(entry['source_index'])
        if source_index in by_source_index:
            raise ValueError(f'Duplicate source_index={source_index} in manifest {manifest_path}')
        by_source_index[source_index] = entry
    if not by_source_index:
        raise ValueError(f'No manifest entries loaded from {manifest_path} for split={manifest_split!r}')
    return by_source_index


def _format_metric(value):
    if value is None:
        return 'nan'
    value = float(value)
    if math.isnan(value):
        return 'nan'
    return f'{value:.6f}'


def _parse_task_manifest(path):
    with open(path, newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        rows = list(reader)
    required = {
        'task_id',
        'model_name',
        'sweep_type',
        'sweep_value',
        'randomness',
        'temperature',
        'checkpoint_path',
        'output_path',
    }
    if reader.fieldnames is None or set(reader.fieldnames) != required:
        raise ValueError(f'Unexpected task manifest header in {path}: {reader.fieldnames}')
    if not rows:
        raise ValueError(f'Task manifest is empty: {path}')
    seen = set()
    for row in rows:
        task_id = int(row['task_id'])
        if task_id in seen:
            raise ValueError(f'Duplicate task_id in {path}: {task_id}')
        seen.add(task_id)
    return sorted(rows, key=lambda item: int(item['task_id']))


def _canonicalize_smiles(smiles, chem):
    if smiles is None:
        return None
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = chem.MolFromSmiles(smiles, sanitize=True)
    except Exception:
        return None
    if mol is None:
        return None
    return chem.MolToSmiles(mol)


def _mean(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return float('nan')
    return float(sum(values) / len(values))


def _median(values):
    values = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not values:
        return float('nan')
    mid = len(values) // 2
    if len(values) % 2 == 1:
        return values[mid]
    return float((values[mid - 1] + values[mid]) / 2.0)


def _compute_internal_diversity_for_group(canonical_smiles, fingerprint_cache, data_structs):
    valid = [smiles for smiles in canonical_smiles if smiles is not None]
    if len(valid) < 2:
        return 0.0
    similarity_sum = 0.0
    pair_count = 0
    for left_idx in range(len(valid)):
        left_fp = fingerprint_cache[valid[left_idx]]
        for right_idx in range(left_idx + 1, len(valid)):
            right_fp = fingerprint_cache[valid[right_idx]]
            similarity_sum += float(data_structs.TanimotoSimilarity(left_fp, right_fp))
            pair_count += 1
    if pair_count == 0:
        return 0.0
    return float(1.0 - similarity_sum / pair_count)


def _load_docking_summary(path, docking_mode):
    with open(path) as handle:
        payload = json.load(handle)
    summaries = payload.get('summaries')
    if not isinstance(summaries, dict) or docking_mode not in summaries:
        raise ValueError(f'Missing docking summary for mode {docking_mode!r}: {path}')
    summary = summaries[docking_mode]
    success_fraction = float(summary['docking_success_fraction'])
    if success_fraction <= 0.0:
        raise ValueError(f'Docking success fraction is zero for {path}')
    return payload, summary


def _compute_task_metrics(task, args, chem, qed, sa_oracle, fingerprint_generator, data_structs):
    generated_rows_path = task['output_path']
    if not os.path.exists(generated_rows_path):
        raise FileNotFoundError(f'Missing generated rows file: {generated_rows_path}')
    generated_rows = _read_jsonl(generated_rows_path)
    if len(generated_rows) != args.expected_rows_per_task:
        raise ValueError(
            f'Generated row count mismatch for {generated_rows_path}: '
            f'{len(generated_rows)} vs {args.expected_rows_per_task}'
        )

    output_dir = os.path.join(
        args.docking_root,
        task['model_name'],
        f"{task['sweep_type']}_{task['sweep_value']}",
    )
    docking_records_path = os.path.join(output_dir, 'docking.records.jsonl')
    docking_summary_path = os.path.join(output_dir, 'docking.summary.json')
    if not os.path.exists(docking_records_path):
        raise FileNotFoundError(f'Missing docking records file: {docking_records_path}')
    if not os.path.exists(docking_summary_path):
        raise FileNotFoundError(f'Missing docking summary file: {docking_summary_path}')
    docking_records = _read_jsonl(docking_records_path)
    if len(docking_records) != args.expected_rows_per_task:
        raise ValueError(
            f'Docking record count mismatch for {docking_records_path}: '
            f'{len(docking_records)} vs {args.expected_rows_per_task}'
        )
    docking_records_by_row_idx = {}
    for docking_row in docking_records:
        row_idx = int(docking_row.get('row_idx', -1))
        if row_idx < 0:
            raise ValueError(f'Docking row is missing a valid row_idx in {docking_records_path}')
        if row_idx in docking_records_by_row_idx:
            raise ValueError(f'Duplicate docking row_idx={row_idx} in {docking_records_path}')
        docking_records_by_row_idx[row_idx] = docking_row
    expected_row_indices = set(range(args.expected_rows_per_task))
    observed_row_indices = set(docking_records_by_row_idx)
    if observed_row_indices != expected_row_indices:
        missing = sorted(expected_row_indices - observed_row_indices)
        extra = sorted(observed_row_indices - expected_row_indices)
        raise ValueError(
            f'Docking row_idx coverage mismatch for {docking_records_path}: '
            f'missing={missing[:10]} extra={extra[:10]}'
        )
    docking_payload, docking_summary = _load_docking_summary(docking_summary_path, args.docking_mode)

    canonical_smiles_by_row = []
    source_indices_by_row = []
    for row_idx, row in enumerate(generated_rows):
        if 'source_index' not in row:
            raise ValueError(f'Generated row {row_idx} missing source_index in {generated_rows_path}')
        source_index = int(row['source_index'])
        canonical_smiles = _canonicalize_smiles(row.get('smiles'), chem)
        canonical_smiles_by_row.append(canonical_smiles)
        source_indices_by_row.append(source_index)

    all_source_indices = set(source_indices_by_row)
    if len(all_source_indices) != args.expected_num_pockets:
        raise ValueError(
            f'Expected {args.expected_num_pockets} pockets for {generated_rows_path}, '
            f'found {len(all_source_indices)}'
        )
    source_index_counts = defaultdict(int)
    for source_index in source_indices_by_row:
        source_index_counts[source_index] += 1
    bad_group_sizes = {
        source_index: count
        for source_index, count in source_index_counts.items()
        if count != args.expected_samples_per_pocket
    }
    if bad_group_sizes:
        raise ValueError(f'Unexpected per-pocket sample counts in {generated_rows_path}: {bad_group_sizes}')

    unique_base_valid_smiles = sorted({smiles for smiles in canonical_smiles_by_row if smiles is not None})
    qed_by_smiles = {}
    sa_by_smiles = {}
    sa_score_by_smiles = {}
    fp_by_smiles = {}
    if unique_base_valid_smiles:
        sa_values = sa_oracle(unique_base_valid_smiles)
        if len(sa_values) != len(unique_base_valid_smiles):
            raise RuntimeError(
                f'SA oracle returned {len(sa_values)} values for {len(unique_base_valid_smiles)} SMILES'
            )
        for smiles, sa_value in zip(unique_base_valid_smiles, sa_values):
            mol = chem.MolFromSmiles(smiles, sanitize=True)
            if mol is None:
                raise RuntimeError(f'Canonical valid SMILES failed to parse on second pass: {smiles}')
            qed_by_smiles[smiles] = float(qed.qed(mol))
            sa_by_smiles[smiles] = float(sa_value)
            sa_score_by_smiles[smiles] = float(sa_to_score(sa_value))
            fp_by_smiles[smiles] = fingerprint_generator.GetFingerprint(mol)
    normalized_reward_weights = normalize_molecular_reward_weights(_reward_weights_for_model(task['model_name']))
    uses_unidock_reward = normalized_reward_weights['unidock_score'] > 0.0
    base_valid_mask = [smiles is not None for smiles in canonical_smiles_by_row]
    final_valid_mask = list(base_valid_mask)
    drugclip_score_by_row = [None] * len(generated_rows)
    if args.drugclip_scorer is not None:
        active_smiles = []
        active_pocket_entries = []
        active_row_indices = []
        for row_index, (canonical_smiles, source_index) in enumerate(
            zip(canonical_smiles_by_row, source_indices_by_row)
        ):
            if canonical_smiles is None:
                continue
            try:
                pocket_entry = args.pocket_entries_by_source_index[source_index]
            except KeyError as exc:
                raise KeyError(
                    f'Manifest is missing source_index={source_index} required for DrugCLIP offline scoring'
                ) from exc
            active_row_indices.append(row_index)
            active_smiles.append(canonical_smiles)
            active_pocket_entries.append(pocket_entry)
        drugclip_scores = args.drugclip_scorer.score(active_smiles, active_pocket_entries)
        if len(drugclip_scores) != len(active_smiles):
            raise ValueError(
                'DrugCLIP offline scorer returned mismatched score count: '
                f'expected {len(active_smiles)}, got {len(drugclip_scores)}'
            )
        for row_index, drugclip_score in zip(active_row_indices, drugclip_scores):
            if drugclip_score is None:
                final_valid_mask[row_index] = False
                continue
            drugclip_score_by_row[row_index] = float(drugclip_score)

    rows_by_source_index = defaultdict(list)
    qeds = []
    sa_values = []
    sa_scores = []
    soft_rewards = []
    used_drugclip_scores = []
    used_unidock_scores = []
    docking_success_flags = []
    docking_affinities = []
    score_only_affinities = []
    minimize_affinities = []
    for row_idx, (smiles, source_index) in enumerate(zip(canonical_smiles_by_row, source_indices_by_row)):
        docking_row = docking_records_by_row_idx[row_idx]
        if int(docking_row.get('source_index', source_index)) != source_index:
            raise ValueError(
                f'Docking source_index mismatch for task_id={task["task_id"]}: '
                f'expected source_index={source_index}, got {docking_row.get("source_index")}'
            )
        docking_record = docking_row['record']
        is_success = bool(docking_record['is_success'])
        if uses_unidock_reward and final_valid_mask[row_idx] and not is_success:
            final_valid_mask[row_idx] = False
        if not final_valid_mask[row_idx]:
            continue
        if smiles is None:
            raise ValueError(f'final_valid_mask row {row_idx} is true but canonical SMILES is None')
        rows_by_source_index[source_index].append(smiles)
        qeds.append(qed_by_smiles[smiles])
        sa_values.append(sa_by_smiles[smiles])
        sa_scores.append(sa_score_by_smiles[smiles])
        drugclip_score_value = drugclip_score_by_row[row_idx]
        if normalized_reward_weights['drugclip_score'] > 0.0:
            if drugclip_score_value is None:
                raise ValueError(
                    f'DrugCLIP score missing for final-valid row {row_idx} in task_id={task["task_id"]}'
                )
            used_drugclip_scores.append(drugclip_score_value)
        unidock_score_value = None
        if is_success:
            dock_affinity = float(docking_record['dock_affinity'])
            docking_affinities.append(dock_affinity)
            score_only_affinities.append(float(docking_record['score_only_affinity']))
            minimize_affinities.append(float(docking_record['minimize_affinity']))
            if uses_unidock_reward:
                unidock_score_value = float(unidock_affinity_to_score(dock_affinity))
                used_unidock_scores.append(unidock_score_value)
        soft_rewards.append(
            compute_soft_reward(
                qed_by_smiles[smiles],
                sa_score_by_smiles[smiles],
                drugclip_score_value=drugclip_score_value,
                unidock_score_value=unidock_score_value,
                reward_weights=normalized_reward_weights,
            )
        )
        docking_success_flags.append(is_success)

    if len(rows_by_source_index) != args.expected_num_pockets:
        raise ValueError(
            f'Final valid set covers {len(rows_by_source_index)} pockets for {generated_rows_path}; '
            f'expected {args.expected_num_pockets}'
        )

    pocket_diversities = []
    for source_index in sorted(rows_by_source_index):
        diversity = _compute_internal_diversity_for_group(
            rows_by_source_index[source_index],
            fingerprint_cache=fp_by_smiles,
            data_structs=data_structs,
        )
        pocket_diversities.append(diversity)

    valid_count = int(sum(final_valid_mask))
    valid_fraction = float(valid_count / len(generated_rows))
    if valid_count == 0:
        raise ValueError(f'No final-valid molecules remain for task_id={task["task_id"]}')
    unique_valid_smiles = sorted({smiles for smiles, is_valid in zip(canonical_smiles_by_row, final_valid_mask) if is_valid})
    duplicate_fraction = 1.0 - float(len(unique_valid_smiles) / valid_count)
    vina_dock_success_fraction = float(sum(docking_success_flags) / valid_count)
    if not docking_affinities:
        raise ValueError(f'No successful dockings remain in final-valid set for task_id={task["task_id"]}')
    row = {
        'task_id': int(task['task_id']),
        'model_name': task['model_name'],
        'sweep_type': task['sweep_type'],
        'sweep_value': float(task['sweep_value']),
        'randomness': float(task['randomness']),
        'temperature': float(task['temperature']),
        'checkpoint_path': task['checkpoint_path'],
        'generated_rows_path': generated_rows_path,
        'docking_records_path': docking_records_path,
        'docking_summary_path': docking_summary_path,
        'num_rows': len(generated_rows),
        'num_pockets': len(rows_by_source_index),
        'samples_per_pocket': args.expected_samples_per_pocket,
        'valid_count': valid_count,
        'valid_fraction': valid_fraction,
        'unique_valid_count': len(unique_valid_smiles),
        'duplicate_fraction': duplicate_fraction,
        'qed_mean': _mean(qeds),
        'sa_mean': _mean(sa_values),
        'sa_score_mean': _mean(sa_scores),
        'drugclip_score_mean': _mean(used_drugclip_scores) if used_drugclip_scores else float('nan'),
        'unidock_score_mean': _mean(used_unidock_scores) if used_unidock_scores else float('nan'),
        'soft_reward_mean': _mean(soft_rewards),
        'diversity': _mean(pocket_diversities),
        'diversity_definition': (
            'mean over pockets of 1 - mean pairwise Morgan-fingerprint Tanimoto similarity '
            'within the generated molecules for that pocket'
        ),
        'vina_dock_success_fraction': vina_dock_success_fraction,
        'vina_dock_num_docked': int(len(docking_affinities)),
        'vina_score_mean': _mean(score_only_affinities),
        'vina_min_mean': _mean(minimize_affinities),
        'vina_dock_mean': _mean(docking_affinities),
        'vina_dock_median': _median(docking_affinities),
        'docking_elapsed_sec': float(docking_payload['elapsed_sec']),
        'drugclip_rescore_failed_count': int(sum(
            1 for base_valid, final_valid in zip(base_valid_mask, final_valid_mask)
            if base_valid and not final_valid
        )),
        'reward_weights': normalized_reward_weights,
    }
    for key in ['qed_mean', 'sa_score_mean', 'soft_reward_mean', 'diversity', 'vina_dock_mean']:
        if not math.isfinite(float(row[key])):
            raise ValueError(f'Non-finite {key} for task {task["task_id"]}: {row[key]}')
    return row


def _display_name(name):
    return {
        'original_5500': 'Original 5500',
        'grpo_1000': 'GRPO 1000',
        'sgrpo_1000': 'SGRPO 1000',
        'grpo_divreg005_1000': 'GRPO DivReg 0.05 1000',
        'grpo_unidock_500': 'GRPO + UniDock 500',
        'grpo_unidock_1000': 'GRPO + UniDock 1000',
        'mmgenmol_dmb_1000': 'Diverse Mini-Batch GRPO + UniDock 1000',
        'mmgenmol_entropy_1000': 'Entropy-Regularized GRPO + UniDock 1000',
        'grpo_unidock_hbd_st07_sc04_1000': 'GRPO + UniDock + HBD 1000',
        'sgrpo_unidock_500': 'SGRPO + UniDock 500',
        'sgrpo_unidock_1000': 'SGRPO + UniDock 1000',
        'sgrpo_unidock_rewardsum_loo_500': 'SGRPO + UniDock RewardSum LOO 500',
        'sgrpo_unidock_rewardsum_loo_1000': 'SGRPO + UniDock RewardSum LOO 1000',
        'grpo_drugclip_1000': 'GRPO + DrugCLIP 1000',
    }.get(name, name)


def _model_order(name):
    order = {
        'original_5500': 0,
        'grpo_1000': 1,
        'grpo_divreg005_1000': 2,
        'grpo_unidock_500': 3,
        'grpo_unidock_1000': 4,
        'mmgenmol_dmb_1000': 5,
        'mmgenmol_entropy_1000': 6,
        'grpo_unidock_hbd_st07_sc04_1000': 7,
        'sgrpo_1000': 8,
        'sgrpo_unidock_500': 9,
        'sgrpo_unidock_1000': 10,
        'sgrpo_unidock_rewardsum_loo_500': 11,
        'sgrpo_unidock_rewardsum_loo_1000': 12,
        'grpo_drugclip_1000': 13,
    }
    return order.get(name, len(order))


def _series_style(series_index):
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color')
    if not color_cycle:
        raise ValueError('Matplotlib color cycle is empty')
    color_index = series_index % len(color_cycle)
    marker_index = (series_index // len(color_cycle)) % len(MARKER_CYCLE)
    return {
        'color': color_cycle[color_index],
        'marker': MARKER_CYCLE[marker_index],
    }


def _sorted_model_names(rows):
    model_names = []
    for row in sorted(rows, key=lambda item: (_model_order(item['model_name']), item['model_name'])):
        if row['model_name'] not in model_names:
            model_names.append(row['model_name'])
    return model_names


def _resolve_metric_value(row, metric_key, *, with_unidock_baseline=False):
    if metric_key == 'unidock_score_mean':
        value = row.get('unidock_score_mean')
        if value is not None and math.isfinite(float(value)):
            return float(value)
        vina_dock_mean = row.get('vina_dock_mean')
        if vina_dock_mean is None or not math.isfinite(float(vina_dock_mean)):
            raise ValueError(
                f"Cannot derive {metric_key} for model={row['model_name']} "
                f"sweep_type={row['sweep_type']} sweep_value={row['sweep_value']}"
            )
        return float(unidock_affinity_to_score(float(vina_dock_mean)))
    if metric_key == 'soft_reward_mean' and with_unidock_baseline and not _uses_unidock_reward(row['model_name']):
        unidock_score_value = _resolve_metric_value(row, 'unidock_score_mean', with_unidock_baseline=False)
        return float(
            compute_soft_reward(
                row['qed_mean'],
                row['sa_score_mean'],
                unidock_score_value=unidock_score_value,
                reward_weights=WITH_UNIDOCK_PLOT_REWARD_WEIGHTS,
            )
        )
    value = row.get(metric_key)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(
            f"Non-finite {metric_key} for model={row['model_name']} "
            f"sweep_type={row['sweep_type']} sweep_value={row['sweep_value']}: {value}"
        )
    return float(value)


def _plot_metric(
    rows,
    sweep_type,
    metric_key,
    metric_label,
    output_path,
    *,
    plot_title_prefix,
    model_names=None,
    with_unidock_baseline=False,
):
    sweep_rows = [row for row in rows if row['sweep_type'] == sweep_type]
    if not sweep_rows:
        raise ValueError(f'No rows for sweep_type={sweep_type!r}')
    if model_names is None:
        model_names = _sorted_model_names(sweep_rows)
    else:
        model_names = [name for name in model_names if any(row['model_name'] == name for row in sweep_rows)]
    if not model_names:
        raise ValueError(f'No model rows remain for sweep_type={sweep_type!r} metric={metric_key!r}')
    fig, ax = plt.subplots(figsize=(8, 6))
    for series_index, model_name in enumerate(model_names):
        model_rows = [row for row in sweep_rows if row['model_name'] == model_name]
        model_rows.sort(key=lambda row: float(row['sweep_value']))
        x_values = [
            _resolve_metric_value(row, metric_key, with_unidock_baseline=with_unidock_baseline)
            for row in model_rows
        ]
        y_values = [float(row['diversity']) for row in model_rows]
        style = _series_style(series_index)
        ax.plot(
            x_values,
            y_values,
            color=style['color'],
            marker=style['marker'],
            linewidth=2,
            label=_display_name(model_name),
        )
        for row, x_value, y_value in zip(model_rows, x_values, y_values):
            ax.annotate(
                f"{float(row['sweep_value']):.1f}",
                (x_value, y_value),
                textcoords='offset points',
                xytext=(4, 4),
                fontsize=8,
            )
    ax.set_title(f'{plot_title_prefix} {sweep_type}: {metric_label} vs Diversity')
    ax.set_xlabel(metric_label)
    ax.set_ylabel('Mean Per-Pocket Internal Diversity')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_plot_jobs(rows, *, output_dir, plot_name_prefix, plot_title_prefix, plot_suffix, sweep_types):
    all_model_names = _sorted_model_names(rows)
    no_unidock_model_names = [name for name in all_model_names if not _uses_unidock_reward(name)]
    with_unidock_model_names = [name for name in all_model_names if _uses_unidock_reward(name)]
    with_unidock_plus_original_model_names = list(with_unidock_model_names)
    if ORIGINAL_MODEL_NAME in all_model_names and ORIGINAL_MODEL_NAME not in with_unidock_plus_original_model_names:
        with_unidock_plus_original_model_names = [ORIGINAL_MODEL_NAME] + with_unidock_plus_original_model_names
    plot_jobs = []
    for sweep_type in sweep_types:
        for metric_key, metric_label in [
            ('qed_mean', 'QED'),
            ('sa_score_mean', 'SA Score'),
        ]:
            plot_jobs.append(
                (
                    f'all-model {sweep_type} {metric_label} vs diversity',
                    os.path.join(
                        output_dir,
                        f'{plot_name_prefix}_{sweep_type}_diversity_vs_{metric_key}_{plot_suffix}.png',
                    ),
                    sweep_type,
                    metric_key,
                    metric_label,
                    f'{plot_title_prefix} All Models',
                    all_model_names,
                    False,
                )
            )
        if no_unidock_model_names:
            plot_jobs.append(
                (
                    f'no-UniDock {sweep_type} Soft Reward vs diversity',
                    os.path.join(
                        output_dir,
                        f'{plot_name_prefix}_no_unidock_{sweep_type}_diversity_vs_soft_reward_mean_{plot_suffix}.png',
                    ),
                    sweep_type,
                    'soft_reward_mean',
                    'Soft Reward',
                    f'{plot_title_prefix} No-UniDock',
                    no_unidock_model_names,
                    False,
                )
            )
        if with_unidock_plus_original_model_names:
            for metric_key, metric_label in [
                ('unidock_score_mean', VINA_DERIVED_DOCKING_REWARD_LABEL),
                ('soft_reward_mean', 'Soft Reward'),
            ]:
                plot_jobs.append(
                    (
                        f'with-UniDock {sweep_type} {metric_label} vs diversity',
                        os.path.join(
                            output_dir,
                            f'{plot_name_prefix}_with_unidock_{sweep_type}_diversity_vs_{metric_key}_{plot_suffix}.png',
                        ),
                        sweep_type,
                        metric_key,
                        metric_label,
                        f'{plot_title_prefix} With-UniDock + Original Baseline',
                        with_unidock_plus_original_model_names,
                        True,
                    )
                )
    return plot_jobs


def _build_markdown(rows, plot_paths, json_path, rows_path):
    include_unidock_column = any(
        row.get('unidock_score_mean') is not None and math.isfinite(float(row['unidock_score_mean']))
        for row in rows
    )
    lines = [
        '# mmGenMol Sweep Results',
        '',
        f'- `summary_json`: `{json_path}`',
        f'- `raw_rows_jsonl`: `{rows_path}`',
        '- `num_pockets`: 100',
        '- `samples_per_pocket`: 16',
        '- `docking_mode`: `vina_dock`',
        '- `diversity`: per sweep point, compute internal diversity separately within each pocket group, then average over pockets.',
        '- `qed_mean` and `sa_score_mean`: means over valid generated molecules in the sweep point.',
        '- `unidock_score_mean`: legacy field name. The value is a Vina-derived docking reward proxy computed by transforming offline `vina_dock` affinity with `unidock_affinity_to_score`; it is reported only for the UniDock-trained model family.',
        '- `soft_reward_mean`: computed per model with that model family\'s registered training-time reward weights.',
        '- `vina_dock_mean`: mean Vina dock affinity over successful dockings; lower is better.',
        '',
    ]
    if include_unidock_column:
        lines.extend(
            [
                '| Model | Sweep | Value | Diversity | QED | SA Score | UniDock Reward | Soft Reward | Vina Dock Mean | Dock Success | Valid Fraction |',
                '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
            ]
        )
    else:
        lines.extend(
            [
                '| Model | Sweep | Value | Diversity | QED | SA Score | Soft Reward | Vina Dock Mean | Dock Success | Valid Fraction |',
                '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
            ]
        )
    for row in rows:
        cells = [
            _display_name(row['model_name']),
            row['sweep_type'],
            _format_metric(row['sweep_value']),
            _format_metric(row['diversity']),
            _format_metric(row['qed_mean']),
            _format_metric(row['sa_score_mean']),
        ]
        if include_unidock_column:
            cells.append(_format_metric(row.get('unidock_score_mean', float('nan'))))
        cells.extend(
            [
                _format_metric(row['soft_reward_mean']),
                _format_metric(row['vina_dock_mean']),
                _format_metric(row['vina_dock_success_fraction']),
                _format_metric(row['valid_fraction']),
            ]
        )
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')
    for title, path in plot_paths:
        lines.extend([f'## {title}', '', f'![{title}]({os.path.basename(path)})', ''])
    return '\n'.join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks_path', required=True)
    parser.add_argument('--docking_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--output_prefix', default='mmgenmol_sweep_results_20260423')
    parser.add_argument('--expected_num_tasks', type=int, default=32)
    parser.add_argument('--expected_rows_per_task', type=int, default=1600)
    parser.add_argument('--expected_num_pockets', type=int, default=100)
    parser.add_argument('--expected_samples_per_pocket', type=int, default=16)
    parser.add_argument('--docking_mode', default='vina_dock')
    parser.add_argument('--plot_name_prefix', default='mmgenmol')
    parser.add_argument('--plot_title_prefix', default='mmGenMol')
    parser.add_argument('--manifest_path')
    parser.add_argument('--manifest_split', default='test')
    parser.add_argument('--crossdocked_lmdb_path')
    parser.add_argument('--drugclip_checkpoint_path')
    parser.add_argument('--drugclip_device', default='cuda')
    parser.add_argument('--drugclip_batch_size', type=int, default=64)
    parser.add_argument('--drugclip_max_pocket_atoms', type=int, default=256)
    parser.add_argument('--drugclip_num_conformers', type=int, default=1)
    parser.add_argument('--drugclip_conformer_num_workers', type=int, default=1)
    parser.add_argument('--drugclip_use_fp16', action='store_true')
    parser.add_argument('--qed_weight', type=float, default=0.6)
    parser.add_argument('--sa_score_weight', type=float, default=0.4)
    parser.add_argument('--drugclip_score_weight', type=float, default=0.0)
    return parser.parse_args()


def _resolve_plot_suffix(output_prefix):
    parts = output_prefix.rsplit('_', 1)
    if len(parts) == 2 and len(parts[1]) == 8 and parts[1].isdigit():
        return parts[1]
    return output_prefix


def main():
    args = parse_args()
    tasks = _parse_task_manifest(args.tasks_path)
    if len(tasks) != args.expected_num_tasks:
        raise ValueError(f'Expected {args.expected_num_tasks} tasks, found {len(tasks)}')

    from rdkit import Chem, DataStructs
    from rdkit.Chem import QED, rdFingerprintGenerator
    from tdc import Oracle

    sa_oracle = Oracle('sa')
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    args.pocket_entries_by_source_index = None
    args.drugclip_scorer = None
    if args.drugclip_score_weight > 0.0:
        from genmol.mm.utils import DrugCLIPConfig, DrugCLIPScorer

        _require_file(args.manifest_path, 'manifest_path')
        _require_file(args.crossdocked_lmdb_path, 'crossdocked_lmdb_path')
        _require_file(args.drugclip_checkpoint_path, 'drugclip_checkpoint_path')
        args.pocket_entries_by_source_index = _build_pocket_entry_index(args.manifest_path, args.manifest_split)
        args.drugclip_scorer = DrugCLIPScorer(
            DrugCLIPConfig(
                checkpoint_path=args.drugclip_checkpoint_path,
                crossdocked_lmdb_path=args.crossdocked_lmdb_path,
                device=args.drugclip_device,
                batch_size=args.drugclip_batch_size,
                max_pocket_atoms=args.drugclip_max_pocket_atoms,
                num_conformers=args.drugclip_num_conformers,
                conformer_num_workers=args.drugclip_conformer_num_workers,
                use_fp16=args.drugclip_use_fp16,
                seed=42,
            )
        )

    try:
        rows = [
            _compute_task_metrics(
                task=task,
                args=args,
                chem=Chem,
                qed=QED,
                sa_oracle=sa_oracle,
                fingerprint_generator=fingerprint_generator,
                data_structs=DataStructs,
            )
            for task in tasks
        ]
    finally:
        if args.drugclip_scorer is not None:
            args.drugclip_scorer.close()
    rows.sort(key=lambda row: (_model_order(row['model_name']), row['model_name'], row['sweep_type'], row['sweep_value']))

    os.makedirs(args.output_dir, exist_ok=True)
    output_json_path = os.path.join(args.output_dir, f'{args.output_prefix}.json')
    output_rows_path = os.path.join(args.output_dir, f'{args.output_prefix}.rows.jsonl')
    output_markdown_path = os.path.join(args.output_dir, f'{args.output_prefix}.md')

    _write_json(output_json_path, rows)
    _write_jsonl(output_rows_path, rows)

    plot_paths = []
    present_sweep_types = {row['sweep_type'] for row in rows}
    sweep_types = [sweep_type for sweep_type in ['randomness', 'temperature', 'paired'] if sweep_type in present_sweep_types]
    if not sweep_types:
        raise ValueError('No supported sweep types found; expected at least one of randomness or temperature')
    plot_suffix = _resolve_plot_suffix(args.output_prefix)
    for (
        title,
        plot_path,
        sweep_type,
        metric_key,
        metric_label,
        title_prefix,
        model_names,
        with_unidock_baseline,
    ) in _build_plot_jobs(
        rows,
        output_dir=args.output_dir,
        plot_name_prefix=args.plot_name_prefix,
        plot_title_prefix=args.plot_title_prefix,
        plot_suffix=plot_suffix,
        sweep_types=sweep_types,
    ):
        _plot_metric(
            rows,
            sweep_type,
            metric_key,
            metric_label,
            plot_path,
            plot_title_prefix=title_prefix,
            model_names=model_names,
            with_unidock_baseline=with_unidock_baseline,
        )
        plot_paths.append((title, plot_path))

    markdown = _build_markdown(
        rows,
        plot_paths,
        output_json_path,
        output_rows_path,
    )
    with open(output_markdown_path, 'w') as handle:
        handle.write(markdown)

    print(json.dumps({
        'output_json_path': output_json_path,
        'output_rows_path': output_rows_path,
        'output_markdown_path': output_markdown_path,
        'plot_paths': [path for _, path in plot_paths],
        'num_rows': len(rows),
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
