import argparse
import csv
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

import torch
import yaml

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.realpath('.'), 'src'))

from progen2.data.prompts import load_prompt_texts
from progen2.evaluation import (
    classify_protein_sequence,
    global_edit_diversity_parallel,
    nanmean,
)
from progen2.modeling.wrapper import OfficialProGen2CausalLM
from progen2.rewards.composite import normalize_protein_reward_weights
from progen2.rewards.developability import ProteinSolScorer
from progen2.rewards.foldability import ESMFoldFoldabilityScorer
from progen2.rewards.naturalness import ESM2NaturalnessScorer
from progen2.rewards.stability import TemBERTureTmScorer
from progen2.rl.policy import ProGen2Policy


logger = logging.getLogger(__name__)
_PYPLOT = None
_MARKER_CYCLE = ('o', '^', 's', 'D', 'P', 'X', 'v', '<', '>')
_AGG_NATURALNESS_INDEX = None
_AGG_STABILITY_INDEX = None
_AGG_FOLDABILITY_INDEX = None
_AGG_DEVELOPABILITY_INDEX = None
_AGG_NUM_SAMPLES = None
POINT_TASK_FIELDNAMES = (
    'task_id',
    'experiment',
    'display_name',
    'checkpoint_dir',
    'checkpoint_subdir',
    'naturalness_weight',
    'foldability_weight',
    'stability_weight',
    'developability_weight',
    'temperature',
    'generation_rows_path',
    'foldability_scores_path',
    'developability_scores_path',
    'diversity_scores_path',
)


@dataclass(frozen=True)
class SweepExperimentConfig:
    name: str
    checkpoint_dir: str
    display_name: str | None = None
    checkpoint_subdir: str | None = None
    naturalness: float | None = None
    foldability: float | None = None
    stability: float | None = None
    developability: float | None = None


@dataclass(frozen=True)
class SweepConfig:
    tasks_path: str
    generation_output_root: str
    foldability_output_root: str
    developability_output_root: str
    diversity_output_root: str
    packed_naturalness_scores_path: str
    packed_stability_scores_path: str
    output_markdown_path: str
    output_json_path: str
    output_rows_path: str
    output_naturalness_diversity_plot_path: str
    output_foldability_diversity_plot_path: str
    output_stability_diversity_plot_path: str
    output_developability_diversity_plot_path: str
    output_soft_reward_diversity_plot_path: str
    official_code_dir: str
    tokenizer_path: str
    prompt_path: str
    rewards: dict
    experiments: list[SweepExperimentConfig]
    seed: int = 42
    bf16: bool = False
    device: str = 'cuda'
    num_samples: int = 512
    generation_prompt_batch_size: int = 1
    num_return_sequences: int = 512
    max_new_tokens: int = 256
    top_p: float = 0.95
    temperature_values: list[float] | None = None
    calibration_temperature: float = 0.8
    reward_calibration_size: int = 256
    reward_calibration_prompt_batch_size: int = 128


def configure_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )


def _ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _get_pyplot():
    global _PYPLOT
    if _PYPLOT is None:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        _PYPLOT = plt
    return _PYPLOT


def _series_style(series_index):
    plt = _get_pyplot()
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color')
    if not color_cycle:
        raise ValueError('Matplotlib color cycle is empty')
    color_index = series_index % len(color_cycle)
    marker_index = (series_index // len(color_cycle)) % len(_MARKER_CYCLE)
    return {
        'color': color_cycle[color_index],
        'marker': _MARKER_CYCLE[marker_index],
    }


def _display_name(experiment):
    return experiment.display_name or experiment.name


def _format_metric(value):
    if value is None:
        return 'nan'
    value = float(value)
    if math.isnan(value):
        return 'nan'
    return f'{value:.6f}'


def _resolve_temperature_values(raw):
    values = raw.get('temperature_values')
    if values is None:
        raise ValueError('temperature_values must be provided')
    if not isinstance(values, list) or not values:
        raise ValueError('temperature_values must be a non-empty list')
    return [float(value) for value in values]


def load_config(path, *, validate_checkpoint_dirs=True):
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f'Expected mapping config at {path}, got {type(raw).__name__}')
    experiments = [SweepExperimentConfig(**item) for item in raw.pop('experiments')]
    raw['temperature_values'] = _resolve_temperature_values(raw)
    config = SweepConfig(experiments=experiments, **raw)
    if not os.path.isdir(config.official_code_dir):
        raise FileNotFoundError(f'official_code_dir not found: {config.official_code_dir}')
    if not os.path.isfile(config.tokenizer_path):
        raise FileNotFoundError(f'tokenizer_path not found: {config.tokenizer_path}')
    if not os.path.isfile(config.prompt_path):
        raise FileNotFoundError(f'prompt_path not found: {config.prompt_path}')
    if config.num_samples <= 0:
        raise ValueError(f'num_samples must be positive, got {config.num_samples}')
    if config.generation_prompt_batch_size <= 0:
        raise ValueError('generation_prompt_batch_size must be positive')
    if config.num_return_sequences <= 0:
        raise ValueError('num_return_sequences must be positive')
    if config.max_new_tokens <= 0:
        raise ValueError('max_new_tokens must be positive')
    if config.num_samples != config.generation_prompt_batch_size * config.num_return_sequences:
        raise ValueError(
            'num_samples must equal generation_prompt_batch_size * num_return_sequences for the '
            'current sweep pipeline; got '
            f'{config.num_samples} vs {config.generation_prompt_batch_size} * {config.num_return_sequences}'
        )
    if not 0.0 < config.top_p <= 1.0:
        raise ValueError('top_p must be in (0, 1]')
    if config.calibration_temperature <= 0.0:
        raise ValueError('calibration_temperature must be positive')
    if config.reward_calibration_size <= 0:
        raise ValueError('reward_calibration_size must be positive')
    if config.reward_calibration_prompt_batch_size <= 0:
        raise ValueError('reward_calibration_prompt_batch_size must be positive')
    for temperature in config.temperature_values or []:
        if temperature <= 0.0:
            raise ValueError(f'all temperature_values must be positive, got {temperature}')
    for experiment in config.experiments:
        if validate_checkpoint_dirs:
            _validate_checkpoint_dir(experiment.checkpoint_dir)
        normalize_protein_reward_weights(
            {
                'naturalness': experiment.naturalness,
                'foldability': experiment.foldability,
                'stability': experiment.stability,
                'developability': experiment.developability,
            }
        )
    return config


def _validate_checkpoint_dir(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f'checkpoint_dir not found: {checkpoint_dir}')


def resolve_device(device_name):
    if device_name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('device=cuda requested but CUDA is not available')
        return torch.device('cuda')
    return torch.device(device_name)


def _cycle_prompt_batch(prompts, batch_size, start_index):
    return [prompts[(start_index + offset) % len(prompts)] for offset in range(batch_size)]


def _write_jsonl(path, payloads):
    _ensure_parent_dir(path)
    temporary_path = f'{path}.tmp.{os.getpid()}'
    with open(temporary_path, 'w') as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, sort_keys=True) + '\n')
    os.replace(temporary_path, path)


def _write_json(path, payload):
    _ensure_parent_dir(path)
    temporary_path = f'{path}.tmp.{os.getpid()}'
    with open(temporary_path, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary_path, path)


def _read_jsonl(path):
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _packed_reward_base_path(config, reward_name):
    if reward_name == 'naturalness':
        return config.packed_naturalness_scores_path
    if reward_name == 'stability':
        return config.packed_stability_scores_path
    raise ValueError(
        f'Packed GPU reward only supports naturalness/stability, got {reward_name}'
    )


def _packed_reward_experiment_path(config, reward_name, experiment_name):
    if (
        not experiment_name
        or os.path.basename(experiment_name) != experiment_name
        or experiment_name in {'.', '..'}
    ):
        raise ValueError(f'Unsafe experiment name for packed reward path: {experiment_name!r}')
    base_path = _packed_reward_base_path(config, reward_name)
    return os.path.join(
        os.path.dirname(base_path),
        'by_experiment',
        experiment_name,
        os.path.basename(base_path),
    )


def _packed_reward_summary_path(output_path):
    return f'{output_path}.summary.json'


def _write_packed_reward_summary(
    output_path,
    reward_name,
    experiment_names,
    task_ids,
    num_output_rows,
):
    _write_json(
        _packed_reward_summary_path(output_path),
        {
            'schema_version': 1,
            'reward_name': reward_name,
            'experiments': list(experiment_names),
            'task_ids': list(map(int, task_ids)),
            'num_output_rows': int(num_output_rows),
            'output_bytes': int(os.path.getsize(output_path)),
        },
    )


def _expected_valid_sample_keys(tasks, expected_rows_per_task=None):
    keys = set()
    for task in tasks:
        task_id = int(task['task_id'])
        rows = _read_jsonl(task['generation_rows_path'])
        if expected_rows_per_task is not None and len(rows) != expected_rows_per_task:
            raise RuntimeError(
                f'Generation task {task_id} has {len(rows)} rows, '
                f'expected {expected_rows_per_task}'
            )
        for row in rows:
            row_task_id = int(row['task_id'])
            if row_task_id != task_id:
                raise RuntimeError(
                    f'Generation row task_id mismatch for task {task_id}: {row_task_id}'
                )
            if not row['is_valid']:
                continue
            key = (task_id, int(row['sample_index']))
            if key in keys:
                raise RuntimeError(f'Duplicate valid generation key: {key}')
            keys.add(key)
    return keys


def _validated_packed_reward_rows(
    output_path,
    reward_name,
    experiment_name,
    tasks,
    expected_rows_per_task,
):
    summary_path = _packed_reward_summary_path(output_path)
    if not os.path.isfile(output_path) or not os.path.isfile(summary_path):
        raise FileNotFoundError(
            f'Missing packed {reward_name} output or summary for '
            f'{experiment_name}: {output_path}'
        )
    with open(summary_path) as handle:
        summary = json.load(handle)
    expected_task_ids = sorted(int(task['task_id']) for task in tasks)
    expected_summary = {
        'schema_version': 1,
        'reward_name': reward_name,
        'experiments': [experiment_name],
        'task_ids': expected_task_ids,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise RuntimeError(
                f'Packed {reward_name} summary mismatch for {experiment_name}: '
                f'{key}={summary.get(key)!r}, expected {expected!r}'
            )
    if int(summary.get('output_bytes', -1)) != os.path.getsize(output_path):
        raise RuntimeError(
            f'Packed {reward_name} output size mismatch for {experiment_name}'
        )

    rows = _read_jsonl(output_path)
    if int(summary.get('num_output_rows', -1)) != len(rows):
        raise RuntimeError(
            f'Packed {reward_name} row-count mismatch for {experiment_name}'
        )
    expected_keys = _expected_valid_sample_keys(tasks, expected_rows_per_task)
    actual_keys = set()
    for row in rows:
        key = (int(row['task_id']), int(row['sample_index']))
        if key in actual_keys:
            raise RuntimeError(
                f'Duplicate packed {reward_name} key for {experiment_name}: {key}'
            )
        actual_keys.add(key)
        for value_key in (f'{reward_name}_raw', reward_name):
            value = row.get(value_key)
            if value is None or not math.isfinite(float(value)):
                raise RuntimeError(
                    f'Invalid {value_key} for {experiment_name} at {key}: {value!r}'
                )
    if actual_keys != expected_keys:
        raise RuntimeError(
            f'Packed {reward_name} key coverage mismatch for {experiment_name}: '
            f'actual={len(actual_keys)} expected={len(expected_keys)}'
        )
    return rows


def _quantiles(values):
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.numel() == 0:
        raise ValueError('quantile calibration requires at least one value')
    return (
        float(torch.quantile(tensor, 0.10).item()),
        float(torch.quantile(tensor, 0.90).item()),
    )


def _scale_quantile(raw_values, q10, q90):
    denom = (q90 - q10) + 1e-8
    outputs = []
    for value in raw_values:
        normalized = (float(value) - q10) / denom
        outputs.append(max(0.0, min(1.0, normalized)))
    return outputs


def _load_point_tasks(tasks_path):
    tasks = []
    with open(tasks_path) as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if reader.fieldnames != list(POINT_TASK_FIELDNAMES):
            raise ValueError(
                f'Unexpected point task columns in {tasks_path}: {reader.fieldnames} '
                f'vs expected {POINT_TASK_FIELDNAMES}'
            )
        for row in reader:
            tasks.append(
                {
                    **row,
                    'task_id': int(row['task_id']),
                    'temperature': float(row['temperature']),
                    'naturalness_weight': float(row['naturalness_weight']),
                    'foldability_weight': float(row['foldability_weight']),
                    'stability_weight': float(row['stability_weight']),
                    'developability_weight': float(row['developability_weight']),
                    'checkpoint_subdir': row['checkpoint_subdir'] or None,
                }
            )
    if not tasks:
        raise ValueError(f'No point tasks found in {tasks_path}')
    return tasks


def _write_point_tasks(tasks_path, tasks):
    _ensure_parent_dir(tasks_path)
    with open(tasks_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=POINT_TASK_FIELDNAMES, delimiter='\t')
        writer.writeheader()
        for task in tasks:
            writer.writerow({key: task[key] for key in POINT_TASK_FIELDNAMES})


def build_point_tasks(config):
    tasks = []
    task_id = 0
    for experiment in config.experiments:
        weights = normalize_protein_reward_weights(
            {
                'naturalness': experiment.naturalness,
                'foldability': experiment.foldability,
                'stability': experiment.stability,
                'developability': experiment.developability,
            }
        )
        for temperature in config.temperature_values or []:
            sweep_dir = os.path.join(
                experiment.name,
                f'temperature_{temperature:.1f}',
            )
            tasks.append(
                {
                    'task_id': task_id,
                    'experiment': experiment.name,
                    'display_name': _display_name(experiment),
                    'checkpoint_dir': experiment.checkpoint_dir,
                    'checkpoint_subdir': experiment.checkpoint_subdir or '',
                    'naturalness_weight': weights['naturalness'],
                    'foldability_weight': weights['foldability'],
                    'stability_weight': weights['stability'],
                    'developability_weight': weights['developability'],
                    'temperature': temperature,
                    'generation_rows_path': os.path.join(
                        config.generation_output_root,
                        sweep_dir,
                        'generated.rows.jsonl',
                    ),
                    'foldability_scores_path': os.path.join(
                        config.foldability_output_root,
                        sweep_dir,
                        'foldability.rows.jsonl',
                    ),
                    'developability_scores_path': os.path.join(
                        config.developability_output_root,
                        sweep_dir,
                        'developability.rows.jsonl',
                    ),
                    'diversity_scores_path': os.path.join(
                        config.diversity_output_root,
                        sweep_dir,
                        'diversity.json',
                    ),
                }
            )
            task_id += 1
    return tasks


def _generate_rows(policy, prompts, config, seed, temperature):
    rows = []
    prompt_batch = _cycle_prompt_batch(prompts, config.generation_prompt_batch_size, 0)
    rollout = policy.generate_rollouts(
        prompt_batch,
        num_return_sequences=config.num_return_sequences,
        max_new_tokens=config.max_new_tokens,
        top_p=config.top_p,
        temperature=float(temperature),
        seed=seed,
    )
    for sample_index, (prompt_text, decoded_text, raw_sequence) in enumerate(
        zip(rollout.prompt_texts, rollout.decoded_texts, rollout.protein_sequences)
    ):
        classification = classify_protein_sequence(raw_sequence)
        rows.append(
            {
                'sample_index': sample_index,
                'prompt_text': prompt_text,
                'decoded_text': decoded_text,
                'raw_sequence': raw_sequence,
                'sequence': classification['sequence'],
                'is_valid': classification['is_valid'],
                'invalid_reason': classification['invalid_reason'],
            }
        )
    if len(rows) != config.num_samples:
        raise RuntimeError(f'Generated {len(rows)} rows, expected {config.num_samples}')
    return rows


def _collect_calibration_sequences(policy, prompts, config, seed):
    collected = []
    prompt_cursor = 0
    attempts = 0
    max_batches = max(
        8,
        math.ceil(config.reward_calibration_size / config.reward_calibration_prompt_batch_size) * 20,
    )
    while len(collected) < config.reward_calibration_size:
        if attempts >= max_batches:
            raise RuntimeError(
                'Failed to collect enough valid calibration sequences before hitting the '
                f'maximum attempt budget: collected={len(collected)} '
                f'target={config.reward_calibration_size} max_batches={max_batches}'
            )
        remaining = config.reward_calibration_size - len(collected)
        prompt_batch_size = min(config.reward_calibration_prompt_batch_size, remaining)
        prompt_batch = _cycle_prompt_batch(prompts, prompt_batch_size, prompt_cursor)
        prompt_cursor = (prompt_cursor + len(prompt_batch)) % len(prompts)
        rollout = policy.generate_rollouts(
            prompt_batch,
            num_return_sequences=1,
            max_new_tokens=config.max_new_tokens,
            top_p=config.top_p,
            temperature=config.calibration_temperature,
            seed=seed + attempts,
        )
        for sequence in rollout.protein_sequences:
            classification = classify_protein_sequence(sequence)
            if classification['is_valid']:
                collected.append(classification['sequence'])
        attempts += 1
    return collected[: config.reward_calibration_size]


def _instantiate_policy(task, config, device):
    model = OfficialProGen2CausalLM(
        official_code_dir=config.official_code_dir,
        checkpoint_dir=task['checkpoint_dir'],
        tokenizer_path=config.tokenizer_path,
        checkpoint_subdir=task['checkpoint_subdir'],
        device=device,
        use_fp16=False,
        autocast_dtype=torch.bfloat16 if config.bf16 and device.type == 'cuda' else None,
    )
    return ProGen2Policy(model, trainable=False)


def _instantiate_gpu_scorer(config, reward_name):
    reward_cfg = dict(config.rewards[reward_name])
    device_name = reward_cfg.get('device', 'cuda')
    if reward_name == 'naturalness':
        return ESM2NaturalnessScorer(
            model_name=str(reward_cfg['model_name']),
            device=device_name,
            batch_size=int(reward_cfg['batch_size']),
        )
    if reward_name == 'stability':
        return TemBERTureTmScorer(
            model_name_or_path=str(reward_cfg['model_name_or_path']),
            tokenizer_name_or_path=reward_cfg.get('tokenizer_name_or_path'),
            device=device_name,
            batch_size=int(reward_cfg['batch_size']),
            base_model_name_or_path=reward_cfg.get('base_model_name_or_path'),
        )
    if reward_name == 'foldability':
        return ESMFoldFoldabilityScorer(
            device=device_name,
            batch_size=int(reward_cfg['batch_size']),
            num_recycles=reward_cfg.get('num_recycles'),
        )
    raise ValueError(f'Unsupported GPU reward {reward_name!r}')


def _score_packed_calibrated_gpu_reward(config, tasks, reward_name, output_path):
    if reward_name not in {'naturalness', 'stability'}:
        raise ValueError(f'Packed calibrated GPU reward only supports naturalness/stability, got {reward_name}')
    device = resolve_device('cuda')
    prompts = load_prompt_texts(config.prompt_path)
    scorer = _instantiate_gpu_scorer(config, reward_name)
    payloads = []
    tasks_by_experiment = {}
    for task in tasks:
        tasks_by_experiment.setdefault(task['experiment'], []).append(task)
    experiments_by_name = {experiment.name: experiment for experiment in config.experiments}
    for experiment_name, experiment_tasks in tasks_by_experiment.items():
        experiment = experiments_by_name[experiment_name]
        logger.info('Scoring %s for experiment=%s across %d points', reward_name, experiment_name, len(experiment_tasks))
        policy = _instantiate_policy(experiment_tasks[0], config, device)
        calibration_sequences = _collect_calibration_sequences(
            policy,
            prompts,
            config,
            seed=config.seed + (100000 if reward_name == 'stability' else 200000),
        )
        calibration_raw = list(map(float, scorer.score_raw(calibration_sequences)))
        q10, q90 = _quantiles(calibration_raw)

        sequence_records = []
        for task in experiment_tasks:
            rows = _read_jsonl(task['generation_rows_path'])
            for row in rows:
                if row['is_valid']:
                    sequence_records.append(
                        {
                            'task_id': int(task['task_id']),
                            'sample_index': int(row['sample_index']),
                            'sequence': row['sequence'],
                        }
                    )
        if not sequence_records:
            raise RuntimeError(f'No valid sequences found for experiment={experiment_name}')
        raw_scores = list(map(float, scorer.score_raw([record['sequence'] for record in sequence_records])))
        scaled_scores = _scale_quantile(raw_scores, q10, q90)
        if len(raw_scores) != len(sequence_records):
            raise RuntimeError(
                f'{reward_name} scorer returned mismatched score count: '
                f'{len(raw_scores)} vs {len(sequence_records)}'
            )
        for record, raw_score, scaled_score in zip(sequence_records, raw_scores, scaled_scores):
            payloads.append(
                {
                    'task_id': int(record['task_id']),
                    'sample_index': int(record['sample_index']),
                    f'{reward_name}_raw': float(raw_score),
                    reward_name: float(scaled_score),
                    f'{reward_name}_q10': float(q10),
                    f'{reward_name}_q90': float(q90),
                }
            )
        del policy
    scorer.release()
    _write_jsonl(output_path, payloads)
    return payloads


def _score_point_reward_task(config, task, reward_name, output_path):
    rows = _read_jsonl(task['generation_rows_path'])
    valid_rows = [row for row in rows if row['is_valid']]
    valid_sequences = [row['sequence'] for row in valid_rows]
    if reward_name == 'foldability':
        scorer = _instantiate_gpu_scorer(config, 'foldability')
        scores = list(map(float, scorer.score_raw(valid_sequences))) if valid_sequences else []
        scorer.release()
        payloads = [
            {
                'task_id': int(task['task_id']),
                'sample_index': int(row['sample_index']),
                'foldability': float(score),
            }
            for row, score in zip(valid_rows, scores)
        ]
        _write_jsonl(output_path, payloads)
        return
    if reward_name == 'developability':
        reward_cfg = dict(config.rewards['developability'])
        scorer = ProteinSolScorer(
            model_name_or_path=str(reward_cfg['model_name_or_path']),
            tokenizer_name_or_path=reward_cfg.get('tokenizer_name_or_path'),
            device=reward_cfg.get('device', 'cpu'),
            batch_size=int(reward_cfg['batch_size']),
            num_workers=reward_cfg.get('num_workers'),
        )
        scores = list(map(float, scorer.score_raw(valid_sequences))) if valid_sequences else []
        scorer.release()
        payloads = [
            {
                'task_id': int(task['task_id']),
                'sample_index': int(row['sample_index']),
                'developability': float(score),
            }
            for row, score in zip(valid_rows, scores)
        ]
        _write_jsonl(output_path, payloads)
        return
    raise ValueError(f'Unsupported point reward {reward_name!r}')


def _point_diversity_num_workers():
    raw = os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('OMP_NUM_THREADS')
    if raw is None:
        cpu_count = os.cpu_count() or 1
        return max(1, int(cpu_count))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f'Expected integer point diversity worker count, got {raw!r}') from exc
    if value <= 0:
        raise ValueError(f'Point diversity worker count must be positive, got {value}')
    return value


def _score_point_diversity_task(config, task, output_path):
    rows = _read_jsonl(task['generation_rows_path'])
    valid_sequences = []
    seen_valid_sequences = set()
    valid_lengths = []
    for row in rows:
        if row['is_valid']:
            sequence = row['sequence']
            valid_sequences.append(sequence)
            seen_valid_sequences.add(sequence)
            valid_lengths.append(len(sequence))
    num_rows = len(rows)
    if num_rows != config.num_samples:
        raise RuntimeError(
            f'Point task {task["task_id"]} has {num_rows} generated rows, expected {config.num_samples}'
        )
    diversity = float(global_edit_diversity_parallel(valid_sequences, num_workers=_point_diversity_num_workers()))
    payload = {
        'task_id': int(task['task_id']),
        'experiment': task['experiment'],
        'display_name': task['display_name'],
        'temperature': float(task['temperature']),
        'diversity': diversity,
        'valid_fraction': float(len(valid_sequences) / num_rows),
        'invalid_fraction': float(1.0 - (len(valid_sequences) / num_rows)),
        'unique_valid_fraction': 0.0 if not valid_sequences else float(len(seen_valid_sequences) / len(valid_sequences)),
        'unique_overall_fraction': float(len(seen_valid_sequences) / num_rows),
        'mean_valid_length': nanmean(valid_lengths),
    }
    _ensure_parent_dir(output_path)
    with open(output_path, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _index_reward_rows(path, value_keys):
    index = {}
    for row in _read_jsonl(path):
        key = (int(row['task_id']), int(row['sample_index']))
        if key in index:
            raise RuntimeError(f'Duplicate reward row key in {path}: {key}')
        index[key] = {field: row.get(field) for field in value_keys}
    return index


def _load_diversity_payload(path):
    with open(path) as handle:
        payload = json.load(handle)
    required_fields = (
        'task_id',
        'experiment',
        'display_name',
        'temperature',
        'diversity',
        'valid_fraction',
        'invalid_fraction',
        'unique_valid_fraction',
        'unique_overall_fraction',
        'mean_valid_length',
    )
    for field in required_fields:
        if field not in payload:
            raise ValueError(f'Missing diversity field {field!r} in {path}')
    return payload


def _aggregate_num_workers():
    raw = os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('OMP_NUM_THREADS')
    if raw is None:
        cpu_count = os.cpu_count() or 1
        return max(1, int(cpu_count))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f'Expected integer worker count environment variable, got {raw!r}') from exc
    if value <= 0:
        raise ValueError(f'Aggregate worker count must be positive, got {value}')
    return value


def _init_aggregate_worker(naturalness_index, stability_index, foldability_index, developability_index, num_samples):
    global _AGG_NATURALNESS_INDEX
    global _AGG_STABILITY_INDEX
    global _AGG_FOLDABILITY_INDEX
    global _AGG_DEVELOPABILITY_INDEX
    global _AGG_NUM_SAMPLES
    _AGG_NATURALNESS_INDEX = naturalness_index
    _AGG_STABILITY_INDEX = stability_index
    _AGG_FOLDABILITY_INDEX = foldability_index
    _AGG_DEVELOPABILITY_INDEX = developability_index
    _AGG_NUM_SAMPLES = int(num_samples)


def _aggregate_one_point(task):
    rows = _read_jsonl(task['generation_rows_path'])
    sum_nat = 0.0
    sum_fold = 0.0
    sum_stab = 0.0
    sum_dev = 0.0
    sum_soft = 0.0
    nat_raw_values = []
    stab_raw_values = []
    point_rows = []
    diversity_payload = _load_diversity_payload(task['diversity_scores_path'])
    for row in rows:
        key = (int(task['task_id']), int(row['sample_index']))
        nat_payload = _AGG_NATURALNESS_INDEX.get(key, {})
        stab_payload = _AGG_STABILITY_INDEX.get(key, {})
        fold_payload = _AGG_FOLDABILITY_INDEX.get(key, {})
        dev_payload = _AGG_DEVELOPABILITY_INDEX.get(key, {})
        naturalness = float(nat_payload.get('naturalness', 0.0) or 0.0)
        foldability = float(fold_payload.get('foldability', 0.0) or 0.0)
        stability = float(stab_payload.get('stability', 0.0) or 0.0)
        developability = float(dev_payload.get('developability', 0.0) or 0.0)
        soft_reward = (
            float(task['naturalness_weight']) * naturalness
            + float(task['foldability_weight']) * foldability
            + float(task['stability_weight']) * stability
            + float(task['developability_weight']) * developability
        )
        if row['is_valid']:
            if nat_payload.get('naturalness_raw') is not None:
                nat_raw_values.append(float(nat_payload['naturalness_raw']))
            if stab_payload.get('stability_raw') is not None:
                stab_raw_values.append(float(stab_payload['stability_raw']))
        sum_nat += naturalness
        sum_fold += foldability
        sum_stab += stability
        sum_dev += developability
        sum_soft += soft_reward
        point_rows.append(
            {
                'task_id': int(task['task_id']),
                'experiment': task['experiment'],
                'display_name': task['display_name'],
                'temperature': float(task['temperature']),
                **row,
                'naturalness_raw': nat_payload.get('naturalness_raw'),
                'naturalness': naturalness,
                'foldability': foldability,
                'stability_raw': stab_payload.get('stability_raw'),
                'stability': stability,
                'developability': developability,
                'soft_reward': soft_reward,
            }
        )
    num_rows = len(rows)
    if num_rows != _AGG_NUM_SAMPLES:
        raise RuntimeError(
            f'Point task {task["task_id"]} has {num_rows} generated rows, expected {_AGG_NUM_SAMPLES}'
        )
    point_result = {
        'task_id': int(task['task_id']),
        'experiment': task['experiment'],
        'display_name': task['display_name'],
        'temperature': float(task['temperature']),
        'soft_reward_mean': float(sum_soft / num_rows),
        'reward_nat_mean': float(sum_nat / num_rows),
        'reward_fold_mean': float(sum_fold / num_rows),
        'reward_stab_mean': float(sum_stab / num_rows),
        'reward_dev_mean': float(sum_dev / num_rows),
        'diversity': float(diversity_payload['diversity']),
        'valid_fraction': float(diversity_payload['valid_fraction']),
        'invalid_fraction': float(diversity_payload['invalid_fraction']),
        'unique_valid_fraction': float(diversity_payload['unique_valid_fraction']),
        'unique_overall_fraction': float(diversity_payload['unique_overall_fraction']),
        'mean_valid_length': float(diversity_payload['mean_valid_length']),
        'naturalness_raw_mean_valid': nanmean(nat_raw_values),
        'stability_raw_mean_valid': nanmean(stab_raw_values),
        'reward_weights': {
            'naturalness': float(task['naturalness_weight']),
            'foldability': float(task['foldability_weight']),
            'stability': float(task['stability_weight']),
            'developability': float(task['developability_weight']),
        },
    }
    return point_rows, point_result


def _plot_metric_tradeoff(results, experiments, x_key, x_label, y_key, y_label, title, output_path):
    plt = _get_pyplot()
    _ensure_parent_dir(output_path)
    figure, axis = plt.subplots(figsize=(7.5, 5.75), dpi=160)
    experiment_order = {experiment.name: idx for idx, experiment in enumerate(experiments)}
    label_offsets = ((4, 4), (4, -10), (-20, 4), (-20, -10))
    for experiment in experiments:
        experiment_rows = [row for row in results if row['experiment'] == experiment.name]
        if not experiment_rows:
            raise ValueError(f'No rows found for experiment {experiment.name}')
        experiment_rows.sort(key=lambda row: float(row['temperature']))
        style = _series_style(experiment_order[experiment.name])
        x_values = [float(row[x_key]) for row in experiment_rows]
        y_values = [float(row[y_key]) for row in experiment_rows]
        if not all(math.isfinite(value) for value in x_values + y_values):
            raise ValueError(f'Non-finite values encountered while plotting {experiment.name} for {x_key} vs {y_key}')
        axis.plot(
            x_values,
            y_values,
            label=_display_name(experiment),
            color=style['color'],
            marker=style['marker'],
            linewidth=1.8,
            markersize=6.5,
        )
        for row, x_value, y_value in zip(experiment_rows, x_values, y_values):
            x_offset, y_offset = label_offsets[int(row['task_id']) % len(label_offsets)]
            axis.annotate(
                f'{float(row["temperature"]):.1f}',
                xy=(x_value, y_value),
                xytext=(x_offset, y_offset),
                textcoords='offset points',
                color=style['color'],
                fontsize=6.5,
                alpha=0.9,
            )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def _build_markdown(config, results):
    lines = [
        '# ProGen2 Temperature Sweep Evaluation',
        '',
        f'- `num_samples`: {config.num_samples}',
        f'- `generation_prompt_batch_size`: {config.generation_prompt_batch_size}',
        f'- `num_return_sequences`: {config.num_return_sequences}',
        f'- `max_new_tokens`: {config.max_new_tokens}',
        f'- `top_p`: {config.top_p}',
        f'- `temperature_values`: {", ".join(f"{value:.1f}" for value in config.temperature_values or [])}',
        f'- `reward_calibration_size`: {config.reward_calibration_size}',
        f'- `reward_calibration_prompt_batch_size`: {config.reward_calibration_prompt_batch_size}',
        '',
        '| Model | Temperature | Soft Reward | Naturalness | Foldability | Stability | Developability | Diversity | Valid Fraction | Unique Valid Fraction | Mean Valid Length |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in results:
        lines.append(
            '| '
            + ' | '.join(
                [
                    row['display_name'],
                    _format_metric(row['temperature']),
                    _format_metric(row['soft_reward_mean']),
                    _format_metric(row['reward_nat_mean']),
                    _format_metric(row['reward_fold_mean']),
                    _format_metric(row['reward_stab_mean']),
                    _format_metric(row['reward_dev_mean']),
                    _format_metric(row['diversity']),
                    _format_metric(row['valid_fraction']),
                    _format_metric(row['unique_valid_fraction']),
                    _format_metric(row['mean_valid_length']),
                ]
            )
            + ' |'
        )
    lines.extend(
        [
            '',
            'Column notes:',
            '- `Soft Reward` is the rollout-level weighted sum of reward components using the training-time reward weights for each experiment.',
            '- `Naturalness` and `Stability` are calibrated once per experiment using generated calibration sequences, then reused across the full sweep.',
            '- `Diversity` is the global edit-distance diversity over all valid sequences at that sweep point.',
            '- `Valid Fraction` uses the supported 20-residue alphabet check.',
            '',
        ]
    )
    return '\n'.join(lines)


def _aggregate_results(config, tasks):
    naturalness_index = _index_reward_rows(
        config.packed_naturalness_scores_path,
        ('naturalness_raw', 'naturalness', 'naturalness_q10', 'naturalness_q90'),
    )
    stability_index = _index_reward_rows(
        config.packed_stability_scores_path,
        ('stability_raw', 'stability', 'stability_q10', 'stability_q90'),
    )
    expected_valid_keys = _expected_valid_sample_keys(tasks, config.num_samples)
    for reward_name, index in (
        ('naturalness', naturalness_index),
        ('stability', stability_index),
    ):
        actual_keys = set(index)
        if actual_keys != expected_valid_keys:
            raise RuntimeError(
                f'Packed {reward_name} coverage mismatch during aggregation: '
                f'actual={len(actual_keys)} expected={len(expected_valid_keys)}'
            )
    foldability_index = {}
    developability_index = {}
    for task in tasks:
        if os.path.isfile(task['foldability_scores_path']):
            foldability_index.update(_index_reward_rows(task['foldability_scores_path'], ('foldability',)))
        if os.path.isfile(task['developability_scores_path']):
            developability_index.update(_index_reward_rows(task['developability_scores_path'], ('developability',)))

    worker_count = min(len(tasks), _aggregate_num_workers())
    point_results = []
    all_rows = []
    if worker_count == 1:
        _init_aggregate_worker(
            naturalness_index,
            stability_index,
            foldability_index,
            developability_index,
            config.num_samples,
        )
        for task in tasks:
            point_rows, point_result = _aggregate_one_point(task)
            all_rows.extend(point_rows)
            point_results.append(point_result)
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_aggregate_worker,
            initargs=(
                naturalness_index,
                stability_index,
                foldability_index,
                developability_index,
                config.num_samples,
            ),
        ) as executor:
            for point_rows, point_result in executor.map(_aggregate_one_point, tasks):
                all_rows.extend(point_rows)
                point_results.append(point_result)
    return all_rows, point_results


def cmd_build_tasks(args):
    config = load_config(args.config)
    tasks = build_point_tasks(config)
    _write_point_tasks(config.tasks_path, tasks)
    logger.info('Wrote %d point tasks to %s', len(tasks), config.tasks_path)


def cmd_generate_task(args):
    config = load_config(args.config, validate_checkpoint_dirs=False)
    tasks = _load_point_tasks(config.tasks_path)
    task = next((task for task in tasks if int(task['task_id']) == args.task_id), None)
    if task is None:
        raise ValueError(f'No task found for task_id={args.task_id}')
    _validate_checkpoint_dir(task['checkpoint_dir'])
    prompts = load_prompt_texts(config.prompt_path)
    device = resolve_device(config.device)
    policy = _instantiate_policy(task, config, device)
    rows = _generate_rows(
        policy,
        prompts,
        config,
        seed=config.seed + int(task['task_id']),
        temperature=float(task['temperature']),
    )
    payloads = []
    for row in rows:
        payloads.append(
            {
                'task_id': int(task['task_id']),
                'experiment': task['experiment'],
                'display_name': task['display_name'],
                'temperature': float(task['temperature']),
                **row,
            }
        )
    _write_jsonl(task['generation_rows_path'], payloads)
    logger.info('Wrote %d generated rows to %s', len(payloads), task['generation_rows_path'])


def cmd_score_packed_gpu_reward(args):
    experiment_name = getattr(args, 'experiment_name', None)
    config = load_config(
        args.config,
        validate_checkpoint_dirs=experiment_name is None,
    )
    tasks = _load_point_tasks(config.tasks_path)
    if experiment_name is None:
        selected_tasks = tasks
        output_path = _packed_reward_base_path(config, args.reward_name)
    else:
        selected_tasks = [
            task for task in tasks if task['experiment'] == experiment_name
        ]
        if not selected_tasks:
            raise ValueError(f'No point tasks found for experiment={experiment_name!r}')
        checkpoint_dirs = {
            task['checkpoint_dir'] for task in selected_tasks
        }
        if len(checkpoint_dirs) != 1:
            raise RuntimeError(
                f'Experiment {experiment_name!r} references multiple checkpoints: '
                f'{sorted(checkpoint_dirs)}'
            )
        _validate_checkpoint_dir(next(iter(checkpoint_dirs)))
        output_path = _packed_reward_experiment_path(
            config,
            args.reward_name,
            experiment_name,
        )
    payloads = _score_packed_calibrated_gpu_reward(
        config,
        selected_tasks,
        args.reward_name,
        output_path,
    )
    if experiment_name is not None:
        _write_packed_reward_summary(
            output_path=output_path,
            reward_name=args.reward_name,
            experiment_names=(experiment_name,),
            task_ids=sorted(int(task['task_id']) for task in selected_tasks),
            num_output_rows=len(payloads),
        )
    logger.info('Wrote packed %s scores to %s', args.reward_name, output_path)


def cmd_merge_packed_gpu_rewards(args):
    config = load_config(args.config, validate_checkpoint_dirs=False)
    tasks = _load_point_tasks(config.tasks_path)
    task_experiments = {task['experiment'] for task in tasks}
    configured_experiments = [experiment.name for experiment in config.experiments]
    if task_experiments != set(configured_experiments):
        raise RuntimeError(
            'Point-task experiments do not match configured experiments: '
            f'tasks={sorted(task_experiments)} config={configured_experiments}'
        )

    for reward_name in ('naturalness', 'stability'):
        merged_rows = []
        for experiment_name in configured_experiments:
            experiment_tasks = [
                task for task in tasks if task['experiment'] == experiment_name
            ]
            input_path = _packed_reward_experiment_path(
                config,
                reward_name,
                experiment_name,
            )
            merged_rows.extend(
                _validated_packed_reward_rows(
                    output_path=input_path,
                    reward_name=reward_name,
                    experiment_name=experiment_name,
                    tasks=experiment_tasks,
                    expected_rows_per_task=config.num_samples,
                )
            )
        merged_rows.sort(
            key=lambda row: (int(row['task_id']), int(row['sample_index']))
        )
        output_path = _packed_reward_base_path(config, reward_name)
        _write_jsonl(output_path, merged_rows)
        _write_packed_reward_summary(
            output_path=output_path,
            reward_name=reward_name,
            experiment_names=configured_experiments,
            task_ids=sorted(int(task['task_id']) for task in tasks),
            num_output_rows=len(merged_rows),
        )
        logger.info(
            'Merged %d packed %s rows across %d experiments into %s',
            len(merged_rows),
            reward_name,
            len(configured_experiments),
            output_path,
        )


def cmd_score_point_reward_task(args):
    config = load_config(args.config, validate_checkpoint_dirs=False)
    tasks = _load_point_tasks(config.tasks_path)
    task = next((task for task in tasks if int(task['task_id']) == args.task_id), None)
    if task is None:
        raise ValueError(f'No task found for task_id={args.task_id}')
    if args.reward_name == 'foldability':
        output_path = task['foldability_scores_path']
    elif args.reward_name == 'developability':
        output_path = task['developability_scores_path']
    else:
        raise ValueError(f'Point reward task only supports foldability/developability, got {args.reward_name}')
    _score_point_reward_task(config, task, args.reward_name, output_path)
    logger.info('Wrote %s scores to %s', args.reward_name, output_path)


def cmd_score_point_diversity_task(args):
    config = load_config(args.config, validate_checkpoint_dirs=False)
    tasks = _load_point_tasks(config.tasks_path)
    task = next((task for task in tasks if int(task['task_id']) == args.task_id), None)
    if task is None:
        raise ValueError(f'No task found for task_id={args.task_id}')
    _score_point_diversity_task(config, task, task['diversity_scores_path'])
    logger.info('Wrote diversity metrics to %s', task['diversity_scores_path'])


def cmd_aggregate(args):
    config = load_config(args.config, validate_checkpoint_dirs=False)
    tasks = _load_point_tasks(config.tasks_path)
    all_rows, point_results = _aggregate_results(config, tasks)

    experiment_order = {experiment.name: idx for idx, experiment in enumerate(config.experiments)}
    point_results.sort(key=lambda row: (experiment_order[row['experiment']], float(row['temperature'])))

    _plot_metric_tradeoff(
        results=point_results,
        experiments=config.experiments,
        x_key='reward_nat_mean',
        x_label='Naturalness',
        y_key='diversity',
        y_label='Diversity',
        title='Naturalness vs Diversity',
        output_path=config.output_naturalness_diversity_plot_path,
    )
    _plot_metric_tradeoff(
        results=point_results,
        experiments=config.experiments,
        x_key='reward_fold_mean',
        x_label='Foldability',
        y_key='diversity',
        y_label='Diversity',
        title='Foldability vs Diversity',
        output_path=config.output_foldability_diversity_plot_path,
    )
    _plot_metric_tradeoff(
        results=point_results,
        experiments=config.experiments,
        x_key='reward_stab_mean',
        x_label='Stability',
        y_key='diversity',
        y_label='Diversity',
        title='Stability vs Diversity',
        output_path=config.output_stability_diversity_plot_path,
    )
    _plot_metric_tradeoff(
        results=point_results,
        experiments=config.experiments,
        x_key='reward_dev_mean',
        x_label='Developability',
        y_key='diversity',
        y_label='Diversity',
        title='Developability vs Diversity',
        output_path=config.output_developability_diversity_plot_path,
    )
    _plot_metric_tradeoff(
        results=point_results,
        experiments=config.experiments,
        x_key='soft_reward_mean',
        x_label='Soft Reward',
        y_key='diversity',
        y_label='Diversity',
        title='Soft Reward vs Diversity',
        output_path=config.output_soft_reward_diversity_plot_path,
    )

    markdown = _build_markdown(config, point_results)
    payload = {
        'config': {
            'tasks_path': config.tasks_path,
            'temperature_values': config.temperature_values,
            'num_samples': config.num_samples,
            'generation_prompt_batch_size': config.generation_prompt_batch_size,
            'num_return_sequences': config.num_return_sequences,
            'max_new_tokens': config.max_new_tokens,
            'top_p': config.top_p,
            'reward_calibration_size': config.reward_calibration_size,
            'reward_calibration_prompt_batch_size': config.reward_calibration_prompt_batch_size,
        },
        'results': point_results,
    }

    _ensure_parent_dir(config.output_markdown_path)
    with open(config.output_markdown_path, 'w') as handle:
        handle.write(markdown)
    _ensure_parent_dir(config.output_json_path)
    with open(config.output_json_path, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    _write_jsonl(config.output_rows_path, all_rows)
    logger.info('Wrote markdown report to %s', config.output_markdown_path)
    logger.info('Wrote JSON report to %s', config.output_json_path)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', required=True)

    build_tasks = subparsers.add_parser('build-tasks')
    build_tasks.add_argument('--config', required=True)

    generate_task = subparsers.add_parser('generate-task')
    generate_task.add_argument('--config', required=True)
    generate_task.add_argument('--task-id', type=int, required=True)

    packed_gpu_reward = subparsers.add_parser('score-packed-gpu-reward')
    packed_gpu_reward.add_argument('--config', required=True)
    packed_gpu_reward.add_argument('--reward-name', required=True, choices=('naturalness', 'stability'))
    packed_gpu_reward.add_argument('--experiment-name')

    merge_packed_gpu_rewards = subparsers.add_parser('merge-packed-gpu-rewards')
    merge_packed_gpu_rewards.add_argument('--config', required=True)

    point_reward_task = subparsers.add_parser('score-point-reward-task')
    point_reward_task.add_argument('--config', required=True)
    point_reward_task.add_argument('--reward-name', required=True, choices=('foldability', 'developability'))
    point_reward_task.add_argument('--task-id', type=int, required=True)

    point_diversity_task = subparsers.add_parser('score-point-diversity-task')
    point_diversity_task.add_argument('--config', required=True)
    point_diversity_task.add_argument('--task-id', type=int, required=True)

    aggregate = subparsers.add_parser('aggregate')
    aggregate.add_argument('--config', required=True)

    args = parser.parse_args()
    configure_logging()
    if args.mode == 'build-tasks':
        cmd_build_tasks(args)
        return
    if args.mode == 'generate-task':
        cmd_generate_task(args)
        return
    if args.mode == 'score-packed-gpu-reward':
        cmd_score_packed_gpu_reward(args)
        return
    if args.mode == 'merge-packed-gpu-rewards':
        cmd_merge_packed_gpu_rewards(args)
        return
    if args.mode == 'score-point-reward-task':
        cmd_score_point_reward_task(args)
        return
    if args.mode == 'score-point-diversity-task':
        cmd_score_point_diversity_task(args)
        return
    if args.mode == 'aggregate':
        cmd_aggregate(args)
        return
    raise ValueError(f'Unsupported mode {args.mode!r}')


if __name__ == '__main__':
    main()
