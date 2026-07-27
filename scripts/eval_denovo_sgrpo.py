# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.join(os.path.realpath('.'), 'src'))

from genmol.diversity import (
    DEFAULT_DIVERSITY_METRIC,
    MORGAN_INTERNAL_DIVERSITY,
    RELATIVE_SCAFFOLD_DIVERSITY,
    validate_diversity_metric,
)


logger = logging.getLogger(__name__)
_PYPLOT = None
_TORCH = None
_YAML = None
_MARKER_CYCLE = ('o', '^', 's', 'D', 'P', 'X', 'v', '<', '>')


def _get_yaml():
    global _YAML
    if _YAML is None:
        logger.info('Importing yaml')
        import yaml

        _YAML = yaml
    return _YAML


def _get_torch():
    global _TORCH
    if _TORCH is None:
        logger.info('Importing torch')
        import torch

        _TORCH = torch
    return _TORCH


def _get_pyplot():
    global _PYPLOT
    if _PYPLOT is None:
        logger.info('Importing matplotlib.pyplot')
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


@dataclass(frozen=True)
class EvalExperimentConfig:
    name: str
    checkpoint_path: str
    display_name: str | None = None
    qed: float | None = None
    sa_score: float | None = None


@dataclass(frozen=True)
class EvalSweepPairConfig:
    randomness: float
    generation_temperature: float


@dataclass(frozen=True)
class EvalSweepPoint:
    sweep_value: float
    generation_temperature: float
    randomness: float
    sweep_label: str | None = None


@dataclass(frozen=True)
class EvalConfig:
    output_markdown_path: str
    output_json_path: str
    output_qed_diversity_plot_path: str
    output_sa_score_diversity_plot_path: str
    output_soft_reward_diversity_plot_path: str
    output_rows_path: str | None = None
    seed: int = 42
    bf16: bool = True
    device: str = 'cuda'
    num_samples: int = 1000
    generation_batch_size: int = 2048
    generation_temperature: float = 1.0
    generation_temperature_values: list[float] | None = None
    randomness: float = 0.3
    randomness_values: list[float] | None = None
    randomness_temperature_pairs: list[EvalSweepPairConfig] | None = None
    sweep_axis: str = 'randomness'
    min_add_len: int = 60
    max_completion_length: int | None = None
    length_path: str | None = None
    diversity_metric: str = DEFAULT_DIVERSITY_METRIC
    experiments: list[EvalExperimentConfig] | None = None


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


def _resolve_randomness_values(raw_config):
    if 'randomness_values' in raw_config and raw_config['randomness_values'] is not None:
        return list(raw_config['randomness_values'])
    return [float(raw_config.get('randomness', 0.3))]


def _resolve_generation_temperature_values(raw_config):
    if 'generation_temperature_values' not in raw_config:
        return None
    values = raw_config['generation_temperature_values']
    if values is None:
        return None
    return list(values)


def _resolve_randomness_temperature_pairs(raw_config):
    if 'randomness_temperature_pairs' not in raw_config:
        return None
    values = raw_config['randomness_temperature_pairs']
    if values is None:
        return None
    if not isinstance(values, list):
        raise TypeError(
            'randomness_temperature_pairs must be a list of {randomness, generation_temperature} mappings'
        )
    return list(values)


def _resolve_sweep_axis(raw_config):
    randomness_values = _resolve_randomness_values(raw_config) if (
        'randomness_values' in raw_config and raw_config['randomness_values'] is not None
    ) else None
    generation_temperature_values = _resolve_generation_temperature_values(raw_config)
    randomness_temperature_pairs = _resolve_randomness_temperature_pairs(raw_config)
    configured_sweeps = [
        randomness_values is not None,
        generation_temperature_values is not None,
        randomness_temperature_pairs is not None,
    ]
    if sum(configured_sweeps) > 1:
        raise ValueError(
            'Specify exactly one of randomness_values, generation_temperature_values, or randomness_temperature_pairs'
        )
    if randomness_temperature_pairs is not None:
        return 'randomness_temperature_pair', randomness_temperature_pairs
    if generation_temperature_values is not None:
        return 'generation_temperature', generation_temperature_values
    if randomness_values is not None:
        return 'randomness', randomness_values
    return 'randomness', [float(raw_config.get('randomness', 0.3))]


def _get_sweep_values(config):
    if config.sweep_axis == 'randomness':
        return config.randomness_values
    if config.sweep_axis == 'generation_temperature':
        return config.generation_temperature_values
    if config.sweep_axis == 'randomness_temperature_pair':
        return [float(idx) for idx in range(1, len(config.randomness_temperature_pairs or []) + 1)]
    raise ValueError(f'Unsupported sweep_axis: {config.sweep_axis}')


def _format_sweep_label(sweep_axis, value):
    if sweep_axis == 'randomness':
        return f'r={float(value):.1f}'
    if sweep_axis == 'generation_temperature':
        return f't={float(value):.1f}'
    if sweep_axis == 'randomness_temperature_pair':
        return f'pair={int(float(value))}'
    raise ValueError(f'Unsupported sweep_axis: {sweep_axis}')


def _build_sweep_points(config):
    if config.sweep_axis == 'randomness':
        return [
            EvalSweepPoint(
                sweep_value=float(value),
                generation_temperature=float(config.generation_temperature),
                randomness=float(value),
                sweep_label=_format_sweep_label(config.sweep_axis, value),
            )
            for value in config.randomness_values or []
        ]
    if config.sweep_axis == 'generation_temperature':
        return [
            EvalSweepPoint(
                sweep_value=float(value),
                generation_temperature=float(value),
                randomness=float(config.randomness),
                sweep_label=_format_sweep_label(config.sweep_axis, value),
            )
            for value in config.generation_temperature_values or []
        ]
    if config.sweep_axis == 'randomness_temperature_pair':
        return [
            EvalSweepPoint(
                sweep_value=float(idx),
                generation_temperature=float(pair.generation_temperature),
                randomness=float(pair.randomness),
                sweep_label=(
                    f'r={float(pair.randomness):.1f},t={float(pair.generation_temperature):.1f}'
                ),
            )
            for idx, pair in enumerate(config.randomness_temperature_pairs or [], start=1)
        ]
    raise ValueError(f'Unsupported sweep_axis: {config.sweep_axis}')


def load_config(path):
    yaml = _get_yaml()
    logger.info('Loading config from %s', path)
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    experiments = [EvalExperimentConfig(**item) for item in raw.pop('experiments')]
    sweep_axis, sweep_values = _resolve_sweep_axis(raw)
    raw.pop('randomness_values', None)
    raw.pop('generation_temperature_values', None)
    raw.pop('randomness_temperature_pairs', None)
    if sweep_axis == 'randomness':
        raw['randomness_values'] = sweep_values
        raw['generation_temperature_values'] = None
        raw['randomness_temperature_pairs'] = None
    elif sweep_axis == 'generation_temperature':
        raw['randomness_values'] = None
        raw['generation_temperature_values'] = sweep_values
        raw['randomness_temperature_pairs'] = None
    elif sweep_axis == 'randomness_temperature_pair':
        raw['randomness_values'] = None
        raw['generation_temperature_values'] = None
        raw['randomness_temperature_pairs'] = [EvalSweepPairConfig(**item) for item in sweep_values]
    else:
        raise ValueError(f'Unsupported sweep_axis: {sweep_axis}')
    raw['diversity_metric'] = validate_diversity_metric(
        raw.get('diversity_metric', DEFAULT_DIVERSITY_METRIC)
    )
    config = EvalConfig(experiments=experiments, sweep_axis=sweep_axis, **raw)

    if config.num_samples <= 0:
        raise ValueError(f'num_samples must be positive, got {config.num_samples}')
    if config.generation_batch_size <= 0:
        raise ValueError(f'generation_batch_size must be positive, got {config.generation_batch_size}')
    if config.generation_temperature <= 0:
        raise ValueError(f'generation_temperature must be positive, got {config.generation_temperature}')
    if config.randomness <= 0:
        raise ValueError(f'randomness must be positive, got {config.randomness}')
    if config.min_add_len <= 0:
        raise ValueError(f'min_add_len must be positive, got {config.min_add_len}')
    configured_sweeps = [
        config.randomness_values is not None,
        config.generation_temperature_values is not None,
        config.randomness_temperature_pairs is not None,
    ]
    if sum(configured_sweeps) != 1:
        raise ValueError('Exactly one sweep grid must be configured')
    sweep_values = _get_sweep_values(config)
    if sweep_values is None or not sweep_values:
        raise ValueError('Sweep values must be non-empty')
    for sweep_value in sweep_values:
        if float(sweep_value) <= 0:
            raise ValueError(f'Sweep value must be positive, got {sweep_value}')
    if len(set(float(value) for value in sweep_values)) != len(sweep_values):
        raise ValueError(f'Sweep values must be unique, got {sweep_values}')
    if config.randomness_temperature_pairs is not None:
        seen_pairs = set()
        for pair in config.randomness_temperature_pairs:
            if float(pair.randomness) <= 0:
                raise ValueError(f'Pair randomness must be positive, got {pair.randomness}')
            if float(pair.generation_temperature) <= 0:
                raise ValueError(
                    'Pair generation_temperature must be positive, '
                    f'got {pair.generation_temperature}'
                )
            pair_key = (float(pair.randomness), float(pair.generation_temperature))
            if pair_key in seen_pairs:
                raise ValueError(f'Paired sweep values must be unique, got duplicate {pair_key}')
            seen_pairs.add(pair_key)
    if config.experiments is None or not config.experiments:
        raise ValueError('experiments must be non-empty')
    experiment_names = [experiment.name for experiment in config.experiments]
    if len(set(experiment_names)) != len(experiment_names):
        raise ValueError(f'experiment names must be unique, got {experiment_names}')
    for experiment in config.experiments:
        if not experiment.name:
            raise ValueError('experiment name must be non-empty')
        if not os.path.exists(experiment.checkpoint_path):
            raise FileNotFoundError(f'checkpoint not found: {experiment.checkpoint_path}')
        from genmol.rl.reward import normalize_molecular_reward_weights

        normalize_molecular_reward_weights(
            {
                'qed': experiment.qed,
                'sa_score': experiment.sa_score,
            }
        )
    return config


def resolve_device(device_name):
    torch = _get_torch()
    if device_name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('device=cuda requested but CUDA is not available')
        return torch.device('cuda')
    return torch.device(device_name)


def _nanmean(values):
    filtered = [float(value) for value in values if value is not None and not math.isnan(float(value))]
    if not filtered:
        return float('nan')
    return float(sum(filtered) / len(filtered))


def _format_metric(value):
    if value is None:
        return 'nan'
    value = float(value)
    if math.isnan(value):
        return 'nan'
    return f'{value:.6f}'


def _display_name(experiment):
    return experiment.display_name or experiment.name


def _diversity_display_name(metric):
    metric = validate_diversity_metric(metric)
    if metric == MORGAN_INTERNAL_DIVERSITY:
        return 'Internal Diversity'
    if metric == RELATIVE_SCAFFOLD_DIVERSITY:
        return 'Scaffold Diversity'
    raise AssertionError(f'Unhandled diversity metric: {metric}')


def _diversity_description(metric):
    metric = validate_diversity_metric(metric)
    if metric == MORGAN_INTERNAL_DIVERSITY:
        return (
            '`1 - mean(pairwise Morgan-fingerprint Tanimoto similarity)` '
            'over generated valid molecules.'
        )
    if metric == RELATIVE_SCAFFOLD_DIVERSITY:
        return (
            'the number of unique canonical Bemis-Murcko scaffolds divided '
            'by the number of generated valid molecules.'
        )
    raise AssertionError(f'Unhandled diversity metric: {metric}')


def _plot_metric_tradeoff(results, experiments, sweep_values, sweep_axis, x_key, x_label, y_key, y_label, title, output_path):
    plt = _get_pyplot()
    experiment_order = {experiment.name: idx for idx, experiment in enumerate(experiments)}
    sweep_order = {float(value): idx for idx, value in enumerate(sweep_values)}

    fig, ax = plt.subplots(figsize=(8, 6))
    ordered_experiments = sorted(experiments, key=lambda item: experiment_order[item.name])
    for series_index, experiment in enumerate(ordered_experiments):
        rows = [row for row in results if row['experiment'] == experiment.name]
        if len(rows) != len(sweep_values):
            raise ValueError(
                f'Expected {len(sweep_values)} rows for {experiment.name}, found {len(rows)}'
            )
        rows.sort(key=lambda row: sweep_order[float(row['sweep_value'])])
        x_values = []
        y_values = []
        for row in rows:
            x_value = float(row[x_key])
            y_value = float(row[y_key])
            if not math.isfinite(x_value):
                raise ValueError(
                    f'Non-finite {x_key} for experiment={experiment.name} {sweep_axis}={row["sweep_value"]}: {x_value}'
                )
            if not math.isfinite(y_value):
                raise ValueError(
                    f'Non-finite {y_key} for experiment={experiment.name} {sweep_axis}={row["sweep_value"]}: {y_value}'
                )
            x_values.append(x_value)
            y_values.append(y_value)

        style = _series_style(series_index)
        ax.plot(
            x_values,
            y_values,
            color=style['color'],
            marker=style['marker'],
            linewidth=2,
            label=_display_name(experiment),
        )
        for row, x_value, y_value in zip(rows, x_values, y_values):
            sweep_label = row.get('sweep_label') or _format_sweep_label(sweep_axis, row['sweep_value'])
            ax.annotate(
                sweep_label,
                (x_value, y_value),
                textcoords='offset points',
                xytext=(4, 4),
                fontsize=8,
            )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_markdown(config, results):
    sweep_values = _get_sweep_values(config)
    diversity_name = _diversity_display_name(config.diversity_metric)
    sweep_lines = []
    if config.sweep_axis == 'randomness_temperature_pair':
        sweep_lines.extend(
            [
                '- `generation_temperature`: paired',
                '- `randomness`: paired',
                '- `randomness_temperature_pairs`: '
                + ', '.join(
                    f'({float(pair.randomness):.1f}, {float(pair.generation_temperature):.1f})'
                    for pair in config.randomness_temperature_pairs or []
                ),
            ]
        )
    else:
        sweep_lines.extend(
            [
                f'- `generation_temperature`: {config.generation_temperature}',
                f'- `randomness`: {config.randomness}',
                f'- `sweep_values`: {", ".join(str(float(value)) for value in sweep_values)}',
            ]
        )
    lines = [
        '# De Novo Evaluation',
        '',
        f'- `num_samples`: {config.num_samples}',
        f'- `generation_batch_size`: {config.generation_batch_size}',
        f'- `min_add_len`: {config.min_add_len}',
        f'- `max_completion_length`: {config.max_completion_length}',
        f'- `sweep_axis`: {config.sweep_axis}',
        f'- `diversity_metric`: {config.diversity_metric}',
        *sweep_lines,
        '',
        f'- `QED vs Diversity plot`: `{config.output_qed_diversity_plot_path}`',
        f'- `SA Score vs Diversity plot`: `{config.output_sa_score_diversity_plot_path}`',
        f'- `Soft Quality Score vs Diversity plot`: `{config.output_soft_reward_diversity_plot_path}`',
        '',
        f'| Model | Sweep Axis | Sweep Value | Generation Temperature | Randomness | Overall De Novo Score | QED | SA Score | Soft Quality Score | {diversity_name} | Valid Molecule Rate | Alert Hit Rate | Invalid Rate |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in results:
        lines.append(
            '| '
            + ' | '.join(
                [
                    row['display_name'],
                    row['sweep_axis'],
                    _format_metric(row['sweep_value']),
                    _format_metric(row['generation_temperature']),
                    _format_metric(row['randomness']),
                    _format_metric(row['reward_mean']),
                    _format_metric(row['qed_mean']),
                    _format_metric(row['sa_score_mean']),
                    _format_metric(row['soft_reward_mean']),
                    _format_metric(row['diversity']),
                    _format_metric(row['valid_fraction']),
                    _format_metric(row['alert_hit_fraction']),
                    _format_metric(row['invalid_fraction']),
                ]
            )
            + ' |'
        )
    lines.extend(
        [
            '',
            'Column notes:',
            '- `Overall De Novo Score`: mean final molecule-level reward after invalid handling and alert gating.',
            '- `QED`: mean QED over valid generated molecules.',
            '- `SA Score`: mean bounded SA-derived score used by training. Higher is better.',
            '- `Soft Quality Score`: mean weighted rollout-level reward before invalid handling and alert gating. '
            'Weights come from the experiment config and default to `0.6 * QED + 0.4 * SA Score`.',
            f'- `{diversity_name}`: {_diversity_description(config.diversity_metric)}',
            '- `Valid Molecule Rate`: fraction of generated outputs that decode to valid molecules.',
            '- `Alert Hit Rate`: fraction of generated outputs that hit the alert rule set.',
            '- `Invalid Rate`: fraction of generated outputs that are invalid.',
            '',
            'Row notes:',
            '- Each row is one model evaluated at one sweep value.',
            '- The line plots connect rows for the same model in increasing sweep order.',
            '- For paired sweeps, `Sweep Value` is the 1-based pair index while `Generation Temperature` '
            'and `Randomness` record the actual pair values.',
            '',
        ]
    )
    return '\n'.join(lines)


def evaluate_model(config, experiment, device):
    from genmol.diversity import compute_molecular_diversity
    from genmol.rl.policy import GenMolCpGRPOPolicy
    from genmol.rl.reward import MolecularReward
    from genmol.rl.specs import sample_group_specs
    from genmol.rl.trainer import write_jsonl

    logger.info('Evaluating %s', experiment.name)
    policy = GenMolCpGRPOPolicy(
        checkpoint_path=experiment.checkpoint_path,
        device=device,
        bf16=config.bf16,
        trainable=False,
    )
    reward_model = MolecularReward(
        reward_weights={
            'qed': experiment.qed,
            'sa_score': experiment.sa_score,
        },
        always_compute_metrics=True,
    )

    results = []
    try:
        for sweep_point in _build_sweep_points(config):
            logger.info(
                'Evaluating %s at %s=%s (randomness=%s generation_temperature=%s)',
                experiment.name,
                config.sweep_axis,
                sweep_point.sweep_label or sweep_point.sweep_value,
                sweep_point.randomness,
                sweep_point.generation_temperature,
            )
            group_specs = sample_group_specs(
                num_groups=config.num_samples,
                generation_temperature=sweep_point.generation_temperature,
                randomness=sweep_point.randomness,
                min_add_len=config.min_add_len,
                seed=config.seed,
                max_completion_length=config.max_completion_length,
                length_path=config.length_path,
            )
            rollout = policy.rollout_specs(
                specs=group_specs,
                generation_batch_size=min(config.generation_batch_size, len(group_specs)),
                seed=config.seed,
            )
            records = reward_model.score(rollout.smiles)
            if len(records) != config.num_samples:
                raise RuntimeError(
                    f'Reward record count mismatch for {experiment.name}: expected {config.num_samples}, got {len(records)}'
                )

            smiles = [record.smiles for record in records]
            rewards = [float(record.reward) for record in records]
            qeds = [record.qed for record in records]
            sas = [record.sa for record in records]
            sa_scores = [record.sa_score for record in records]
            soft_rewards = [record.soft_reward for record in records]
            valid_flags = [1.0 if record.is_valid else 0.0 for record in records]
            alert_flags = [1.0 if record.alert_hit else 0.0 for record in records]
            invalid_flags = [1.0 if not record.is_valid else 0.0 for record in records]

            results.append(
                {
                    'experiment': experiment.name,
                    'display_name': _display_name(experiment),
                    'checkpoint_path': experiment.checkpoint_path,
                    'qed_weight': reward_model.reward_weights['qed'],
                    'sa_score_weight': reward_model.reward_weights['sa_score'],
                    'num_samples': len(records),
                    'sweep_axis': config.sweep_axis,
                    'sweep_value': sweep_point.sweep_value,
                    'sweep_label': sweep_point.sweep_label,
                    'generation_temperature': sweep_point.generation_temperature,
                    'randomness': sweep_point.randomness,
                    'diversity_metric': config.diversity_metric,
                    'reward_mean': float(sum(rewards) / len(rewards)),
                    'qed_mean': _nanmean(qeds),
                    'sa_mean': _nanmean(sas),
                    'sa_score_mean': _nanmean(sa_scores),
                    'soft_reward_mean': _nanmean(soft_rewards),
                    'diversity': float(
                        compute_molecular_diversity(
                            smiles,
                            metric=config.diversity_metric,
                        )
                    ),
                    'valid_fraction': float(sum(valid_flags) / len(valid_flags)),
                    'alert_hit_fraction': float(sum(alert_flags) / len(alert_flags)),
                    'invalid_fraction': float(sum(invalid_flags) / len(invalid_flags)),
                }
            )

            if config.output_rows_path is not None:
                for idx, record in enumerate(records):
                    write_jsonl(
                        config.output_rows_path,
                        {
                            'experiment': experiment.name,
                            'display_name': _display_name(experiment),
                            'checkpoint_path': experiment.checkpoint_path,
                            'qed_weight': reward_model.reward_weights['qed'],
                            'sa_score_weight': reward_model.reward_weights['sa_score'],
                            'sample_index': idx,
                            'sweep_axis': config.sweep_axis,
                            'sweep_value': sweep_point.sweep_value,
                            'sweep_label': sweep_point.sweep_label,
                            'generation_temperature': sweep_point.generation_temperature,
                            'randomness': sweep_point.randomness,
                            'diversity_metric': config.diversity_metric,
                            'reward': float(record.reward),
                            'is_valid': bool(record.is_valid),
                            'alert_hit': bool(record.alert_hit),
                            'qed': record.qed,
                            'sa': record.sa,
                            'sa_score': record.sa_score,
                            'soft_reward': record.soft_reward,
                            'smiles': record.smiles,
                        },
                    )
    finally:
        reward_model.close()
        del policy
        torch = _get_torch()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    configure_logging()
    logger.info('Starting de novo eval with config=%s', args.config)
    config = load_config(args.config)
    device = resolve_device(config.device)

    if config.output_rows_path is not None:
        _ensure_parent_dir(config.output_rows_path)
        if os.path.exists(config.output_rows_path):
            os.remove(config.output_rows_path)

    results = []
    for experiment in config.experiments:
        results.extend(evaluate_model(config, experiment, device))

    experiment_order = {experiment.name: idx for idx, experiment in enumerate(config.experiments)}
    sweep_values = [point.sweep_value for point in _build_sweep_points(config)]
    sweep_order = {float(value): idx for idx, value in enumerate(sweep_values)}
    diversity_name = _diversity_display_name(config.diversity_metric)
    results.sort(key=lambda row: (experiment_order[row['experiment']], sweep_order[float(row['sweep_value'])]))

    _plot_metric_tradeoff(
        results=results,
        experiments=config.experiments,
        sweep_values=sweep_values,
        sweep_axis=config.sweep_axis,
        x_key='qed_mean',
        x_label='QED',
        y_key='diversity',
        y_label=diversity_name,
        title=f'QED vs {diversity_name}',
        output_path=config.output_qed_diversity_plot_path,
    )
    _plot_metric_tradeoff(
        results=results,
        experiments=config.experiments,
        sweep_values=sweep_values,
        sweep_axis=config.sweep_axis,
        x_key='sa_score_mean',
        x_label='SA Score',
        y_key='diversity',
        y_label=diversity_name,
        title=f'SA Score vs {diversity_name}',
        output_path=config.output_sa_score_diversity_plot_path,
    )
    _plot_metric_tradeoff(
        results=results,
        experiments=config.experiments,
        sweep_values=sweep_values,
        sweep_axis=config.sweep_axis,
        x_key='soft_reward_mean',
        x_label='Soft Quality Score',
        y_key='diversity',
        y_label=diversity_name,
        title=f'Soft Quality Score vs {diversity_name}',
        output_path=config.output_soft_reward_diversity_plot_path,
    )

    markdown = _build_markdown(config, results)
    _ensure_parent_dir(config.output_markdown_path)
    with open(config.output_markdown_path, 'w') as handle:
        handle.write(markdown)

    _ensure_parent_dir(config.output_json_path)
    with open(config.output_json_path, 'w') as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    logger.info('Wrote markdown results to %s', config.output_markdown_path)
    logger.info('Wrote JSON results to %s', config.output_json_path)
    logger.info('Wrote QED-vs-diversity plot to %s', config.output_qed_diversity_plot_path)
    logger.info('Wrote SA-score-vs-diversity plot to %s', config.output_sa_score_diversity_plot_path)
    logger.info(
        'Wrote soft-quality-score-vs-diversity plot to %s',
        config.output_soft_reward_diversity_plot_path,
    )


if __name__ == '__main__':
    main()
