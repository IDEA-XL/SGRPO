import json
import logging
import math
import os
import random
import re
import shutil
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime

import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed

from progen2.checkpoint import (
    PROGEN2_SGRPO_VARIANT,
    require_checkpoint_variant,
    stamp_checkpoint_variant,
)
from progen2.data.prompts import load_prompt_texts
from progen2.modeling.wrapper import OfficialProGen2CausalLM
from progen2.rewards import (
    CompositeProteinReward,
    compute_group_diversity_loo_credits,
    compute_group_diversity_reward_or_zero,
)
from progen2.rewards.common import is_valid_protein_sequence
from progen2.rewards.composite import (
    CALIBRATION_REWARD_NAMES,
    REWARD_NAME_ORDER,
    build_protein_reward_calibration,
    normalize_protein_reward_weights,
    normalize_reward_compute_every_n_steps,
)
from progen2.rl.policy import ProGen2Policy, ProGen2RolloutBatch
from rl_shared.sampling import normalize_scalar_or_range, sample_scalar_or_range
from rl_shared.sgrpo import (
    VALID_GROUP_REWRAD_CREDITS,
    VALID_SGRPO_HIERARCHIES,
    compute_clipped_grpo_loss,
    compute_grouped_advantages,
    compute_sgrpo_advantages,
    validate_reward_threshold_names,
)
from rl_shared.hbd import build_sequence_hbd_memory, validate_hbd_config
from rl_shared.diverse_minibatch import (
    optimization_group_size,
    select_sequence_groups,
    validate_diverse_minibatch_config,
)


logger = logging.getLogger(__name__)
_CHECKPOINT_PATTERN = re.compile(r'^checkpoint-(\d+)$')


def default_reward_compute_every_n_steps():
    return normalize_reward_compute_every_n_steps(None)


@dataclass
class ProGen2TrainConfig:
    model_variant: str
    official_code_dir: str
    tokenizer_path: str
    init_checkpoint_dir: str
    ref_checkpoint_dir: str | None = None
    checkpoint_subdir: str | None = None
    output_dir: str | None = None
    overwrite_output_dir: bool = False
    seed: int = 42
    prompt_path: str = ''
    per_device_prompt_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    bf16: bool = False
    max_steps: int = 50
    learning_rate: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    beta: float = 0.01
    epsilon: float = 0.2
    logging_steps: int = 1
    save_steps: int = 25
    save_total_limit: int = 3
    report_to: list[str] = field(default_factory=list)
    rl_algorithm: str = 'sgrpo'
    num_generations: int = 4
    supergroup_num_groups: int = 2
    group_advantage_weight: float = 0.5
    hierarchy: str = 'advantage_sum'
    naturalness: float | None = None
    foldability: float | None = None
    stability: float | None = None
    developability: float | None = None
    individual_reward_thresholds: dict[str, float | None] = field(default_factory=dict)
    group_rewrad_credit: str = 'broadcast'
    group_rewrad_credit_temperature: float = 1.0
    reward_compute_every_n_steps: dict[str, int] = field(
        default_factory=default_reward_compute_every_n_steps
    )
    max_new_tokens: int = 128
    top_p: float = 0.95
    temperature: float | list[float] = 0.8
    reward_calibration_size: int = 4096
    reward_calibration_prompt_batch_size: int = 8
    rewards: dict = field(default_factory=dict)
    hbd: bool = False
    hbd_bucket_size: int = 25
    hbd_score_threshold_for_memory: float = 0.6
    hbd_similarity_cutoff: float = 0.6
    diverse_minibatch: bool = False
    diverse_minibatch_oversample_factor: int = 2
    entropy_regularization_weight: float = 0.0


def load_config(path):
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    int_fields = [
        'seed',
        'per_device_prompt_batch_size',
        'gradient_accumulation_steps',
        'max_steps',
        'logging_steps',
        'save_steps',
        'save_total_limit',
        'num_generations',
        'supergroup_num_groups',
        'max_new_tokens',
        'reward_calibration_size',
        'reward_calibration_prompt_batch_size',
        'diverse_minibatch_oversample_factor',
    ]
    float_fields = [
        'learning_rate',
        'adam_beta1',
        'adam_beta2',
        'adam_eps',
        'weight_decay',
        'max_grad_norm',
        'beta',
        'epsilon',
        'group_advantage_weight',
        'top_p',
        'entropy_regularization_weight',
    ]
    for field_name in int_fields:
        if field_name in raw and raw[field_name] is not None:
            raw[field_name] = int(raw[field_name])
    for field_name in float_fields:
        if field_name in raw and raw[field_name] is not None:
            raw[field_name] = float(raw[field_name])
    if 'group_rewrad_credit_temperature' in raw and raw['group_rewrad_credit_temperature'] is not None:
        raw['group_rewrad_credit_temperature'] = float(raw['group_rewrad_credit_temperature'])
    raw['reward_compute_every_n_steps'] = normalize_reward_compute_every_n_steps(
        raw.get('reward_compute_every_n_steps')
    )
    config = ProGen2TrainConfig(**raw)
    if config.model_variant != PROGEN2_SGRPO_VARIANT:
        raise ValueError(
            f'Expected model_variant={PROGEN2_SGRPO_VARIANT!r}, got {config.model_variant!r}'
        )
    if config.ref_checkpoint_dir is None:
        config.ref_checkpoint_dir = config.init_checkpoint_dir
    if not config.prompt_path:
        raise ValueError('prompt_path is required')
    if config.per_device_prompt_batch_size <= 0:
        raise ValueError('per_device_prompt_batch_size must be positive')
    if config.gradient_accumulation_steps <= 0:
        raise ValueError('gradient_accumulation_steps must be positive')
    if config.max_steps <= 0:
        raise ValueError('max_steps must be positive')
    if config.num_generations <= 1:
        raise ValueError('num_generations must be greater than 1')
    config.diverse_minibatch_oversample_factor = validate_diverse_minibatch_config(
        enabled=config.diverse_minibatch,
        oversample_factor=config.diverse_minibatch_oversample_factor,
    )
    config.entropy_regularization_weight = float(
        config.entropy_regularization_weight
    )
    if config.entropy_regularization_weight < 0.0:
        raise ValueError('entropy_regularization_weight must be non-negative')
    if config.rl_algorithm not in {'grpo', 'sgrpo'}:
        raise ValueError("rl_algorithm must be 'grpo' or 'sgrpo'")
    if config.supergroup_num_groups <= 0:
        raise ValueError('supergroup_num_groups must be positive')
    if config.rl_algorithm == 'sgrpo' and config.supergroup_num_groups <= 1:
        raise ValueError('supergroup_num_groups must be greater than 1 for sgrpo')
    if not 0.0 <= config.group_advantage_weight <= 1.0:
        raise ValueError('group_advantage_weight must be in [0, 1]')
    config.rollout_reward_weights = normalize_protein_reward_weights(
        {
            'naturalness': config.naturalness,
            'foldability': config.foldability,
            'stability': config.stability,
            'developability': config.developability,
        }
    )
    config.temperature = normalize_scalar_or_range(
        config.temperature,
        name='temperature',
        min_exclusive=0.0,
    )
    config.individual_reward_thresholds = validate_reward_threshold_names(
        config.individual_reward_thresholds,
        REWARD_NAME_ORDER,
    )
    _validate_active_protein_reward_thresholds(
        config.individual_reward_thresholds,
        config.rollout_reward_weights,
    )
    if config.hierarchy not in VALID_SGRPO_HIERARCHIES:
        raise ValueError(
            f"hierarchy must be one of {sorted(VALID_SGRPO_HIERARCHIES)}, got {config.hierarchy!r}"
        )
    if config.group_rewrad_credit not in VALID_GROUP_REWRAD_CREDITS:
        raise ValueError(
            f"group_rewrad_credit must be one of {sorted(VALID_GROUP_REWRAD_CREDITS)}, "
            f'got {config.group_rewrad_credit!r}'
        )
    if config.group_rewrad_credit_temperature <= 0.0:
        raise ValueError('group_rewrad_credit_temperature must be positive')
    has_active_threshold = any(
        threshold is not None for threshold in config.individual_reward_thresholds.values()
    )
    if config.rl_algorithm != 'sgrpo':
        if has_active_threshold:
            raise ValueError('individual_reward_thresholds is only supported when rl_algorithm=sgrpo')
        if config.hierarchy != 'advantage_sum':
            raise ValueError('hierarchy is only supported when rl_algorithm=sgrpo')
        if config.group_rewrad_credit != 'broadcast':
            raise ValueError('group_rewrad_credit is only supported when rl_algorithm=sgrpo')
        if config.group_rewrad_credit_temperature != 1.0:
            raise ValueError('group_rewrad_credit_temperature is only supported when rl_algorithm=sgrpo')
    for reward_name, threshold in config.individual_reward_thresholds.items():
        if threshold is not None and config.reward_compute_every_n_steps[reward_name] != 1:
            raise ValueError(
                'thresholded progen2 rewards must be computed every step; '
                f'{reward_name!r} has every_n_steps={config.reward_compute_every_n_steps[reward_name]}'
            )
    if not config.rewards:
        raise ValueError('rewards config is required')
    if config.reward_calibration_size <= 0:
        raise ValueError('reward_calibration_size must be positive')
    if config.reward_calibration_prompt_batch_size <= 0:
        raise ValueError('reward_calibration_prompt_batch_size must be positive')
    validate_hbd_config(
        enabled=config.hbd,
        bucket_size=config.hbd_bucket_size,
        score_threshold_for_memory=config.hbd_score_threshold_for_memory,
        similarity_cutoff=config.hbd_similarity_cutoff,
    )
    if config.hbd and config.rl_algorithm != 'grpo':
        raise ValueError('hbd is only supported when rl_algorithm=grpo')
    if config.hbd and config.gradient_accumulation_steps > 1:
        raise ValueError(
            'ProGen2 HBD does not support gradient_accumulation_steps > 1 '
            'because HBD memory must be updated once from the full optimizer-step batch'
        )
    if config.diverse_minibatch:
        if config.rl_algorithm != 'grpo':
            raise ValueError(
                'diverse_minibatch is only supported when rl_algorithm=grpo'
            )
        if config.hbd:
            raise ValueError('diverse_minibatch and hbd cannot be enabled together')
        if config.entropy_regularization_weight != 0.0:
            raise ValueError(
                'diverse_minibatch and entropy_regularization_weight cannot be enabled together'
            )
    if config.entropy_regularization_weight > 0.0:
        if config.rl_algorithm != 'grpo':
            raise ValueError(
                'entropy regularization is only supported when rl_algorithm=grpo'
            )
        if config.hbd:
            raise ValueError(
                'entropy regularization and hbd cannot be enabled together'
            )
    return config


def find_last_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    matches = []
    for name in os.listdir(output_dir):
        match = _CHECKPOINT_PATTERN.match(name)
        if match is not None and os.path.isdir(os.path.join(output_dir, name)):
            matches.append((int(match.group(1)), os.path.join(output_dir, name)))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def _require_path(path, description):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing {description}: {path}')


def resolve_output_dir(config, config_path, resume_from_checkpoint=None):
    if resume_from_checkpoint is not None:
        checkpoint_dir = os.path.abspath(resume_from_checkpoint)
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(f'resume checkpoint directory not found: {checkpoint_dir}')
        checkpoint_name = os.path.basename(checkpoint_dir)
        if _CHECKPOINT_PATTERN.match(checkpoint_name) is None:
            raise ValueError(
                'resume_from_checkpoint must point to a checkpoint-XXXXXX directory, '
                f'got {checkpoint_dir}'
            )
        checkpoint_output_dir = os.path.dirname(checkpoint_dir)
        if config.output_dir is not None:
            configured_output_dir = os.path.abspath(config.output_dir)
            if configured_output_dir != checkpoint_output_dir:
                raise ValueError(
                    'resume checkpoint parent directory must match configured output_dir: '
                    f'{checkpoint_output_dir} vs {configured_output_dir}'
                )
        return checkpoint_output_dir
    if config.output_dir is not None:
        return config.output_dir
    cluster_root = '/public/home/xinwuye/ai4s-tool-joint-train'
    if os.path.isdir(cluster_root):
        base_dir = os.path.join(cluster_root, 'runs', 'progen2_sgrpo')
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        base_dir = os.path.join(repo_root, 'runs', 'progen2_sgrpo')
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id:
        return os.path.join(base_dir, f'{config_name}_slurm{slurm_job_id}')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(base_dir, f'{config_name}_{timestamp}')


def _validate_active_protein_reward_thresholds(thresholds, reward_weights):
    for reward_name in REWARD_NAME_ORDER:
        threshold = thresholds.get(reward_name)
        if threshold is not None and reward_weights[reward_name] <= 0.0:
            raise ValueError(
                f'individual_reward_thresholds[{reward_name!r}] requires a positive rollout reward weight, '
                f'got {reward_weights[reward_name]}'
            )


def _cycle_prompt_batch(prompts, batch_size, start_index):
    batch = []
    for offset in range(batch_size):
        batch.append(prompts[(start_index + offset) % len(prompts)])
    return batch


def _distributed_calibration_batch_sizes(total_remaining, per_device_batch_size, world_size):
    if total_remaining < 0:
        raise ValueError(f'total_remaining must be non-negative, got {total_remaining}')
    if per_device_batch_size <= 0:
        raise ValueError(f'per_device_batch_size must be positive, got {per_device_batch_size}')
    if world_size <= 0:
        raise ValueError(f'world_size must be positive, got {world_size}')
    batch_sizes = []
    remaining = int(total_remaining)
    for _ in range(world_size):
        local_batch_size = min(per_device_batch_size, remaining)
        batch_sizes.append(local_batch_size)
        remaining -= local_batch_size
    return batch_sizes


def _shard_list_for_rank(items, world_size, rank):
    if world_size <= 0:
        raise ValueError(f'world_size must be positive, got {world_size}')
    if not 0 <= rank < world_size:
        raise ValueError(f'rank must be in [0, {world_size}), got {rank}')
    total = len(items)
    base = total // world_size
    remainder = total % world_size
    start = rank * base + min(rank, remainder)
    stop = start + base + (1 if rank < remainder else 0)
    return items[start:stop]


def _write_jsonl(path, payload):
    with open(path, 'a') as handle:
        handle.write(json.dumps(payload, sort_keys=True) + '\n')


def default_reward_batch_size(config, *, num_generations=None):
    group_size = (
        config.num_generations
        if num_generations is None
        else int(num_generations)
    )
    return int(
        config.per_device_prompt_batch_size
        * group_size
        * config.supergroup_num_groups
    )


def _aggregate_micro_batch_metrics(metrics_by_micro_batch):
    if not metrics_by_micro_batch:
        raise ValueError('metrics_by_micro_batch must be non-empty')
    all_keys = set()
    for metrics in metrics_by_micro_batch:
        if not isinstance(metrics, dict):
            raise TypeError('each micro-batch metrics payload must be a dict')
        all_keys.update(metrics)

    aggregated = {}
    for key in sorted(all_keys):
        values = []
        for metrics in metrics_by_micro_batch:
            if key not in metrics:
                continue
            try:
                values.append(float(metrics[key]))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f'micro-batch metric {key!r} must be scalar, got {metrics[key]!r}'
                ) from exc
        if not values:
            raise RuntimeError(f'micro-batch metric {key!r} has no values')
        if key.endswith(('_count', '_sec', '_sec_total')):
            aggregated[key] = sum(values)
        elif key.startswith('cuda_') and ('_max_' in key or '_run_max_' in key):
            aggregated[key] = max(values)
        else:
            aggregated[key] = sum(values) / len(values)
    return aggregated


def _merge_rollout_batches(batches, pad_token_id):
    if not batches:
        raise ValueError('cannot merge an empty rollout batch list')
    if len(batches) == 1:
        return batches[0]

    max_len = max(batch.full_token_ids.size(1) for batch in batches)
    padded_ids = []
    padded_attention = []
    padded_generated_mask = []
    for batch in batches:
        pad_len = max_len - batch.full_token_ids.size(1)
        if pad_len < 0:
            raise ValueError('internal error: negative rollout pad length')
        if pad_len == 0:
            padded_ids.append(batch.full_token_ids)
            padded_attention.append(batch.full_attention_mask)
            padded_generated_mask.append(batch.generated_mask)
            continue
        id_pad = torch.full(
            (batch.full_token_ids.size(0), pad_len),
            fill_value=pad_token_id,
            device=batch.full_token_ids.device,
            dtype=batch.full_token_ids.dtype,
        )
        zero_pad = torch.zeros(
            (batch.full_token_ids.size(0), pad_len),
            device=batch.full_token_ids.device,
            dtype=batch.full_attention_mask.dtype,
        )
        mask_pad = torch.zeros(
            (batch.full_token_ids.size(0), pad_len),
            device=batch.full_token_ids.device,
            dtype=batch.generated_mask.dtype,
        )
        padded_ids.append(torch.cat([batch.full_token_ids, id_pad], dim=1))
        padded_attention.append(torch.cat([batch.full_attention_mask, zero_pad], dim=1))
        padded_generated_mask.append(torch.cat([batch.generated_mask, mask_pad], dim=1))

    prompt_texts = []
    decoded_texts = []
    protein_sequences = []
    for batch in batches:
        prompt_texts.extend(batch.prompt_texts)
        decoded_texts.extend(batch.decoded_texts)
        protein_sequences.extend(batch.protein_sequences)

    return ProGen2RolloutBatch(
        prompt_texts=prompt_texts,
        prompt_lengths=torch.cat([batch.prompt_lengths for batch in batches], dim=0),
        full_token_ids=torch.cat(padded_ids, dim=0),
        full_attention_mask=torch.cat(padded_attention, dim=0),
        generated_mask=torch.cat(padded_generated_mask, dim=0),
        decoded_texts=decoded_texts,
        protein_sequences=protein_sequences,
    )


class ProGen2SGRPOTrainer:
    _CALIBRATION_FILE_NAME = 'reward_calibration.json'
    _CALIBRATION_POLL_SEC = 2.0
    _CALIBRATION_TIMEOUT_SEC = 3 * 60 * 60

    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.autocast_dtype = torch.bfloat16 if config.bf16 else None
        self.accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with=config.report_to if config.report_to else None,
            mixed_precision='bf16' if config.bf16 else 'no',
        )
        self.device = self.accelerator.device
        set_seed(config.seed, device_specific=True)

        self.prompts = load_prompt_texts(config.prompt_path)
        # Accumulation slots act as virtual ranks so 4x2 and 8x1 shard identically.
        self.effective_world_size = (
            self.accelerator.num_processes * config.gradient_accumulation_steps
        )
        self._prompt_cursor = self._prompt_cursor_after_steps(0)
        self.optimization_group_size = optimization_group_size(
            config.num_generations,
            enabled=config.diverse_minibatch,
            oversample_factor=config.diverse_minibatch_oversample_factor,
        )
        self.num_return_sequences = (
            self.optimization_group_size * config.supergroup_num_groups
            if config.rl_algorithm == 'sgrpo'
            else self.optimization_group_size
        )
        self.candidate_num_return_sequences = self.num_return_sequences
        if config.diverse_minibatch:
            self.candidate_num_return_sequences = int(config.num_generations)

        self.policy = ProGen2Policy(
            OfficialProGen2CausalLM(
                official_code_dir=config.official_code_dir,
                checkpoint_dir=config.init_checkpoint_dir,
                tokenizer_path=config.tokenizer_path,
                checkpoint_subdir=config.checkpoint_subdir,
                device=self.device,
                use_fp16=False,
                autocast_dtype=self.autocast_dtype,
            ),
            trainable=True,
        )
        self.reference = ProGen2Policy(
            OfficialProGen2CausalLM(
                official_code_dir=config.official_code_dir,
                checkpoint_dir=config.ref_checkpoint_dir,
                tokenizer_path=config.tokenizer_path,
                checkpoint_subdir=config.checkpoint_subdir,
                device=self.device,
                use_fp16=False,
                autocast_dtype=self.autocast_dtype,
            ),
            trainable=False,
        )
        optimizer = torch.optim.AdamW(
            self.policy.model.trainable_parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        self.policy.model.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.policy.model.model,
            optimizer,
            scheduler,
        )
        self.reward_model = CompositeProteinReward(
            config.rewards,
            device=self.device,
            default_reward_batch_size=default_reward_batch_size(
                config,
                num_generations=self.optimization_group_size,
            ),
            reward_compute_every_n_steps=config.reward_compute_every_n_steps,
            reward_weights=config.rollout_reward_weights,
        )
        self._hbd_memory = None
        if self.config.hbd:
            hbd_config = validate_hbd_config(
                enabled=self.config.hbd,
                bucket_size=self.config.hbd_bucket_size,
                score_threshold_for_memory=self.config.hbd_score_threshold_for_memory,
                similarity_cutoff=self.config.hbd_similarity_cutoff,
            )
            self._hbd_memory = build_sequence_hbd_memory(hbd_config)
        self.metrics_path = os.path.join(output_dir, 'metrics.jsonl')
        self.state_path = os.path.join(output_dir, 'trainer_state.json')
        self.calibration_path = os.path.join(output_dir, self._CALIBRATION_FILE_NAME)
        self.global_step = 0
        self._cuda_run_max_reserved = 0
        self._cuda_run_max_allocated = 0
        self._cuda_phase_run_max_reserved = {
            'rollout': 0,
            'reward': 0,
            'training': 0,
        }
        self._cuda_phase_run_max_allocated = {
            'rollout': 0,
            'reward': 0,
            'training': 0,
        }

        if config.report_to:
            tracker_project = os.environ.get('WANDB_PROJECT', 'progen2')
            init_kwargs = {}
            if 'wandb' in config.report_to:
                init_kwargs['wandb'] = {
                    'name': os.path.basename(output_dir),
                }
            self.accelerator.init_trackers(
                tracker_project,
                config=asdict(config),
                init_kwargs=init_kwargs or None,
            )

    def _generate_rollouts(self, prompts, *, num_return_sequences, seed):
        rng = random.Random(seed)
        temperatures = [
            sample_scalar_or_range(
                self.config.temperature,
                rng,
                name='temperature',
                min_exclusive=0.0,
            )
            for _ in prompts
        ]
        if all(temperature == temperatures[0] for temperature in temperatures):
            return self.policy.generate_rollouts(
                prompts,
                num_return_sequences=num_return_sequences,
                max_new_tokens=self.config.max_new_tokens,
                top_p=self.config.top_p,
                temperature=temperatures[0],
                seed=seed,
            )

        batches = []
        for prompt_idx, (prompt, temperature) in enumerate(zip(prompts, temperatures)):
            batches.append(
                self.policy.generate_rollouts(
                    [prompt],
                    num_return_sequences=num_return_sequences,
                    max_new_tokens=self.config.max_new_tokens,
                    top_p=self.config.top_p,
                    temperature=temperature,
                    seed=seed + prompt_idx,
                )
            )
        return _merge_rollout_batches(
            batches,
            pad_token_id=self.policy.model.tokenizer.pad_token_id,
        )

    def _select_diverse_rollout(self, rollout, *, seed):
        if not self.config.diverse_minibatch:
            active_mask = torch.ones(
                len(rollout.protein_sequences),
                device=self.device,
                dtype=torch.bool,
            )
            return rollout, active_mask, {}

        selection = select_sequence_groups(
            rollout.protein_sequences,
            candidate_size=self.candidate_num_return_sequences,
            selected_size=self.optimization_group_size,
            seed=seed,
        )
        index_tensor = torch.tensor(
            selection.indices,
            device=self.device,
            dtype=torch.long,
        )
        active_mask = torch.tensor(
            selection.active_mask,
            device=self.device,
            dtype=torch.bool,
        )
        expected_size = (
            self.config.per_device_prompt_batch_size
            * self.optimization_group_size
        )
        if index_tensor.numel() != expected_size:
            raise RuntimeError(
                'Diverse Mini-Batch selection returned the wrong fixed batch size: '
                f'{index_tensor.numel()} vs {expected_size}'
            )
        selected_rollout = replace(
            rollout,
            prompt_texts=[rollout.prompt_texts[idx] for idx in selection.indices],
            prompt_lengths=rollout.prompt_lengths.index_select(0, index_tensor),
            full_token_ids=rollout.full_token_ids.index_select(0, index_tensor),
            full_attention_mask=rollout.full_attention_mask.index_select(
                0,
                index_tensor,
            ),
            generated_mask=(
                rollout.generated_mask.index_select(0, index_tensor)
                & active_mask.unsqueeze(1)
            ),
            decoded_texts=[
                rollout.decoded_texts[idx] for idx in selection.indices
            ],
            protein_sequences=[
                rollout.protein_sequences[idx] for idx in selection.indices
            ],
        )
        metrics = {
            f'diverse_minibatch/{key}': float(value)
            for key, value in selection.metrics.items()
        }
        metrics['diverse_minibatch/enabled'] = 1.0
        return selected_rollout, active_mask, metrics

    def _all_gather_object(self, payload):
        if self.accelerator.num_processes == 1:
            return [payload]
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError('torch.distributed must be initialized for multi-process calibration')
        gathered = [None] * self.accelerator.num_processes
        dist.all_gather_object(gathered, payload)
        return gathered

    def _broadcast_object(self, payload):
        if self.accelerator.num_processes == 1:
            return payload
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError('torch.distributed must be initialized for multi-process calibration')
        container = [payload]
        dist.broadcast_object_list(container, src=0)
        return container[0]

    def _validate_accumulation_index(self, accumulation_index):
        index = int(accumulation_index)
        if not 0 <= index < self.config.gradient_accumulation_steps:
            raise ValueError(
                'accumulation_index must be in '
                f'[0, {self.config.gradient_accumulation_steps}), got {accumulation_index}'
            )
        return index

    def _virtual_process_index(self, accumulation_index):
        index = self._validate_accumulation_index(accumulation_index)
        return (
            self.accelerator.process_index * self.config.gradient_accumulation_steps
            + index
        )

    def _micro_batch_seed(self, accumulation_index):
        self._validate_accumulation_index(accumulation_index)
        return self.config.seed + self.global_step

    def _selection_seed(self, accumulation_index):
        virtual_process_index = self._virtual_process_index(accumulation_index)
        return (
            self.config.seed
            + self.global_step * 10000
            + virtual_process_index * 1000
            + 991
        )

    def _calibration_seed(self, attempt, accumulation_index):
        attempt = int(attempt)
        if attempt < 0:
            raise ValueError(f'calibration attempt must be non-negative, got {attempt}')
        return (
            self.config.seed
            + attempt * self.effective_world_size
            + self._virtual_process_index(accumulation_index)
        )

    def _calibration_prompt_cursor(self, accumulation_index):
        return (
            self._virtual_process_index(accumulation_index)
            * self.config.reward_calibration_prompt_batch_size
        ) % len(self.prompts)

    def _next_prompt_batch(self, accumulation_index=0):
        index = self._validate_accumulation_index(accumulation_index)
        expected_cursor = self._prompt_cursor_after_steps(self.global_step)
        if self._prompt_cursor != expected_cursor:
            raise RuntimeError(
                'prompt cursor drifted from optimizer-step state: '
                f'{self._prompt_cursor} vs {expected_cursor}'
            )
        micro_batch_cursor = (
            self._prompt_cursor
            + index * self.config.per_device_prompt_batch_size
        ) % len(self.prompts)
        return _cycle_prompt_batch(
            self.prompts,
            self.config.per_device_prompt_batch_size,
            micro_batch_cursor,
        )

    def _prompt_cursor_after_steps(self, completed_steps):
        completed_steps = int(completed_steps)
        if completed_steps < 0:
            raise ValueError(f'completed_steps must be non-negative, got {completed_steps}')
        virtual_process_index = (
            self.accelerator.process_index * self.config.gradient_accumulation_steps
        )
        return (
            (virtual_process_index + completed_steps)
            * self.config.per_device_prompt_batch_size
        ) % len(self.prompts)

    def _calibration_sequences(self):
        target = int(self.config.reward_calibration_size)
        per_device_prompt_batch_size = int(self.config.reward_calibration_prompt_batch_size)
        calibration_cursors = [
            self._calibration_prompt_cursor(accumulation_index)
            for accumulation_index in range(
                self.config.gradient_accumulation_steps
            )
        ]
        collected = [] if self.accelerator.is_main_process else None
        attempts = 0
        max_batches = max(
            8,
            math.ceil(
                target
                / max(
                    1,
                    per_device_prompt_batch_size * self.effective_world_size,
                )
            )
            * 20,
        )
        logger.info(
            'Starting reward calibration: target_sequences=%d '
            'per_device_prompt_batch_size=%d physical_world_size=%d '
            'effective_world_size=%d',
            target,
            per_device_prompt_batch_size,
            self.accelerator.num_processes,
            self.effective_world_size,
        )
        while True:
            batch_sizes = None
            if self.accelerator.is_main_process:
                remaining = target - len(collected)
                if remaining > 0 and attempts >= max_batches:
                    raise RuntimeError(
                        'Failed to collect enough valid calibration sequences before hitting the '
                        f'maximum attempt budget: collected={len(collected)} '
                        f'target={target} max_batches={max_batches}'
                    )
                batch_sizes = _distributed_calibration_batch_sizes(
                    remaining,
                    per_device_prompt_batch_size,
                    self.effective_world_size,
                )
            batch_sizes = self._broadcast_object(batch_sizes)
            local_batch_sizes = [
                int(batch_sizes[self._virtual_process_index(accumulation_index)])
                for accumulation_index in range(
                    self.config.gradient_accumulation_steps
                )
            ]
            total_batch_size = sum(batch_sizes)
            if total_batch_size == 0:
                break
            valid = []
            for accumulation_index, local_batch_size in enumerate(
                local_batch_sizes
            ):
                if local_batch_size > 0:
                    prompts = _cycle_prompt_batch(
                        self.prompts,
                        local_batch_size,
                        calibration_cursors[accumulation_index],
                    )
                    rollout = self._generate_rollouts(
                        prompts,
                        num_return_sequences=1,
                        seed=self._calibration_seed(
                            attempts,
                            accumulation_index,
                        ),
                    )
                    valid.extend(
                        sequence
                        for sequence in rollout.protein_sequences
                        if sequence
                    )
                calibration_cursors[accumulation_index] = (
                    calibration_cursors[accumulation_index]
                    + total_batch_size
                ) % len(self.prompts)
            gathered = self._all_gather_object(valid)
            if self.accelerator.is_main_process:
                for sequences in gathered:
                    collected.extend(sequences)
                collected = collected[:target]
                remaining = target - len(collected)
                logger.info(
                    'Reward calibration progress: collected=%d remaining=%d',
                    len(collected),
                    max(remaining, 0),
                )
            attempts += 1
        return self._broadcast_object(collected if self.accelerator.is_main_process else None)

    def _distributed_reward_calibration(self, sequences):
        local_sequences = _shard_list_for_rank(
            sequences,
            self.accelerator.num_processes,
            self.accelerator.process_index,
        )
        logger.info(
            'Scoring reward calibration shard on rank=%d: local_sequences=%d global_sequences=%d',
            self.accelerator.process_index,
            len(local_sequences),
            len(sequences),
        )
        local_raw_scores = (
            self.reward_model.collect_calibration_raw(local_sequences)
            if local_sequences
            else {}
        )
        gathered_raw_scores = self._all_gather_object(local_raw_scores)
        calibration = None
        if self.accelerator.is_main_process:
            merged_raw_scores = {reward_name: [] for reward_name in CALIBRATION_REWARD_NAMES}
            for payload in gathered_raw_scores:
                for reward_name in CALIBRATION_REWARD_NAMES:
                    merged_raw_scores[reward_name].extend(payload.get(reward_name, []))
            calibration = build_protein_reward_calibration(merged_raw_scores)
            logger.info(
                'Built reward calibration statistics for keys=%s',
                sorted(calibration.keys()),
            )
        calibration = self._broadcast_object(calibration)
        self.reward_model.set_calibration(calibration)
        return calibration

    def calibrate(self):
        calibration_sequences = self._calibration_sequences()
        calibration = self._distributed_reward_calibration(calibration_sequences)
        if self.accelerator.is_main_process:
            self._write_calibration_json(self.calibration_path, calibration)
            logger.info('Reward calibration complete')
        self.accelerator.wait_for_everyone()

    def _write_calibration_json(self, path, calibration):
        calibration_tmp_path = f'{path}.tmp'
        with open(calibration_tmp_path, 'w') as handle:
            json.dump(calibration, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(calibration_tmp_path, path)

    def _score_rollout_rewards(
        self,
        sequences,
        *,
        step_number,
        active_mask=None,
    ):
        if active_mask is None:
            active_flags = [True] * len(sequences)
        else:
            if not torch.is_tensor(active_mask):
                raise ValueError('active_mask must be a tensor when provided')
            if active_mask.dim() != 1 or active_mask.numel() != len(sequences):
                raise ValueError('active_mask must match the sequence batch')
            active_flags = active_mask.to(dtype=torch.bool).tolist()
        active_count = sum(active_flags)
        if active_count == 0:
            raise RuntimeError('reward scoring requires at least one active sequence')
        valid_indices = [
            idx for idx, sequence in enumerate(sequences)
            if active_flags[idx] and is_valid_protein_sequence(sequence)
        ]
        rewards = [0.0] * len(sequences)
        individual_reward_values = {
            reward_name: [0.0] * len(sequences) for reward_name in REWARD_NAME_ORDER
        }
        metrics = {
            'invalid_sequence_rate': 1.0 - (
                len(valid_indices) / active_count
            )
        }
        if valid_indices:
            valid_sequences = [sequences[idx] for idx in valid_indices]
            reward_details, reward_metrics = self.reward_model.score_details(
                valid_sequences,
                step_number=step_number,
            )
            valid_rewards = reward_details['total']
            for target_idx, reward in zip(valid_indices, valid_rewards):
                rewards[target_idx] = float(reward)
            for reward_name in REWARD_NAME_ORDER:
                for target_idx, reward_value in zip(valid_indices, reward_details[reward_name]):
                    individual_reward_values[reward_name][target_idx] = float(reward_value)
            metrics.update(reward_metrics)
        individual_reward_tensors = {
            reward_name: torch.tensor(values, device=self.device, dtype=torch.float32)
            for reward_name, values in individual_reward_values.items()
        }
        return (
            torch.tensor(rewards, device=self.device, dtype=torch.float32),
            metrics,
            individual_reward_tensors,
        )

    def _score_group_rewards(self, sequences, active_mask=None):
        if active_mask is None:
            active_flags = [True] * len(sequences)
        else:
            if active_mask.dim() != 1 or active_mask.numel() != len(sequences):
                raise ValueError('active_mask must match the sequence batch')
            active_flags = active_mask.to(dtype=torch.bool).tolist()
        group_rewards = []
        for start in range(0, len(sequences), self.optimization_group_size):
            end = start + self.optimization_group_size
            group_sequences = [
                sequence
                for sequence, active in zip(
                    sequences[start:end],
                    active_flags[start:end],
                )
                if active
            ]
            group_rewards.append(float(compute_group_diversity_reward_or_zero(group_sequences)))
        return torch.tensor(group_rewards, device=self.device, dtype=torch.float32)

    def _score_group_reward_credits(self, sequences):
        group_credits = []
        for start in range(0, len(sequences), self.optimization_group_size):
            group_sequences = sequences[start:start + self.optimization_group_size]
            group_credits.extend(compute_group_diversity_loo_credits(group_sequences))
        return torch.tensor(group_credits, device=self.device, dtype=torch.float32)

    def _apply_hbd(self, sequences, rollout_rewards):
        if self._hbd_memory is None:
            raise RuntimeError('_apply_hbd called while HBD is disabled')
        if rollout_rewards.dim() != 1:
            raise ValueError(f'rollout_rewards must be 1D, got shape={tuple(rollout_rewards.shape)}')
        local_count = len(sequences)
        if rollout_rewards.numel() != local_count:
            raise ValueError(
                'sequence/reward length mismatch for HBD: '
                f'{local_count} vs {rollout_rewards.numel()}'
            )

        gathered_sequences = self._all_gather_object(sequences)
        global_sequences = []
        for shard in gathered_sequences:
            global_sequences.extend(shard)
        global_rewards = self.accelerator.gather(rollout_rewards.detach()).to(device=self.device, dtype=torch.float32)
        if global_rewards.numel() != len(global_sequences):
            raise ValueError(
                'global sequence/reward length mismatch for HBD: '
                f'{len(global_sequences)} vs {global_rewards.numel()}'
            )

        hbd_update_payload = None
        if self.accelerator.is_main_process:
            planned_update = self._hbd_memory.plan_update(
                global_sequences,
                memory_scores=global_rewards.tolist(),
                reward_values=global_rewards.tolist(),
            )
            hbd_update_payload = planned_update.to_payload()
        hbd_update_payload = self._broadcast_object(hbd_update_payload)
        self._hbd_memory.apply_update(hbd_update_payload)
        adjusted_rewards = torch.tensor(
            hbd_update_payload['final_rewards'],
            device=self.device,
            dtype=torch.float32,
        )
        local_start = self.accelerator.process_index * local_count
        local_end = local_start + local_count
        return adjusted_rewards[local_start:local_end], hbd_update_payload['metrics']

    def _build_group_mean_individual_rewards(self, individual_reward_tensors):
        outputs = {}
        for reward_name, reward_values in individual_reward_tensors.items():
            if reward_values.dim() != 1:
                raise ValueError(f'individual reward tensor for {reward_name!r} must be 1D')
            if reward_values.numel() % self.optimization_group_size != 0:
                raise ValueError(
                    'individual reward tensor length must be divisible by num_generations: '
                    f'{reward_name!r} has {reward_values.numel()} vs {self.optimization_group_size}'
                )
            outputs[reward_name] = reward_values.view(
                -1,
                self.optimization_group_size,
            ).mean(dim=1)
        return outputs

    def _compute_advantages(
        self,
        rollout_rewards,
        group_rewards,
        group_mean_individual_rewards=None,
        group_reward_credits=None,
        sample_mask=None,
    ):
        advantages = []
        expanded_group_advantages = []
        rollout_advantages = []
        metrics = {}
        chunk_rollout = self.num_return_sequences
        chunk_groups = (
            self.config.supergroup_num_groups
            if self.config.rl_algorithm == 'sgrpo'
            else 1
        )
        for prompt_idx in range(self.config.per_device_prompt_batch_size):
            start = prompt_idx * chunk_rollout
            end = start + chunk_rollout
            prompt_rollout_rewards = rollout_rewards[start:end]
            prompt_sample_mask = None
            if sample_mask is not None:
                if sample_mask.dim() != 1 or sample_mask.shape != rollout_rewards.shape:
                    raise ValueError(
                        'sample_mask must be 1D and match rollout_rewards'
                    )
                prompt_sample_mask = sample_mask[start:end]
            group_start = prompt_idx * chunk_groups
            group_end = group_start + chunk_groups
            prompt_group_rewards = group_rewards[group_start:group_end]
            prompt_group_mean_individual_rewards = None
            if group_mean_individual_rewards is not None:
                prompt_group_mean_individual_rewards = {
                    reward_name: reward_means[group_start:group_end]
                    for reward_name, reward_means in group_mean_individual_rewards.items()
                }
            if self.config.rl_algorithm == 'sgrpo':
                prompt_group_reward_credits = None
                if group_reward_credits is not None:
                    prompt_group_reward_credits = group_reward_credits[start:end]
                prompt_adv, prompt_group_adv, prompt_rollout_adv, prompt_metrics = compute_sgrpo_advantages(
                    rollout_rewards=prompt_rollout_rewards,
                    group_rewards=prompt_group_rewards,
                    num_generations=self.optimization_group_size,
                    supergroup_num_groups=self.config.supergroup_num_groups,
                    group_advantage_weight=self.config.group_advantage_weight,
                    scale_rewards=False,
                    hierarchy=self.config.hierarchy,
                    group_mean_individual_rewards=prompt_group_mean_individual_rewards,
                    individual_reward_thresholds=self.config.individual_reward_thresholds,
                    group_rewrad_credit=self.config.group_rewrad_credit,
                    group_rewrad_credit_temperature=self.config.group_rewrad_credit_temperature,
                    group_reward_credits=prompt_group_reward_credits,
                )
            else:
                prompt_adv, _, zero_std_ratio = compute_grouped_advantages(
                    rewards=prompt_rollout_rewards,
                    num_generations=self.optimization_group_size,
                    scale_rewards=False,
                    sample_mask=prompt_sample_mask,
                )
                prompt_rollout_adv = prompt_adv
                prompt_group_adv = torch.zeros_like(prompt_rollout_rewards)
                prompt_metrics = {
                    'rollout_zero_std_ratio': zero_std_ratio,
                    'group_zero_std_ratio': float('nan'),
                    'group_reward_mean': prompt_group_rewards.mean().item() if prompt_group_rewards.numel() else float('nan'),
                }
            advantages.append(prompt_adv)
            expanded_group_advantages.append(prompt_group_adv)
            rollout_advantages.append(prompt_rollout_adv)
            for key, value in prompt_metrics.items():
                metrics.setdefault(key, []).append(float(value))
        merged_metrics = {key: sum(values) / len(values) for key, values in metrics.items()}
        return (
            torch.cat(advantages, dim=0),
            torch.cat(expanded_group_advantages, dim=0),
            torch.cat(rollout_advantages, dim=0),
            merged_metrics,
        )

    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.output_dir, f'checkpoint-{self.global_step:06d}')
        accelerator_state_dir = os.path.join(checkpoint_dir, 'accelerator_state')
        calibration_checkpoint_path = os.path.join(checkpoint_dir, self._CALIBRATION_FILE_NAME)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(accelerator_state_dir)
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return
        _require_path(self.calibration_path, 'reward calibration')
        unwrapped = self.accelerator.unwrap_model(self.policy.model.model)
        unwrapped.save_pretrained(
            checkpoint_dir,
            state_dict=self.accelerator.get_state_dict(self.policy.model.model),
        )
        trainer_state = {
            'global_step': int(self.global_step),
            'config': asdict(self.config),
            'hbd_state': None if self._hbd_memory is None else self._hbd_memory.state_dict(),
        }
        stamp_checkpoint_variant(trainer_state, PROGEN2_SGRPO_VARIANT)
        torch.save(trainer_state, os.path.join(checkpoint_dir, 'trainer_state.pt'))
        shutil.copyfile(self.calibration_path, calibration_checkpoint_path)

    def _load_checkpoint(self, checkpoint_dir):
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        accelerator_state_dir = os.path.join(checkpoint_dir, 'accelerator_state')
        trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.pt')
        calibration_checkpoint_path = os.path.join(checkpoint_dir, self._CALIBRATION_FILE_NAME)
        _require_path(accelerator_state_dir, 'accelerator state')
        _require_path(trainer_state_path, 'trainer state')
        _require_path(calibration_checkpoint_path, 'reward calibration snapshot')

        trainer_state = torch.load(trainer_state_path, map_location='cpu', weights_only=False)
        require_checkpoint_variant(trainer_state, PROGEN2_SGRPO_VARIANT, trainer_state_path)

        self.accelerator.load_state(accelerator_state_dir)
        with open(calibration_checkpoint_path) as handle:
            calibration = json.load(handle)
        self.reward_model.set_calibration(calibration)
        self.global_step = int(trainer_state['global_step'])
        self._prompt_cursor = self._prompt_cursor_after_steps(self.global_step)
        hbd_state = trainer_state.get('hbd_state')
        if self._hbd_memory is None:
            if hbd_state is not None:
                raise ValueError('checkpoint contains HBD state but current config has hbd disabled')
        else:
            if hbd_state is None:
                raise ValueError('checkpoint is missing HBD state while current config has hbd enabled')
            self._hbd_memory.load_state_dict(hbd_state)
        if self.accelerator.is_main_process:
            self._write_calibration_json(self.calibration_path, calibration)
        self.accelerator.wait_for_everyone()

    def _log(self, metrics):
        payload = {'step': int(self.global_step), **metrics}
        if self.accelerator.is_main_process:
            _write_jsonl(self.metrics_path, payload)
            with open(self.state_path, 'w') as handle:
                json.dump(
                    {
                        'global_step': self.global_step,
                        'prompt_cursor': self._prompt_cursor,
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
        if self.config.report_to:
            self.accelerator.log(metrics, step=self.global_step)

    def _reset_cuda_phase_peak(self):
        if self.device.type != 'cuda':
            return
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

    def _capture_cuda_phase_peak(self, phase_name):
        if self.device.type != 'cuda':
            raise RuntimeError('CUDA phase peak capture requires a CUDA device')
        if phase_name not in self._cuda_phase_run_max_reserved:
            raise ValueError(f'Unsupported CUDA phase name: {phase_name}')
        torch.cuda.synchronize(self.device)
        total_memory = float(torch.cuda.get_device_properties(self.device).total_memory)
        step_peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
        step_peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
        self._cuda_phase_run_max_reserved[phase_name] = max(
            self._cuda_phase_run_max_reserved[phase_name],
            step_peak_reserved,
        )
        self._cuda_phase_run_max_allocated[phase_name] = max(
            self._cuda_phase_run_max_allocated[phase_name],
            step_peak_allocated,
        )
        metrics = {
            f'cuda_{phase_name}_step_max_reserved_gib': float(step_peak_reserved / (1024 ** 3)),
            f'cuda_{phase_name}_step_max_reserved_ratio': float(step_peak_reserved / total_memory),
            f'cuda_{phase_name}_step_max_allocated_gib': float(step_peak_allocated / (1024 ** 3)),
            f'cuda_{phase_name}_step_max_allocated_ratio': float(step_peak_allocated / total_memory),
            f'cuda_{phase_name}_run_max_reserved_gib': float(
                self._cuda_phase_run_max_reserved[phase_name] / (1024 ** 3)
            ),
            f'cuda_{phase_name}_run_max_reserved_ratio': float(
                self._cuda_phase_run_max_reserved[phase_name] / total_memory
            ),
            f'cuda_{phase_name}_run_max_allocated_gib': float(
                self._cuda_phase_run_max_allocated[phase_name] / (1024 ** 3)
            ),
            f'cuda_{phase_name}_run_max_allocated_ratio': float(
                self._cuda_phase_run_max_allocated[phase_name] / total_memory
            ),
        }
        return metrics, step_peak_reserved, step_peak_allocated

    def _run_training_micro_batch(self, accumulation_index):
        accumulation_index = self._validate_accumulation_index(accumulation_index)
        rollout_phase_metrics = {}
        reward_phase_metrics = {}
        rollout_step_peak_reserved = 0
        rollout_step_peak_allocated = 0
        reward_step_peak_reserved = 0
        reward_step_peak_allocated = 0

        logger.info(
            'Starting train step %d/%d micro-batch %d/%d',
            self.global_step + 1,
            self.config.max_steps,
            accumulation_index + 1,
            self.config.gradient_accumulation_steps,
        )
        if self.device.type == 'cuda':
            self._reset_cuda_phase_peak()
        prompts = self._next_prompt_batch(accumulation_index)
        rollout = self._generate_rollouts(
            prompts,
            num_return_sequences=self.candidate_num_return_sequences,
            seed=self._micro_batch_seed(accumulation_index),
        )
        rollout, active_mask, selection_metrics = self._select_diverse_rollout(
            rollout,
            seed=self._selection_seed(accumulation_index),
        )
        if self.device.type == 'cuda':
            (
                rollout_phase_metrics,
                rollout_step_peak_reserved,
                rollout_step_peak_allocated,
            ) = self._capture_cuda_phase_peak('rollout')

        if self.device.type == 'cuda':
            self._reset_cuda_phase_peak()
        rollout_rewards, reward_metrics, individual_reward_tensors = (
            self._score_rollout_rewards(
                rollout.protein_sequences,
                step_number=self.global_step + 1,
                active_mask=active_mask,
            )
        )
        hbd_metrics = {}
        if self._hbd_memory is not None:
            rollout_rewards, hbd_metrics = self._apply_hbd(
                rollout.protein_sequences,
                rollout_rewards,
            )
        group_rewards = self._score_group_rewards(
            rollout.protein_sequences,
            active_mask=active_mask,
        )
        group_reward_credits = None
        if self.config.rl_algorithm == 'sgrpo' and self.config.group_rewrad_credit == 'loo':
            group_reward_credits = self._score_group_reward_credits(
                rollout.protein_sequences
            )
        group_mean_individual_rewards = None
        if any(
            threshold is not None
            for threshold in self.config.individual_reward_thresholds.values()
        ):
            group_mean_individual_rewards = self._build_group_mean_individual_rewards(
                individual_reward_tensors
            )
        (
            advantages,
            group_advantages,
            rollout_advantages,
            advantage_metrics,
        ) = self._compute_advantages(
            rollout_rewards,
            group_rewards,
            group_mean_individual_rewards=group_mean_individual_rewards,
            group_reward_credits=group_reward_credits,
            sample_mask=active_mask if self.config.diverse_minibatch else None,
        )
        if self.device.type == 'cuda':
            (
                reward_phase_metrics,
                reward_step_peak_reserved,
                reward_step_peak_allocated,
            ) = self._capture_cuda_phase_peak('reward')

        if self.device.type == 'cuda':
            self._reset_cuda_phase_peak()
        with torch.no_grad():
            old_log_probs, completion_mask = self.policy.per_token_logps(
                rollout.full_token_ids,
                rollout.full_attention_mask,
                rollout.generated_mask,
                requires_grad=False,
            )
            ref_log_probs, _ = self.reference.per_token_logps(
                rollout.full_token_ids,
                rollout.full_attention_mask,
                rollout.generated_mask,
                requires_grad=False,
            )

        policy_outputs = self.policy.per_token_logps(
            rollout.full_token_ids,
            rollout.full_attention_mask,
            rollout.generated_mask,
            requires_grad=True,
            return_normalized_entropy=(
                self.config.entropy_regularization_weight > 0.0
            ),
        )
        if self.config.entropy_regularization_weight > 0.0:
            (
                new_log_probs,
                completion_mask,
                per_token_normalized_entropy,
            ) = policy_outputs
        else:
            new_log_probs, completion_mask = policy_outputs
            per_token_normalized_entropy = None
        loss, loss_metrics = compute_clipped_grpo_loss(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            completion_mask=completion_mask,
            epsilon=self.config.epsilon,
            ref_log_probs=ref_log_probs,
            beta=self.config.beta,
        )
        if self.config.entropy_regularization_weight > 0.0:
            if per_token_normalized_entropy is None:
                raise RuntimeError(
                    'policy did not return entropy while entropy regularization is enabled'
                )
            entropy_mask = completion_mask.to(dtype=torch.float32)
            normalized_entropy_mean = (
                per_token_normalized_entropy[:, 0, :] * entropy_mask
            ).sum() / entropy_mask.sum().clamp(min=1.0)
            entropy_regularization_loss = -normalized_entropy_mean
            loss = loss + (
                self.config.entropy_regularization_weight
                * entropy_regularization_loss
            )
            loss_metrics['entropy/normalized_mean'] = (
                normalized_entropy_mean.detach()
            )
            loss_metrics['entropy/regularization_loss'] = (
                entropy_regularization_loss.detach()
            )

        active_rollout_rewards = rollout_rewards[active_mask]
        active_advantages = advantages[active_mask]
        active_group_advantages = group_advantages[active_mask]
        active_rollout_advantages = rollout_advantages[active_mask]
        metrics = {
            'loss': float(loss.detach().item()),
            'reward_mean': float(active_rollout_rewards.mean().item()),
            'group_reward_mean': float(
                advantage_metrics.get('group_reward_mean', group_rewards.mean().item())
            ),
            'group_reward_raw_mean': float(
                advantage_metrics.get(
                    'group_reward_raw_mean',
                    group_rewards.mean().item(),
                )
            ),
            'group_reward_indicator_mean': float(
                advantage_metrics.get('group_reward_indicator_mean', 1.0)
            ),
            'advantage_mean': float(active_advantages.mean().item()),
            'group_advantage_mean': float(active_group_advantages.mean().item()),
            'rollout_advantage_mean': float(active_rollout_advantages.mean().item()),
            **{key: float(value) for key, value in reward_metrics.items()},
            **selection_metrics,
            **{f'hbd/{key}': float(value) for key, value in hbd_metrics.items()},
            **{key: float(value) for key, value in advantage_metrics.items()},
            **{
                key: float(value.item() if hasattr(value, 'item') else value)
                for key, value in loss_metrics.items()
            },
            **rollout_phase_metrics,
            **reward_phase_metrics,
        }
        return {
            'loss': loss,
            'metrics': metrics,
            'step_peak_reserved': [
                rollout_step_peak_reserved,
                reward_step_peak_reserved,
            ],
            'step_peak_allocated': [
                rollout_step_peak_allocated,
                reward_step_peak_allocated,
            ],
        }

    def _finish_cuda_training_phase(self, micro_batch_result):
        if self.device.type != 'cuda':
            return
        (
            training_phase_metrics,
            training_step_peak_reserved,
            training_step_peak_allocated,
        ) = self._capture_cuda_phase_peak('training')
        micro_batch_result['metrics'].update(training_phase_metrics)
        step_peak_reserved = max(
            *micro_batch_result['step_peak_reserved'],
            training_step_peak_reserved,
        )
        step_peak_allocated = max(
            *micro_batch_result['step_peak_allocated'],
            training_step_peak_allocated,
        )
        self._cuda_run_max_reserved = max(
            self._cuda_run_max_reserved,
            step_peak_reserved,
        )
        self._cuda_run_max_allocated = max(
            self._cuda_run_max_allocated,
            step_peak_allocated,
        )
        total_memory = float(torch.cuda.get_device_properties(self.device).total_memory)
        micro_batch_result['metrics'].update(
            {
                'cuda_step_max_reserved_gib': float(
                    step_peak_reserved / (1024 ** 3)
                ),
                'cuda_step_max_reserved_ratio': float(
                    step_peak_reserved / total_memory
                ),
                'cuda_step_max_allocated_gib': float(
                    step_peak_allocated / (1024 ** 3)
                ),
                'cuda_step_max_allocated_ratio': float(
                    step_peak_allocated / total_memory
                ),
                'cuda_run_max_reserved_gib': float(
                    self._cuda_run_max_reserved / (1024 ** 3)
                ),
                'cuda_run_max_reserved_ratio': float(
                    self._cuda_run_max_reserved / total_memory
                ),
                'cuda_run_max_allocated_gib': float(
                    self._cuda_run_max_allocated / (1024 ** 3)
                ),
                'cuda_run_max_allocated_ratio': float(
                    self._cuda_run_max_allocated / total_memory
                ),
            }
        )

    def train(self, resume_from_checkpoint=None):
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info('Starting ProGen2 SGRPO training in %s', self.output_dir)
        if resume_from_checkpoint is not None:
            logger.info('Resuming from checkpoint: %s', resume_from_checkpoint)
            self._load_checkpoint(resume_from_checkpoint)
        else:
            self.calibrate()

        while self.global_step < self.config.max_steps:
            logger.info(
                'Starting optimizer step %d/%d with %d micro-batches',
                self.global_step + 1,
                self.config.max_steps,
                self.config.gradient_accumulation_steps,
            )
            self.optimizer.zero_grad(set_to_none=True)
            micro_batch_results = []
            for accumulation_index in range(
                self.config.gradient_accumulation_steps
            ):
                is_last_micro_batch = (
                    accumulation_index
                    == self.config.gradient_accumulation_steps - 1
                )
                sync_context = (
                    nullcontext()
                    if is_last_micro_batch
                    else self.accelerator.no_sync(self.policy.model.model)
                )
                with sync_context:
                    micro_batch_result = self._run_training_micro_batch(
                        accumulation_index
                    )
                    loss = micro_batch_result.pop('loss')
                    # Accelerate scales each backward call by accumulation steps.
                    self.accelerator.backward(loss)
                    del loss
                if (
                    self.device.type == 'cuda'
                    and not is_last_micro_batch
                ):
                    self._finish_cuda_training_phase(micro_batch_result)
                micro_batch_results.append(micro_batch_result)

            grad_norm = self.accelerator.clip_grad_norm_(
                self.policy.model.trainable_parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()
            self.scheduler.step()
            if self.device.type == 'cuda':
                self._finish_cuda_training_phase(micro_batch_results[-1])

            self.global_step += 1
            self._prompt_cursor = self._prompt_cursor_after_steps(self.global_step)
            metrics = _aggregate_micro_batch_metrics(
                [result['metrics'] for result in micro_batch_results]
            )
            metrics.update(
                {
                    'grad_norm': float(
                        grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
                    ),
                    'gradient_accumulation_steps': float(
                        self.config.gradient_accumulation_steps
                    ),
                    'effective_world_size': float(self.effective_world_size),
                    'effective_global_prompt_batch_size': float(
                        self.effective_world_size
                        * self.config.per_device_prompt_batch_size
                    ),
                }
            )
            if self.global_step % self.config.logging_steps == 0:
                self._log(metrics)
            if self.global_step % self.config.save_steps == 0:
                self.save_checkpoint()

        if self.config.report_to:
            self.accelerator.end_training()
