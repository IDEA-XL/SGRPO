import json
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime

import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed

from genmol.rl.cpgrpo import (
    VALID_ADVANTAGE_BASELINES,
    VALID_GROUP_REWRAD_CREDITS,
    VALID_SGRPO_HIERARCHIES,
    compute_clipped_grpo_loss,
    compute_grouped_advantages,
    compute_group_reward_regularizer_advantages,
    compute_sgrpo_advantages,
    compute_warmup_steps,
    normalize_reward_thresholds,
    split_tensor_dict,
    validate_reward_threshold_names,
)
from genmol.rl.policy import GenMolCpGRPOPolicy
from genmol.rl.reward import (
    MOLECULAR_REWARD_NAME_ORDER,
    MolecularReward,
    RewardRecord,
    combine_candidate_diversity_reward,
    compute_internal_diversity,
    compute_internal_diversity_with_loo_credits,
    normalize_molecular_reward_weights,
)
from genmol.rl.specs import (
    deserialize_specs,
    expand_group_specs,
    sample_group_specs,
    sample_supergroup_shared_specs,
    serialize_specs,
)
from rl_shared.hbd import build_molecule_hbd_memory, validate_hbd_config
from rl_shared.diverse_minibatch import (
    optimization_group_size,
    select_molecule_groups,
    validate_diverse_minibatch_config,
)
from rl_shared.sampling import normalize_scalar_or_range


logger = logging.getLogger(__name__)
GENMOL_SGRPO_THRESHOLD_REWARD_NAMES = ('qed', 'sa_score')


@dataclass
class TrainConfig:
    init_ckpt_path: str
    ref_ckpt_path: str | None = None
    output_dir: str | None = None
    overwrite_output_dir: bool = False
    seed: int = 42
    sync_ref_model: bool = True
    ref_model_sync_steps: int = 64
    ref_model_mixup_alpha: float = 0.6
    beta: float = 0.01
    epsilon: float = 0.5
    scale_rewards: bool = False
    advantage_baseline: str = 'leave_one_out'
    bf16: bool = True
    do_eval: bool = False
    num_generations: int = 8
    num_iterations: int = 1
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {'use_reentrant': False})
    learning_rate: float = 1e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    adam_eps: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 0.2
    logging_first_step: bool = True
    logging_steps: int = 10
    logging_strategy: str = 'steps'
    max_steps: int = 100
    lr_scheduler_type: str = 'cosine_with_min_lr'
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {'min_lr_rate': 0.1})
    warmup_ratio: float = 0.0001
    save_strategy: str = 'steps'
    save_steps: int = 50
    save_total_limit: int = 5
    random_masking: bool = True
    generation_batch_size: int = 16
    generation_temperature: float | list[float] = 1.0
    randomness: float | list[float] = 0.3
    min_add_len: int = 60
    max_completion_length: int | None = None
    log_completions: bool = True
    log_level: str = 'info'
    report_to: list[str] = field(default_factory=list)
    rl_algorithm: str = 'coupled_grpo'
    supergroup_num_groups: int = 1
    group_advantage_weight: float = 0.5
    diversity_regularizer_weight: float = 0.0
    hierarchy: str = 'advantage_sum'
    qed: float | None = None
    sa_score: float | None = None
    individual_reward_thresholds: dict[str, float | None] = field(default_factory=dict)
    group_rewrad_credit: str = 'broadcast'
    group_rewrad_credit_temperature: float = 1.0
    hbd: bool = False
    hbd_bucket_size: int = 25
    hbd_score_threshold_for_memory: float = 0.6
    hbd_similarity_cutoff: float = 0.6
    diverse_minibatch: bool = False
    diverse_minibatch_oversample_factor: int = 2
    entropy_regularization_weight: float = 0.0
    candidate_diversity_reward_weight: float = 0.0


@dataclass
class TrainResult:
    metrics: dict


def load_config(path):
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    config = TrainConfig(**raw)
    if config.ref_ckpt_path is None:
        config.ref_ckpt_path = config.init_ckpt_path
    if config.logging_strategy != 'steps':
        raise ValueError('Only logging_strategy=steps is supported')
    if config.save_strategy != 'steps':
        raise ValueError('Only save_strategy=steps is supported')
    if config.num_generations <= 1:
        raise ValueError('num_generations must be greater than 1')
    if config.num_iterations <= 0:
        raise ValueError('num_iterations must be positive')
    if config.per_device_train_batch_size <= 0:
        raise ValueError('per_device_train_batch_size must be positive')
    if config.gradient_accumulation_steps <= 0:
        raise ValueError('gradient_accumulation_steps must be positive')
    if config.generation_batch_size <= 0:
        raise ValueError('generation_batch_size must be positive')
    config.generation_temperature = normalize_scalar_or_range(
        config.generation_temperature,
        name='generation_temperature',
        min_exclusive=0.0,
    )
    config.randomness = normalize_scalar_or_range(
        config.randomness,
        name='randomness',
        min_exclusive=0.0,
    )
    if not 0.0 <= config.ref_model_mixup_alpha <= 1.0:
        raise ValueError('ref_model_mixup_alpha must be in [0, 1]')
    if config.advantage_baseline not in VALID_ADVANTAGE_BASELINES:
        raise ValueError(
            f"advantage_baseline must be one of {sorted(VALID_ADVANTAGE_BASELINES)}, "
            f'got {config.advantage_baseline!r}'
        )
    if config.rl_algorithm not in {'coupled_grpo', 'coupled_sgrpo'}:
        raise ValueError(
            f"Unsupported rl_algorithm: {config.rl_algorithm}. Expected 'coupled_grpo' or 'coupled_sgrpo'"
        )
    if config.supergroup_num_groups <= 0:
        raise ValueError('supergroup_num_groups must be positive')
    if not 0.0 <= config.group_advantage_weight <= 1.0:
        raise ValueError('group_advantage_weight must be in [0, 1]')
    if config.diversity_regularizer_weight < 0.0:
        raise ValueError('diversity_regularizer_weight must be non-negative')
    config.diverse_minibatch_oversample_factor = validate_diverse_minibatch_config(
        enabled=config.diverse_minibatch,
        oversample_factor=config.diverse_minibatch_oversample_factor,
    )
    config.entropy_regularization_weight = float(
        config.entropy_regularization_weight
    )
    if config.entropy_regularization_weight < 0.0:
        raise ValueError('entropy_regularization_weight must be non-negative')
    config.candidate_diversity_reward_weight = float(
        config.candidate_diversity_reward_weight
    )
    if not 0.0 <= config.candidate_diversity_reward_weight <= 1.0:
        raise ValueError(
            'candidate_diversity_reward_weight must be in [0, 1]'
        )
    config.rollout_reward_weights = normalize_molecular_reward_weights(
        {
            'qed': config.qed,
            'sa_score': config.sa_score,
        }
    )
    config.individual_reward_thresholds = validate_reward_threshold_names(
        config.individual_reward_thresholds,
        GENMOL_SGRPO_THRESHOLD_REWARD_NAMES,
    )
    _validate_active_molecular_reward_thresholds(
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
    config.group_rewrad_credit_temperature = float(config.group_rewrad_credit_temperature)
    if config.group_rewrad_credit_temperature <= 0.0:
        raise ValueError('group_rewrad_credit_temperature must be positive')
    validate_hbd_config(
        enabled=config.hbd,
        bucket_size=config.hbd_bucket_size,
        score_threshold_for_memory=config.hbd_score_threshold_for_memory,
        similarity_cutoff=config.hbd_similarity_cutoff,
    )
    if config.rl_algorithm != 'coupled_sgrpo':
        has_active_threshold = any(
            threshold is not None for threshold in config.individual_reward_thresholds.values()
        )
        if has_active_threshold:
            raise ValueError(
                'individual_reward_thresholds is only supported when rl_algorithm=coupled_sgrpo'
            )
        if config.hierarchy != 'advantage_sum':
            raise ValueError('hierarchy is only supported when rl_algorithm=coupled_sgrpo')
        if config.group_rewrad_credit != 'broadcast':
            raise ValueError('group_rewrad_credit is only supported when rl_algorithm=coupled_sgrpo')
        if config.group_rewrad_credit_temperature != 1.0:
            raise ValueError('group_rewrad_credit_temperature is only supported when rl_algorithm=coupled_sgrpo')
    if config.hbd and config.rl_algorithm != 'coupled_grpo':
        raise ValueError('hbd is only supported when rl_algorithm=coupled_grpo')
    if config.diverse_minibatch:
        if config.rl_algorithm != 'coupled_grpo':
            raise ValueError(
                'diverse_minibatch is only supported when rl_algorithm=coupled_grpo'
            )
        if config.hbd:
            raise ValueError('diverse_minibatch and hbd cannot be enabled together')
        if config.diversity_regularizer_weight != 0.0:
            raise ValueError(
                'diverse_minibatch and diversity_regularizer_weight cannot be enabled together'
            )
        if config.entropy_regularization_weight != 0.0:
            raise ValueError(
                'diverse_minibatch and entropy_regularization_weight cannot be enabled together'
            )
    if config.entropy_regularization_weight > 0.0:
        if config.rl_algorithm != 'coupled_grpo':
            raise ValueError(
                'entropy regularization is only supported when rl_algorithm=coupled_grpo'
            )
        if config.hbd:
            raise ValueError(
                'entropy regularization and hbd cannot be enabled together'
            )
        if config.diversity_regularizer_weight != 0.0:
            raise ValueError(
                'entropy regularization and diversity_regularizer_weight cannot be enabled together'
            )
    if config.candidate_diversity_reward_weight > 0.0:
        if config.rl_algorithm != 'coupled_grpo':
            raise ValueError(
                'candidate diversity reward is only supported when rl_algorithm=coupled_grpo'
            )
        incompatible_options = []
        if config.hbd:
            incompatible_options.append('hbd')
        if config.diverse_minibatch:
            incompatible_options.append('diverse_minibatch')
        if config.entropy_regularization_weight > 0.0:
            incompatible_options.append('entropy_regularization_weight')
        if config.diversity_regularizer_weight > 0.0:
            incompatible_options.append('diversity_regularizer_weight')
        if incompatible_options:
            raise ValueError(
                'candidate diversity reward cannot be combined with: '
                + ', '.join(incompatible_options)
            )
    return config


def resolve_output_dir(config, config_path):
    if config.output_dir is not None:
        return config.output_dir

    config_realpath = os.path.realpath(config_path)
    config_dir = os.path.dirname(config_realpath)
    if os.path.basename(config_realpath) == 'config.yaml' and find_last_checkpoint(config_dir) is not None:
        return config_dir

    cluster_root = '/public/home/xinwuye/ai4s-tool-joint-train'
    if os.path.isdir(cluster_root):
        base_dir = os.path.join(cluster_root, 'runs', 'cpgrpo_denovo')
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        base_dir = os.path.join(repo_root, 'runs', 'cpgrpo_denovo')

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(base_dir, f'{config_name}_{timestamp}')


def ensure_exists(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f'{label} not found: {path}')


def build_scheduler(optimizer, config):
    if config.lr_scheduler_type != 'cosine_with_min_lr':
        raise ValueError(f'Unsupported lr_scheduler_type: {config.lr_scheduler_type}')

    min_lr_rate = float(config.lr_scheduler_kwargs.get('min_lr_rate', 0.1))
    warmup_steps = compute_warmup_steps(config.max_steps, config.warmup_ratio)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, config.max_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_rate + (1.0 - min_lr_rate) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def broadcast_specs(group_specs, num_generations, accelerator):
    payload = [None]
    if accelerator.is_main_process:
        payload[0] = serialize_specs(expand_group_specs(group_specs, num_generations))
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    return deserialize_specs(payload[0])


def broadcast_object(payload, accelerator):
    wrapped = [payload if accelerator.is_main_process else None]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(wrapped, src=0)
    return wrapped[0]


def write_jsonl(path, payload):
    with open(path, 'a') as handle:
        handle.write(json.dumps(payload, sort_keys=True) + '\n')


def _nanmean(tensor):
    if tensor.numel() == 0 or torch.isnan(tensor).all():
        return float('nan')
    return torch.nanmean(tensor).item()


def _nanreduce(tensor, mode):
    valid = tensor[~torch.isnan(tensor)]
    if valid.numel() == 0:
        return float('nan')
    if mode == 'min':
        return valid.min().item()
    if mode == 'max':
        return valid.max().item()
    raise ValueError(f'Unsupported nan reduction mode: {mode}')


def _aggregate_scalar_list(values, mode='mean'):
    if not values:
        return float('nan')
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.numel() == 0 or torch.isnan(tensor).all():
        return float('nan')
    if mode == 'mean':
        return torch.nanmean(tensor).item()
    if mode == 'min':
        return _nanreduce(tensor, mode='min')
    if mode == 'max':
        return _nanreduce(tensor, mode='max')
    raise ValueError(f'Unsupported scalar aggregation mode: {mode}')


def _has_active_individual_reward_thresholds(thresholds):
    normalized = normalize_reward_thresholds(thresholds)
    return any(threshold is not None for threshold in normalized.values())


def _validate_active_molecular_reward_thresholds(thresholds, reward_weights):
    normalized_thresholds = normalize_reward_thresholds(thresholds)
    for reward_name in MOLECULAR_REWARD_NAME_ORDER:
        threshold = normalized_thresholds.get(reward_name)
        if threshold is not None and reward_weights[reward_name] <= 0.0:
            raise ValueError(
                f'individual_reward_thresholds[{reward_name!r}] requires a positive rollout reward weight, '
                f'got {reward_weights[reward_name]}'
            )


def _build_group_mean_individual_rewards(*, qed_values, sa_score_values, num_generations, device):
    qed_tensor = torch.as_tensor(qed_values, device=device, dtype=torch.float32)
    sa_score_tensor = torch.as_tensor(sa_score_values, device=device, dtype=torch.float32)
    if qed_tensor.dim() != 1 or sa_score_tensor.dim() != 1:
        raise ValueError('individual reward tensors must be 1D')
    if qed_tensor.numel() != sa_score_tensor.numel():
        raise ValueError(
            'qed and sa_score tensors must have matching lengths: '
            f'{qed_tensor.numel()} vs {sa_score_tensor.numel()}'
        )
    if qed_tensor.numel() % num_generations != 0:
        raise ValueError(
            'individual reward tensor length must be divisible by num_generations: '
            f'{qed_tensor.numel()} vs {num_generations}'
        )
    return {
        'qed': qed_tensor.view(-1, num_generations).mean(dim=1),
        'sa_score': sa_score_tensor.view(-1, num_generations).mean(dim=1),
    }


def find_last_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None

    pattern = re.compile(r'^checkpoint-(\d+)$')
    matches = []
    for name in os.listdir(output_dir):
        match = pattern.match(name)
        if match is not None and os.path.isdir(os.path.join(output_dir, name)):
            matches.append((int(match.group(1)), os.path.join(output_dir, name)))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def maybe_trim_checkpoints(output_dir, save_total_limit):
    if save_total_limit is None or save_total_limit <= 0:
        return

    pattern = re.compile(r'^checkpoint-(\d+)$')
    checkpoints = []
    for name in os.listdir(output_dir):
        match = pattern.match(name)
        if match is not None and os.path.isdir(os.path.join(output_dir, name)):
            checkpoints.append((int(match.group(1)), os.path.join(output_dir, name)))
    checkpoints.sort()
    while len(checkpoints) > save_total_limit:
        _, path = checkpoints.pop(0)
        for root, dirs, files in os.walk(path, topdown=False):
            for filename in files:
                os.remove(os.path.join(root, filename))
            for dirname in dirs:
                os.rmdir(os.path.join(root, dirname))
        os.rmdir(path)


class GenMolCpGRPOTrainer:
    def __init__(self, config, output_dir):
        self.config = config
        self.output_dir = output_dir
        self.accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with=config.report_to if config.report_to else None,
            mixed_precision='bf16' if config.bf16 else 'no',
        )
        self.device = self.accelerator.device
        self.world_size = self.accelerator.num_processes
        self.local_sample_count = config.per_device_train_batch_size * config.gradient_accumulation_steps
        self.global_sample_count = self.local_sample_count * self.world_size
        self.optimization_group_size = optimization_group_size(
            config.num_generations,
            enabled=config.diverse_minibatch,
            oversample_factor=config.diverse_minibatch_oversample_factor,
        )
        self.candidate_group_size = int(config.num_generations)
        if self.global_sample_count % self.optimization_group_size != 0:
            raise ValueError(
                'global train batch size must be divisible by the optimization '
                f'group size: {self.global_sample_count} vs {self.optimization_group_size}'
            )
        self.num_groups_global = (
            self.global_sample_count // self.optimization_group_size
        )
        self.local_candidate_count = self.local_sample_count
        self.global_candidate_count = self.global_sample_count
        if config.diverse_minibatch:
            if self.local_sample_count % self.optimization_group_size != 0:
                raise ValueError(
                    'diverse_minibatch requires each rank batch to contain whole '
                    f'groups: {self.local_sample_count} vs {self.optimization_group_size}'
                )
            local_group_count = (
                self.local_sample_count // self.optimization_group_size
            )
            self.local_candidate_count = (
                local_group_count * self.candidate_group_size
            )
            self.global_candidate_count = (
                self.num_groups_global * self.candidate_group_size
            )
        if config.rl_algorithm == 'coupled_sgrpo':
            if config.supergroup_num_groups <= 1:
                raise ValueError('supergroup_num_groups must be greater than 1 for coupled_sgrpo')
            if self.num_groups_global % config.supergroup_num_groups != 0:
                raise ValueError(
                    'num_groups_global must be divisible by supergroup_num_groups for coupled_sgrpo: '
                    f'{self.num_groups_global} vs {config.supergroup_num_groups}'
                )
        if config.diversity_regularizer_weight > 0.0 and self.num_groups_global <= 1:
            raise ValueError(
                'diversity_regularizer_weight > 0 requires at least two groups in the global batch'
            )

        deepspeed_plugin = getattr(self.accelerator.state, 'deepspeed_plugin', None)
        if deepspeed_plugin is not None:
            deepspeed_config = deepspeed_plugin.deepspeed_config
            deepspeed_config['train_micro_batch_size_per_gpu'] = int(config.per_device_train_batch_size)
            deepspeed_config['gradient_accumulation_steps'] = int(config.gradient_accumulation_steps)
            deepspeed_config['train_batch_size'] = int(self.global_sample_count)

        ensure_exists(config.init_ckpt_path, 'init checkpoint')
        ensure_exists(config.ref_ckpt_path, 'reference checkpoint')

        set_seed(config.seed, device_specific=True)

        self.policy = GenMolCpGRPOPolicy(
            checkpoint_path=config.init_ckpt_path,
            device=self.device,
            bf16=config.bf16,
            trainable=True,
        )
        self.reference = GenMolCpGRPOPolicy(
            checkpoint_path=config.ref_ckpt_path,
            device=self.device,
            bf16=config.bf16,
            trainable=False,
        )
        if config.gradient_checkpointing:
            self.policy.enable_gradient_checkpointing(config.gradient_checkpointing_kwargs)
        self.policy.train()

        optimizer = torch.optim.AdamW(
            self.policy.trainable_parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
            weight_decay=config.weight_decay,
        )
        scheduler = build_scheduler(optimizer, config)
        self.policy.model.backbone, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.policy.model.backbone,
            optimizer,
            scheduler,
        )

        self.reward_model = MolecularReward(reward_weights=self.config.rollout_reward_weights)
        self._hbd_memory = None
        if self.config.hbd:
            hbd_config = validate_hbd_config(
                enabled=self.config.hbd,
                bucket_size=self.config.hbd_bucket_size,
                score_threshold_for_memory=self.config.hbd_score_threshold_for_memory,
                similarity_cutoff=self.config.hbd_similarity_cutoff,
            )
            self._hbd_memory = build_molecule_hbd_memory(hbd_config)
        self.metrics_path = os.path.join(output_dir, 'metrics.jsonl')
        self.text_logs_path = os.path.join(output_dir, 'completions.jsonl')
        self.state_path = os.path.join(output_dir, 'trainer_state.json')

        self.global_step = 0
        self.generation_cycle_idx = 0
        self._step = 0
        self._buffered_inputs = None
        self._buffer_metadata = None
        self._last_train_metrics = None
        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
        self._textual_logs = {'train': [], 'eval': []}
        self._last_reward_metrics = {'train': None, 'eval': None}

        if config.report_to:
            init_kwargs = {}
            if 'wandb' in config.report_to:
                init_kwargs['wandb'] = {'name': os.path.basename(output_dir)}
            self.accelerator.init_trackers('genmol-cpgrpo', config=asdict(config), init_kwargs=init_kwargs or None)

        logger.info(
            'process_index=%s device=%s world_size=%s local_sample_count=%s global_sample_count=%s reward_workers=%s',
            self.accelerator.process_index,
            self.device,
            self.world_size,
            self.local_sample_count,
            self.global_sample_count,
            self.reward_model.num_workers,
        )

    def _record_reward_metrics(self, mode, metadata):
        if mode not in self._metrics:
            raise ValueError(f'Unsupported mode: {mode}')
        bucket = self._metrics[mode]
        self._last_reward_metrics[mode] = {
            'reward': float(metadata['reward_mean']),
            'reward_std': float(metadata['reward_std']),
            'advantage_mean': float(metadata['advantage_mean']),
            'completion_length': float(metadata['completion_length']),
            'zero_std_ratio': float(metadata['zero_std_ratio']),
            'valid_fraction': float(metadata['valid_fraction']),
            'alert_hit_fraction': float(metadata['alert_hit_fraction']),
            'invalid_fraction': float(metadata['invalid_fraction']),
            'rewards/qed_mean': float(metadata['rewards/qed_mean']),
            'rewards/sa_mean': float(metadata['rewards/sa_mean']),
            'rewards/sa_score_mean': float(metadata['rewards/sa_score_mean']),
            'rewards/soft_mean': float(metadata['rewards/soft_mean']),
        }
        if 'rollout_advantage_mean' in metadata:
            self._last_reward_metrics[mode]['rollout_advantage_mean'] = float(metadata['rollout_advantage_mean'])
        if 'group_advantage_mean' in metadata:
            self._last_reward_metrics[mode]['group_advantage_mean'] = float(metadata['group_advantage_mean'])
        if 'group_reward/diversity_mean' in metadata:
            self._last_reward_metrics[mode]['group_reward/diversity_mean'] = float(metadata['group_reward/diversity_mean'])
        if 'rollout_zero_std_ratio' in metadata:
            self._last_reward_metrics[mode]['rollout_zero_std_ratio'] = float(metadata['rollout_zero_std_ratio'])
        if 'group_zero_std_ratio' in metadata:
            self._last_reward_metrics[mode]['group_zero_std_ratio'] = float(metadata['group_zero_std_ratio'])
        if 'diversity_regularizer/advantage_mean' in metadata:
            self._last_reward_metrics[mode]['diversity_regularizer/advantage_mean'] = float(
                metadata['diversity_regularizer/advantage_mean']
            )
        if 'diversity_regularizer/group_reward_mean' in metadata:
            self._last_reward_metrics[mode]['diversity_regularizer/group_reward_mean'] = float(
                metadata['diversity_regularizer/group_reward_mean']
            )
        if 'diversity_regularizer/zero_std_ratio' in metadata:
            self._last_reward_metrics[mode]['diversity_regularizer/zero_std_ratio'] = float(
                metadata['diversity_regularizer/zero_std_ratio']
            )
        for name in (
            'hbd/enabled',
            'hbd/eligible_count',
            'hbd/penalized_count',
            'hbd/accepted_existing_count',
            'hbd/created_bucket_count',
            'hbd/bucket_count_before',
            'hbd/bucket_count_after',
        ):
            if name in metadata:
                self._last_reward_metrics[mode][name] = float(metadata[name])
        for name, value in metadata.items():
            if name.startswith('candidate_diversity/'):
                self._last_reward_metrics[mode][name] = float(value)
        bucket['reward'].append(float(metadata['reward_mean']))
        bucket['reward_std'].append(float(metadata['reward_std']))
        bucket['advantage_mean'].append(float(metadata['advantage_mean']))
        bucket['completion_length'].append(float(metadata['completion_length']))
        bucket['zero_std_ratio'].append(float(metadata['zero_std_ratio']))
        bucket['valid_fraction'].append(float(metadata['valid_fraction']))
        bucket['alert_hit_fraction'].append(float(metadata['alert_hit_fraction']))
        bucket['invalid_fraction'].append(float(metadata['invalid_fraction']))
        bucket['rewards/qed_mean'].append(float(metadata['rewards/qed_mean']))
        bucket['rewards/sa_mean'].append(float(metadata['rewards/sa_mean']))
        bucket['rewards/sa_score_mean'].append(float(metadata['rewards/sa_score_mean']))
        bucket['rewards/soft_mean'].append(float(metadata['rewards/soft_mean']))
        if 'rollout_advantage_mean' in metadata:
            bucket['rollout_advantage_mean'].append(float(metadata['rollout_advantage_mean']))
        if 'group_advantage_mean' in metadata:
            bucket['group_advantage_mean'].append(float(metadata['group_advantage_mean']))
        if 'group_reward/diversity_mean' in metadata:
            bucket['group_reward/diversity_mean'].append(float(metadata['group_reward/diversity_mean']))
        if 'rollout_zero_std_ratio' in metadata:
            bucket['rollout_zero_std_ratio'].append(float(metadata['rollout_zero_std_ratio']))
        if 'group_zero_std_ratio' in metadata:
            bucket['group_zero_std_ratio'].append(float(metadata['group_zero_std_ratio']))
        if 'diversity_regularizer/advantage_mean' in metadata:
            bucket['diversity_regularizer/advantage_mean'].append(float(metadata['diversity_regularizer/advantage_mean']))
        if 'diversity_regularizer/group_reward_mean' in metadata:
            bucket['diversity_regularizer/group_reward_mean'].append(float(metadata['diversity_regularizer/group_reward_mean']))
        if 'diversity_regularizer/zero_std_ratio' in metadata:
            bucket['diversity_regularizer/zero_std_ratio'].append(float(metadata['diversity_regularizer/zero_std_ratio']))
        for name in (
            'hbd/enabled',
            'hbd/eligible_count',
            'hbd/penalized_count',
            'hbd/accepted_existing_count',
            'hbd/created_bucket_count',
            'hbd/bucket_count_before',
            'hbd/bucket_count_after',
        ):
            if name in metadata:
                bucket[name].append(float(metadata[name]))
        for name, value in metadata.items():
            if name.startswith(
                ('diverse_minibatch/', 'candidate_diversity/')
            ):
                bucket[name].append(float(value))

    def _record_loss_metrics(self, mode, step_metrics):
        if mode not in self._metrics:
            raise ValueError(f'Unsupported mode: {mode}')
        bucket = self._metrics[mode]

        gathered_ratio = self.accelerator.gather_for_metrics(step_metrics['ratio_mean'].detach().reshape(1))
        bucket['ratio_mean'].append(torch.nanmean(gathered_ratio).item())

        gathered_low = self.accelerator.gather_for_metrics(step_metrics['clip_ratio_low_mean'].detach().reshape(1))
        bucket['clip_ratio/low_mean'].append(torch.nanmean(gathered_low).item())
        bucket['clip_ratio/low_min'].append(_nanreduce(gathered_low, mode='min'))

        gathered_high = self.accelerator.gather_for_metrics(step_metrics['clip_ratio_high_mean'].detach().reshape(1))
        bucket['clip_ratio/high_mean'].append(torch.nanmean(gathered_high).item())
        bucket['clip_ratio/high_max'].append(_nanreduce(gathered_high, mode='max'))

        gathered_region = self.accelerator.gather_for_metrics(step_metrics['clip_ratio_region_mean'].detach().reshape(1))
        bucket['clip_ratio/region_mean'].append(torch.nanmean(gathered_region).item())

        if 'kl_mean' in step_metrics:
            gathered_kl = self.accelerator.gather_for_metrics(step_metrics['kl_mean'].detach().reshape(1))
            bucket['kl'].append(torch.nanmean(gathered_kl).item())
        if 'diversity_regularizer_loss' in step_metrics:
            gathered_diversity_loss = self.accelerator.gather_for_metrics(
                step_metrics['diversity_regularizer_loss'].detach().reshape(1)
            )
            bucket['diversity_regularizer/loss'].append(torch.nanmean(gathered_diversity_loss).item())
        for name in (
            'entropy/normalized_mean',
            'entropy/regularization_loss',
        ):
            if name in step_metrics:
                gathered_entropy_metric = self.accelerator.gather_for_metrics(
                    step_metrics[name].detach().reshape(1)
                )
                bucket[name].append(
                    torch.nanmean(gathered_entropy_metric).item()
                )

    def _record_optimizer_metrics(self, mode, grad_norm, lr):
        if mode not in self._metrics:
            raise ValueError(f'Unsupported mode: {mode}')
        self._metrics[mode]['grad_norm'].append(float(grad_norm))
        self._metrics[mode]['lr'].append(float(lr))

    def _record_text_logs(self, mode, rows):
        if mode not in self._textual_logs:
            raise ValueError(f'Unsupported mode: {mode}')
        if not self.config.log_completions:
            return
        if not rows:
            return
        gathered = self._all_gather_objects(rows)
        if self.accelerator.is_main_process:
            self._textual_logs[mode].extend(gathered)

    def _flush_text_logs(self, mode):
        if not self.config.log_completions:
            self._textual_logs[mode] = []
            return
        if not self.accelerator.is_main_process:
            self._textual_logs[mode] = []
            return
        for row in self._textual_logs[mode]:
            write_jsonl(self.text_logs_path, row)
        self._textual_logs[mode] = []

    def _has_pending_metrics(self, mode):
        return any(self._metrics[mode].values())

    def _sample_mask_seeds(self):
        if self.config.random_masking:
            return torch.randint(0, 2**12, (self.config.num_iterations,), device=self.device).tolist()
        return [42] * self.config.num_iterations

    def _all_gather_objects(self, payload):
        if not dist.is_available() or not dist.is_initialized():
            return payload
        gathered = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, payload)
        merged = []
        for shard in gathered:
            merged.extend(shard)
        return merged

    def _compute_group_diversity_statistics(
        self,
        global_smiles,
        *,
        include_loo_credits,
    ):
        if len(global_smiles) != self.global_sample_count:
            raise ValueError(
                'global smiles count must match global sample count: '
                f'{len(global_smiles)} vs {self.global_sample_count}'
            )

        payload = None
        if self.accelerator.is_main_process:
            group_diversities = []
            group_loo_credits = [] if include_loo_credits else None
            for group_start in range(
                0,
                len(global_smiles),
                self.config.num_generations,
            ):
                group_smiles = global_smiles[
                    group_start:group_start + self.config.num_generations
                ]
                if len(group_smiles) != self.config.num_generations:
                    raise ValueError(
                        'incomplete rollout group while computing diversity: '
                        f'{len(group_smiles)} vs {self.config.num_generations}'
                    )
                if include_loo_credits:
                    group_diversity, loo_credits = (
                        compute_internal_diversity_with_loo_credits(
                            group_smiles
                        )
                    )
                    group_loo_credits.extend(loo_credits)
                else:
                    group_diversity = compute_internal_diversity(
                        group_smiles
                    )
                group_diversities.append(group_diversity)
            payload = {
                'group_diversities': group_diversities,
                'group_loo_credits': group_loo_credits,
            }

        payload = broadcast_object(payload, self.accelerator)
        if payload is None:
            raise RuntimeError(
                'group diversity statistics were not broadcast'
            )
        group_diversities = payload['group_diversities']
        if len(group_diversities) != self.num_groups_global:
            raise RuntimeError(
                'group diversity count mismatch: '
                f'{len(group_diversities)} vs {self.num_groups_global}'
            )
        group_diversity_tensor = torch.tensor(
            group_diversities,
            device=self.device,
            dtype=torch.float32,
        )

        group_loo_credits = payload['group_loo_credits']
        if not include_loo_credits:
            if group_loo_credits is not None:
                raise RuntimeError(
                    'unexpected LOO credits when they were not requested'
                )
            return group_diversity_tensor, None
        if len(group_loo_credits) != self.global_sample_count:
            raise RuntimeError(
                'group LOO credit count mismatch: '
                f'{len(group_loo_credits)} vs {self.global_sample_count}'
            )
        return (
            group_diversity_tensor,
            torch.tensor(
                group_loo_credits,
                device=self.device,
                dtype=torch.float32,
            ),
        )

    def _apply_candidate_diversity_reward(
        self,
        *,
        global_soft_rewards,
        global_valid_mask,
        global_alert_mask,
        global_loo_credits,
        global_group_diversities,
    ):
        expected_shape = (self.global_sample_count,)
        for name, tensor in (
            ('global_soft_rewards', global_soft_rewards),
            ('global_valid_mask', global_valid_mask),
            ('global_alert_mask', global_alert_mask),
            ('global_loo_credits', global_loo_credits),
        ):
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f'{name} must have shape {expected_shape}, '
                    f'got {tuple(tensor.shape)}'
                )
        expected_group_shape = (self.num_groups_global,)
        if tuple(global_group_diversities.shape) != expected_group_shape:
            raise ValueError(
                'global_group_diversities must have shape '
                f'{expected_group_shape}, '
                f'got {tuple(global_group_diversities.shape)}'
            )
        if not torch.isfinite(
            global_soft_rewards[global_valid_mask]
        ).all():
            raise ValueError(
                'valid candidates must have finite soft rewards'
            )
        if not torch.isfinite(global_loo_credits).all():
            raise ValueError('raw LOO diversity credits must be finite')
        if not torch.isfinite(global_group_diversities).all():
            raise ValueError('group diversities must be finite')

        payload = None
        if self.accelerator.is_main_process:
            soft_values = global_soft_rewards.tolist()
            valid_values = global_valid_mask.tolist()
            alert_values = global_alert_mask.tolist()
            loo_values = global_loo_credits.tolist()
            final_rewards = [
                combine_candidate_diversity_reward(
                    None if not is_valid else soft_reward,
                    loo_credit,
                    weight=self.config.candidate_diversity_reward_weight,
                    is_valid=is_valid,
                    alert_hit=alert_hit,
                )
                for soft_reward, loo_credit, is_valid, alert_hit in zip(
                    soft_values,
                    loo_values,
                    valid_values,
                    alert_values,
                )
            ]

            valid_loo_credits = global_loo_credits[
                global_valid_mask
            ].to(dtype=torch.float32)
            final_reward_tensor = torch.tensor(
                final_rewards,
                dtype=torch.float32,
            )

            def _summarize(values):
                if values.numel() == 0:
                    return {
                        'mean': float('nan'),
                        'std': float('nan'),
                        'min': float('nan'),
                        'max': float('nan'),
                    }
                return {
                    'mean': values.mean().item(),
                    'std': values.std(unbiased=False).item(),
                    'min': values.min().item(),
                    'max': values.max().item(),
                }

            raw_loo_summary = _summarize(valid_loo_credits)
            combined_summary = _summarize(final_reward_tensor)
            metrics = {
                'enabled': 1.0,
                'weight': float(
                    self.config.candidate_diversity_reward_weight
                ),
                'valid_count': float(valid_loo_credits.numel()),
                'group_diversity_mean': (
                    global_group_diversities.mean().item()
                ),
            }
            metrics.update(
                {
                    f'raw_loo_{name}': value
                    for name, value in raw_loo_summary.items()
                }
            )
            metrics.update(
                {
                    f'combined_reward_{name}': value
                    for name, value in combined_summary.items()
                }
            )
            payload = {
                'final_rewards': final_rewards,
                'metrics': metrics,
            }

        payload = broadcast_object(payload, self.accelerator)
        if payload is None:
            raise RuntimeError(
                'candidate diversity rewards were not broadcast'
            )
        final_rewards = payload['final_rewards']
        if len(final_rewards) != self.global_sample_count:
            raise RuntimeError(
                'candidate diversity reward count mismatch: '
                f'{len(final_rewards)} vs {self.global_sample_count}'
            )
        return (
            torch.tensor(
                final_rewards,
                device=self.device,
                dtype=torch.float32,
            ),
            payload['metrics'],
        )

    def _apply_hbd(self, *, global_items, global_memory_scores, global_reward_values):
        if self._hbd_memory is None:
            raise RuntimeError('_apply_hbd called while HBD is disabled')
        if global_memory_scores.dim() != 1:
            raise ValueError(
                f'global_memory_scores must be 1D for HBD, got shape={tuple(global_memory_scores.shape)}'
            )
        if global_reward_values.dim() != 1:
            raise ValueError(
                f'global_reward_values must be 1D for HBD, got shape={tuple(global_reward_values.shape)}'
            )
        if len(global_items) != global_memory_scores.numel():
            raise ValueError(
                'global item/memory-score length mismatch for HBD: '
                f'{len(global_items)} vs {global_memory_scores.numel()}'
            )
        if len(global_items) != global_reward_values.numel():
            raise ValueError(
                'global item/reward length mismatch for HBD: '
                f'{len(global_items)} vs {global_reward_values.numel()}'
            )

        hbd_update_payload = None
        if self.accelerator.is_main_process:
            planned_update = self._hbd_memory.plan_update(
                global_items,
                memory_scores=global_memory_scores.tolist(),
                reward_values=global_reward_values.tolist(),
            )
            hbd_update_payload = planned_update.to_payload()
        hbd_update_payload = broadcast_object(hbd_update_payload, self.accelerator)
        self._hbd_memory.apply_update(hbd_update_payload)
        adjusted_rewards = torch.tensor(
            hbd_update_payload['final_rewards'],
            device=self.device,
            dtype=torch.float32,
        )
        return adjusted_rewards, hbd_update_payload['metrics']

    def _select_diverse_rollout(self, rollout, local_specs, *, seed):
        if not self.config.diverse_minibatch:
            active_mask = torch.ones(
                len(local_specs),
                device=self.device,
                dtype=torch.bool,
            )
            return rollout, local_specs, active_mask, {}

        selection = select_molecule_groups(
            rollout.smiles,
            candidate_size=self.candidate_group_size,
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
        if index_tensor.numel() != self.local_sample_count:
            raise RuntimeError(
                'Diverse Mini-Batch selection returned the wrong fixed batch size: '
                f'{index_tensor.numel()} vs {self.local_sample_count}'
            )
        selected_rollout = replace(
            rollout,
            prompt_ids=rollout.prompt_ids.index_select(0, index_tensor),
            completion_ids=rollout.completion_ids.index_select(0, index_tensor),
            completion_mask=(
                rollout.completion_mask.index_select(0, index_tensor)
                & active_mask.unsqueeze(1)
            ),
            full_token_ids=rollout.full_token_ids.index_select(0, index_tensor),
            specs=[rollout.specs[idx] for idx in selection.indices],
            safe_strings=[rollout.safe_strings[idx] for idx in selection.indices],
            smiles=[rollout.smiles[idx] for idx in selection.indices],
        )
        selected_specs = [local_specs[idx] for idx in selection.indices]
        metrics = {
            f'diverse_minibatch/{key}': float(value)
            for key, value in selection.metrics.items()
        }
        metrics['diverse_minibatch/enabled'] = 1.0
        return selected_rollout, selected_specs, active_mask, metrics

    def _score_selected_rollout(self, rollout, active_mask):
        if active_mask.dim() != 1 or active_mask.numel() != len(rollout.smiles):
            raise ValueError('active_mask must match the selected rollout batch')
        active_positions = torch.nonzero(
            active_mask,
            as_tuple=False,
        ).view(-1).tolist()
        if not active_positions:
            raise RuntimeError('selected rollout contains no active samples')
        active_records = self.reward_model.score(
            [rollout.smiles[idx] for idx in active_positions]
        )
        if len(active_records) != len(active_positions):
            raise RuntimeError(
                'reward scorer returned an unexpected record count: '
                f'{len(active_records)} vs {len(active_positions)}'
            )
        records = [
            RewardRecord(
                reward=0.0,
                is_valid=False,
                alert_hit=False,
                qed=None,
                sa=None,
                sa_score=None,
                soft_reward=None,
                smiles=rollout.smiles[idx],
            )
            for idx in range(len(rollout.smiles))
        ]
        for position, record in zip(active_positions, active_records):
            records[position] = record
        return records

    def _generate_and_score_completions(self, mode):
        cycle_seed = self.config.seed + self.generation_cycle_idx * 10000
        if self.accelerator.is_main_process:
            if self.config.rl_algorithm == 'coupled_sgrpo':
                group_specs = sample_supergroup_shared_specs(
                    num_groups=self.num_groups_global,
                    supergroup_num_groups=self.config.supergroup_num_groups,
                    generation_temperature=self.config.generation_temperature,
                    randomness=self.config.randomness,
                    min_add_len=self.config.min_add_len,
                    max_completion_length=self.config.max_completion_length,
                    seed=cycle_seed,
                )
            else:
                group_specs = sample_group_specs(
                    num_groups=self.num_groups_global,
                    generation_temperature=self.config.generation_temperature,
                    randomness=self.config.randomness,
                    min_add_len=self.config.min_add_len,
                    max_completion_length=self.config.max_completion_length,
                    seed=cycle_seed,
                )
        else:
            group_specs = []

        expanded_specs = broadcast_specs(
            group_specs,
            self.candidate_group_size,
            self.accelerator,
        )
        if len(expanded_specs) != self.global_candidate_count:
            raise ValueError(
                f'Expected {self.global_candidate_count} candidate specs, '
                f'got {len(expanded_specs)}'
            )
        candidate_local_start = (
            self.accelerator.process_index * self.local_candidate_count
        )
        candidate_local_end = candidate_local_start + self.local_candidate_count
        local_specs = expanded_specs[candidate_local_start:candidate_local_end]
        rollout_seed = cycle_seed + self.accelerator.process_index * 1000
        rollout = self.policy.rollout_specs(
            specs=local_specs,
            generation_batch_size=self.config.generation_batch_size,
            seed=rollout_seed,
        )
        rollout, local_specs, local_active_mask, selection_metrics = (
            self._select_diverse_rollout(
                rollout,
                local_specs,
                seed=cycle_seed + self.accelerator.process_index * 100000 + 991,
            )
        )

        reward_records = self._score_selected_rollout(
            rollout,
            local_active_mask,
        )
        local_rewards = torch.tensor(
            [record.reward for record in reward_records],
            device=self.device,
            dtype=torch.float32,
        )
        local_valid = torch.tensor(
            [record.is_valid for record in reward_records],
            device=self.device,
            dtype=torch.bool,
        )
        local_alert = torch.tensor(
            [record.alert_hit for record in reward_records],
            device=self.device,
            dtype=torch.bool,
        )
        local_soft = torch.tensor(
            [
                float('nan')
                if record.soft_reward is None
                else float(record.soft_reward)
                for record in reward_records
            ],
            device=self.device,
            dtype=torch.float32,
        )
        global_rewards = self.accelerator.gather(local_rewards).detach()
        global_active_mask = self.accelerator.gather(local_active_mask).detach()
        gathered_valid = self.accelerator.gather(local_valid).detach()
        gathered_alert = self.accelerator.gather(local_alert).detach()
        gathered_soft = self.accelerator.gather(local_soft).detach()
        local_start = self.accelerator.process_index * self.local_sample_count
        local_end = local_start + self.local_sample_count
        hbd_metrics = {}
        candidate_diversity_metrics = {}
        local_candidate_loo_credits = None
        global_group_rewards = None
        global_group_reward_credits = None
        global_group_mean_individual_rewards = None
        if (
            self.config.rl_algorithm == 'coupled_sgrpo'
            or self.config.diversity_regularizer_weight > 0.0
            or self._hbd_memory is not None
            or self.config.candidate_diversity_reward_weight > 0.0
        ):
            global_smiles = self._all_gather_objects(
                [record.smiles for record in reward_records]
            )
            if len(global_smiles) != self.global_sample_count:
                raise ValueError(
                    f'Expected {self.global_sample_count} gathered smiles, got {len(global_smiles)}'
                )
            if self._hbd_memory is not None:
                local_soft_for_hbd = torch.tensor(
                    [
                        float('-inf') if record.soft_reward is None else float(record.soft_reward)
                        for record in reward_records
                    ],
                    device=self.device,
                    dtype=torch.float32,
                )
                global_soft_for_hbd = self.accelerator.gather(local_soft_for_hbd).detach()
                global_rewards, hbd_metrics = self._apply_hbd(
                    global_items=global_smiles,
                    global_memory_scores=global_soft_for_hbd,
                    global_reward_values=global_rewards,
                )
            include_loo_credits = (
                self.config.candidate_diversity_reward_weight > 0.0
                or (
                    self.config.rl_algorithm == 'coupled_sgrpo'
                    and self.config.group_rewrad_credit == 'loo'
                )
            )
            (
                global_group_rewards,
                global_group_reward_credits,
            ) = self._compute_group_diversity_statistics(
                global_smiles,
                include_loo_credits=include_loo_credits,
            )
            if self.config.candidate_diversity_reward_weight > 0.0:
                if global_group_reward_credits is None:
                    raise RuntimeError(
                        'candidate diversity reward requires LOO credits'
                    )
                (
                    global_rewards,
                    candidate_diversity_metrics,
                ) = self._apply_candidate_diversity_reward(
                    global_soft_rewards=gathered_soft,
                    global_valid_mask=gathered_valid,
                    global_alert_mask=gathered_alert,
                    global_loo_credits=global_group_reward_credits,
                    global_group_diversities=global_group_rewards,
                )
                local_candidate_loo_credits = (
                    global_group_reward_credits[local_start:local_end]
                    .to(device=self.device)
                )
        if self.config.rl_algorithm == 'coupled_sgrpo' and _has_active_individual_reward_thresholds(
            self.config.individual_reward_thresholds
        ):
            local_qed_for_threshold = torch.tensor(
                [0.0 if record.qed is None else float(record.qed) for record in reward_records],
                device=self.device,
                dtype=torch.float32,
            )
            local_sa_score_for_threshold = torch.tensor(
                [0.0 if record.sa_score is None else float(record.sa_score) for record in reward_records],
                device=self.device,
                dtype=torch.float32,
            )
            global_group_mean_individual_rewards = _build_group_mean_individual_rewards(
                qed_values=self.accelerator.gather(local_qed_for_threshold).detach(),
                sa_score_values=self.accelerator.gather(local_sa_score_for_threshold).detach(),
                num_generations=self.config.num_generations,
                device=self.device,
            )

        local_diversity_regularizer_advantages = None
        if self.config.rl_algorithm == 'coupled_grpo':
            global_advantages, global_reward_std, zero_std_ratio = compute_grouped_advantages(
                rewards=global_rewards,
                num_generations=self.optimization_group_size,
                scale_rewards=self.config.scale_rewards,
                advantage_baseline=self.config.advantage_baseline,
                sample_mask=(
                    global_active_mask
                    if self.config.diverse_minibatch
                    else None
                ),
            )
            local_advantages = global_advantages[local_start:local_end].to(device=self.device)
            extra_advantage_metrics = {}
        else:
            global_advantages, _, _, sgrpo_metrics = compute_sgrpo_advantages(
                rollout_rewards=global_rewards,
                group_rewards=global_group_rewards,
                num_generations=self.config.num_generations,
                supergroup_num_groups=self.config.supergroup_num_groups,
                group_advantage_weight=self.config.group_advantage_weight,
                scale_rewards=self.config.scale_rewards,
                hierarchy=self.config.hierarchy,
                group_mean_individual_rewards=global_group_mean_individual_rewards,
                individual_reward_thresholds=self.config.individual_reward_thresholds,
                group_rewrad_credit=self.config.group_rewrad_credit,
                group_rewrad_credit_temperature=self.config.group_rewrad_credit_temperature,
                group_reward_credits=global_group_reward_credits,
                advantage_baseline=self.config.advantage_baseline,
            )
            local_advantages = global_advantages[local_start:local_end].to(device=self.device)
            global_reward_std = torch.full_like(global_rewards, float(sgrpo_metrics['rollout_reward_std_mean']))
            zero_std_ratio = float(sgrpo_metrics['rollout_zero_std_ratio'])
            extra_advantage_metrics = {
                'rollout_advantage_mean': float(sgrpo_metrics['rollout_advantage_mean']),
                'group_advantage_mean': float(sgrpo_metrics['group_advantage_mean']),
                'group_reward/diversity_mean': float(sgrpo_metrics['group_reward_mean']),
                'group_reward/raw_diversity_mean': float(sgrpo_metrics['group_reward_raw_mean']),
                'group_reward/indicator_mean': float(sgrpo_metrics['group_reward_indicator_mean']),
                'rollout_zero_std_ratio': float(sgrpo_metrics['rollout_zero_std_ratio']),
                'group_zero_std_ratio': float(sgrpo_metrics['group_zero_std_ratio']),
                'sgrpo/hierarchy_reward_sum_enabled': float(sgrpo_metrics['hierarchy_reward_sum_enabled']),
                'sgrpo/hierarchy_hierarchical_sum_enabled': float(
                    sgrpo_metrics['hierarchy_hierarchical_sum_enabled']
                ),
                'sgrpo/group_rewrad_credit_loo_enabled': float(
                    sgrpo_metrics['group_rewrad_credit_loo_enabled']
                ),
            }
        extra_advantage_metrics.update(
            {
                f'candidate_diversity/{key}': float(value)
                for key, value in candidate_diversity_metrics.items()
            }
        )
        if self.config.diversity_regularizer_weight > 0.0:
            if global_group_rewards is None:
                raise ValueError('global_group_rewards must be populated when diversity_regularizer_weight > 0')
            global_diversity_regularizer_advantages, diversity_regularizer_metrics = (
                compute_group_reward_regularizer_advantages(
                    group_rewards=global_group_rewards,
                    num_generations=self.config.num_generations,
                    scale_rewards=self.config.scale_rewards,
                )
            )
            local_diversity_regularizer_advantages = global_diversity_regularizer_advantages[local_start:local_end].to(
                device=self.device
            )
            extra_advantage_metrics.update(
                {
                    'diversity_regularizer/advantage_mean': float(diversity_regularizer_metrics['group_advantage_mean']),
                    'diversity_regularizer/group_reward_mean': float(diversity_regularizer_metrics['group_reward_mean']),
                    'diversity_regularizer/zero_std_ratio': float(diversity_regularizer_metrics['group_zero_std_ratio']),
                }
            )
        extra_advantage_metrics.update({f'hbd/{key}': float(value) for key, value in hbd_metrics.items()})
        extra_advantage_metrics.update(selection_metrics)
        local_final_rewards = global_rewards[local_start:local_end].to(device=self.device)

        local_qed = torch.tensor(
            [float('nan') if record.qed is None else float(record.qed) for record in reward_records],
            device=self.device,
        )
        local_sa = torch.tensor(
            [float('nan') if record.sa is None else float(record.sa) for record in reward_records],
            device=self.device,
        )
        local_sa_score = torch.tensor(
            [float('nan') if record.sa_score is None else float(record.sa_score) for record in reward_records],
            device=self.device,
        )
        local_lengths = rollout.completion_mask.sum(dim=1).float()

        gathered_qed = self.accelerator.gather(local_qed)
        gathered_sa = self.accelerator.gather(local_sa)
        gathered_sa_score = self.accelerator.gather(local_sa_score)
        gathered_lengths = self.accelerator.gather(local_lengths)
        gathered_advantages = self.accelerator.gather(local_advantages.detach())

        mask_seeds = self._sample_mask_seeds()
        prompt_completion_ids = torch.cat([rollout.prompt_ids, rollout.completion_ids], dim=1)
        logits_to_keep = rollout.completion_ids.size(1)
        expanded_ids = prompt_completion_ids.unsqueeze(0).expand(self.config.num_iterations, -1, -1)
        if self.config.num_iterations > 1:
            old_per_token_logps = self.policy.per_token_logps(
                input_ids=expanded_ids,
                logits_to_keep=logits_to_keep,
                completion_mask=rollout.completion_mask,
                mask_seeds=mask_seeds,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                requires_grad=False,
            )
        else:
            old_per_token_logps = None

        if self.config.beta == 0.0:
            ref_per_token_logps = None
        else:
            ref_per_token_logps = self.reference.per_token_logps(
                input_ids=expanded_ids,
                logits_to_keep=logits_to_keep,
                completion_mask=rollout.completion_mask,
                mask_seeds=mask_seeds,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                requires_grad=False,
            )

        log_rows = []
        for row_idx, (
            spec,
            safe_string,
            record,
            final_reward,
            is_active,
        ) in enumerate(
            zip(
                local_specs,
                rollout.safe_strings,
                reward_records,
                local_final_rewards.tolist(),
                local_active_mask.tolist(),
            )
        ):
            if not is_active:
                continue
            row = {
                'mode': mode,
                'buffer_cycle': self.generation_cycle_idx,
                'step': self.global_step,
                'spec': spec.__dict__,
                'safe': safe_string,
                'smiles': record.smiles,
                'reward': float(final_reward),
                'qed': record.qed,
                'sa': record.sa,
                'sa_score': record.sa_score,
                'soft_reward': record.soft_reward,
                'is_valid': record.is_valid,
                'alert_hit': record.alert_hit,
            }
            if local_candidate_loo_credits is not None:
                row['raw_loo_diversity_reward'] = float(
                    local_candidate_loo_credits[row_idx].item()
                )
                row['combined_reward'] = float(final_reward)
            log_rows.append(row)

        active_global_rewards = global_rewards[global_active_mask]
        active_global_reward_std = global_reward_std[global_active_mask]
        active_gathered_advantages = gathered_advantages[global_active_mask]
        active_gathered_lengths = gathered_lengths[global_active_mask]
        active_gathered_valid = gathered_valid[global_active_mask]
        active_gathered_alert = gathered_alert[global_active_mask]
        valid_fraction = (
            active_gathered_valid.to(dtype=torch.float32).mean().item()
        )
        alert_hit_fraction = (
            active_gathered_alert.to(dtype=torch.float32).mean().item()
        )
        metadata = {
            'buffer_cycle': self.generation_cycle_idx,
            'reward_mean': active_global_rewards.mean().item(),
            'reward_std': active_global_reward_std.mean().item(),
            'advantage_mean': active_gathered_advantages.mean().item(),
            'zero_std_ratio': zero_std_ratio,
            'completion_length': active_gathered_lengths.mean().item(),
            'valid_fraction': valid_fraction,
            'alert_hit_fraction': alert_hit_fraction,
            'invalid_fraction': 1.0 - valid_fraction,
            'rewards/qed_mean': _nanmean(gathered_qed),
            'rewards/sa_mean': _nanmean(gathered_sa),
            'rewards/sa_score_mean': _nanmean(gathered_sa_score),
            'rewards/soft_mean': _nanmean(gathered_soft),
        }
        metadata.update(extra_advantage_metrics)
        self._record_reward_metrics(mode, metadata)
        self._record_text_logs(mode, log_rows)

        return {
            'prompt_ids': rollout.prompt_ids,
            'completion_ids': rollout.completion_ids,
            'completion_mask': rollout.completion_mask,
            'advantages': local_advantages,
            'diversity_regularizer_advantages': local_diversity_regularizer_advantages,
            'old_per_token_logps': old_per_token_logps,
            'ref_per_token_logps': ref_per_token_logps,
            'mask_seeds': mask_seeds,
        }, metadata

    def _prepare_inputs(self, mode='train'):
        generate_every = self.config.gradient_accumulation_steps * self.config.num_iterations
        if self._step % generate_every == 0 or self._buffered_inputs is None:
            accumulated_local_batch, metadata = self._generate_and_score_completions(mode=mode)
            self._buffered_inputs = split_tensor_dict(accumulated_local_batch, self.config.gradient_accumulation_steps)
            self._buffer_metadata = metadata
            self.generation_cycle_idx += 1

        inputs = self._buffered_inputs[self._step % self.config.gradient_accumulation_steps]
        self._step += 1
        return inputs

    def _compute_loss(self, inputs, mode='train', iteration_idx=None, requires_grad=True):
        if iteration_idx is None:
            iteration_idx = self._step % self.config.num_iterations
        prompt_completion_ids = torch.cat([inputs['prompt_ids'], inputs['completion_ids']], dim=1).unsqueeze(0)
        logits_to_keep = inputs['completion_ids'].size(1)
        current_seed = [inputs['mask_seeds'][iteration_idx]]
        policy_outputs = self.policy.per_token_logps(
            input_ids=prompt_completion_ids,
            logits_to_keep=logits_to_keep,
            completion_mask=inputs['completion_mask'],
            mask_seeds=current_seed,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            requires_grad=requires_grad,
            return_normalized_entropy=(
                self.config.entropy_regularization_weight > 0.0
            ),
        )
        if self.config.entropy_regularization_weight > 0.0:
            per_token_logps, per_token_normalized_entropy = policy_outputs
        else:
            per_token_logps = policy_outputs
            per_token_normalized_entropy = None
        if inputs['old_per_token_logps'] is None:
            old_per_token_logps = per_token_logps.detach()
        else:
            old_per_token_logps = inputs['old_per_token_logps'][:, iteration_idx, :].unsqueeze(1)

        if inputs['ref_per_token_logps'] is None:
            ref_per_token_logps = None
        else:
            ref_per_token_logps = inputs['ref_per_token_logps'][:, iteration_idx, :].unsqueeze(1)

        loss, step_metrics = compute_clipped_grpo_loss(
            new_log_probs=per_token_logps,
            old_log_probs=old_per_token_logps,
            advantages=inputs['advantages'],
            completion_mask=inputs['completion_mask'],
            epsilon=self.config.epsilon,
            ref_log_probs=ref_per_token_logps,
            beta=self.config.beta,
        )
        if self.config.entropy_regularization_weight > 0.0:
            if per_token_normalized_entropy is None:
                raise RuntimeError(
                    'policy did not return entropy while entropy regularization is enabled'
                )
            entropy_mask = inputs['completion_mask'].to(dtype=torch.float32)
            normalized_entropy_mean = (
                per_token_normalized_entropy[:, 0, :] * entropy_mask
            ).sum() / entropy_mask.sum().clamp(min=1.0)
            entropy_regularization_loss = -normalized_entropy_mean
            loss = loss + (
                self.config.entropy_regularization_weight
                * entropy_regularization_loss
            )
            step_metrics['entropy/normalized_mean'] = (
                normalized_entropy_mean.detach()
            )
            step_metrics['entropy/regularization_loss'] = (
                entropy_regularization_loss.detach()
            )
        if self.config.diversity_regularizer_weight > 0.0:
            diversity_regularizer_advantages = inputs['diversity_regularizer_advantages']
            if diversity_regularizer_advantages is None:
                raise ValueError(
                    'diversity_regularizer_advantages must be present when diversity_regularizer_weight > 0'
                )
            diversity_regularizer_loss, _ = compute_clipped_grpo_loss(
                new_log_probs=per_token_logps,
                old_log_probs=old_per_token_logps,
                advantages=diversity_regularizer_advantages,
                completion_mask=inputs['completion_mask'],
                epsilon=self.config.epsilon,
                ref_log_probs=None,
                beta=0.0,
            )
            loss = loss + (self.config.diversity_regularizer_weight * diversity_regularizer_loss)
            step_metrics['diversity_regularizer_loss'] = diversity_regularizer_loss.detach()
        self._record_loss_metrics(mode, step_metrics)
        return loss

    def _consume_logged_metrics(self, mode, step, buffer_cycle):
        bucket = self._metrics[mode]
        last_reward_metrics = self._last_reward_metrics[mode] or {}

        def _reward_metric(name):
            if bucket[name]:
                return _aggregate_scalar_list(bucket[name])
            return float(last_reward_metrics.get(name, float('nan')))

        metrics = {
            'step': step,
            'buffer_cycle': buffer_cycle,
            'reward': _reward_metric('reward'),
            'reward_std': _reward_metric('reward_std'),
            'advantage_mean': _reward_metric('advantage_mean'),
            'zero_std_ratio': _reward_metric('zero_std_ratio'),
            'completion_length': _reward_metric('completion_length'),
            'valid_fraction': _reward_metric('valid_fraction'),
            'alert_hit_fraction': _reward_metric('alert_hit_fraction'),
            'invalid_fraction': _reward_metric('invalid_fraction'),
            'rewards/qed_mean': _reward_metric('rewards/qed_mean'),
            'rewards/sa_mean': _reward_metric('rewards/sa_mean'),
            'rewards/sa_score_mean': _reward_metric('rewards/sa_score_mean'),
            'rewards/soft_mean': _reward_metric('rewards/soft_mean'),
            'ratio_mean': _aggregate_scalar_list(bucket['ratio_mean']),
            'clip_ratio/low_mean': _aggregate_scalar_list(bucket['clip_ratio/low_mean']),
            'clip_ratio/low_min': _aggregate_scalar_list(bucket['clip_ratio/low_min']),
            'clip_ratio/high_mean': _aggregate_scalar_list(bucket['clip_ratio/high_mean']),
            'clip_ratio/high_max': _aggregate_scalar_list(bucket['clip_ratio/high_max']),
            'clip_ratio/region_mean': _aggregate_scalar_list(bucket['clip_ratio/region_mean']),
            'grad_norm': float(bucket['grad_norm'][-1]) if bucket['grad_norm'] else float('nan'),
            'lr': float(bucket['lr'][-1]) if bucket['lr'] else float('nan'),
        }
        if bucket['rollout_advantage_mean'] or 'rollout_advantage_mean' in last_reward_metrics:
            metrics['rollout_advantage_mean'] = _reward_metric('rollout_advantage_mean')
        if bucket['group_advantage_mean'] or 'group_advantage_mean' in last_reward_metrics:
            metrics['group_advantage_mean'] = _reward_metric('group_advantage_mean')
        if bucket['group_reward/diversity_mean'] or 'group_reward/diversity_mean' in last_reward_metrics:
            metrics['group_reward/diversity_mean'] = _reward_metric('group_reward/diversity_mean')
        if bucket['rollout_zero_std_ratio'] or 'rollout_zero_std_ratio' in last_reward_metrics:
            metrics['rollout_zero_std_ratio'] = _reward_metric('rollout_zero_std_ratio')
        if bucket['group_zero_std_ratio'] or 'group_zero_std_ratio' in last_reward_metrics:
            metrics['group_zero_std_ratio'] = _reward_metric('group_zero_std_ratio')
        if (
            bucket['diversity_regularizer/advantage_mean']
            or 'diversity_regularizer/advantage_mean' in last_reward_metrics
        ):
            metrics['diversity_regularizer/advantage_mean'] = _reward_metric('diversity_regularizer/advantage_mean')
        if (
            bucket['diversity_regularizer/group_reward_mean']
            or 'diversity_regularizer/group_reward_mean' in last_reward_metrics
        ):
            metrics['diversity_regularizer/group_reward_mean'] = _reward_metric('diversity_regularizer/group_reward_mean')
        if (
            bucket['diversity_regularizer/zero_std_ratio']
            or 'diversity_regularizer/zero_std_ratio' in last_reward_metrics
        ):
            metrics['diversity_regularizer/zero_std_ratio'] = _reward_metric('diversity_regularizer/zero_std_ratio')
        for name in (
            'hbd/enabled',
            'hbd/eligible_count',
            'hbd/penalized_count',
            'hbd/accepted_existing_count',
            'hbd/created_bucket_count',
            'hbd/bucket_count_before',
            'hbd/bucket_count_after',
        ):
            if bucket[name] or name in last_reward_metrics:
                metrics[name] = _reward_metric(name)
        for name, values in bucket.items():
            if name.startswith('diverse_minibatch/') and values:
                metrics[name] = _aggregate_scalar_list(values)
            if (
                name.startswith('candidate_diversity/')
                and (values or name in last_reward_metrics)
            ):
                metrics[name] = _reward_metric(name)
        for name in last_reward_metrics:
            if (
                name.startswith('candidate_diversity/')
                and name not in metrics
            ):
                metrics[name] = _reward_metric(name)
        metrics['reward_mean'] = metrics['reward']
        if bucket['kl']:
            metrics['kl'] = _aggregate_scalar_list(bucket['kl'])
            metrics['kl_mean'] = metrics['kl']
        if bucket['diversity_regularizer/loss']:
            metrics['diversity_regularizer/loss'] = _aggregate_scalar_list(bucket['diversity_regularizer/loss'])
        for name in (
            'entropy/normalized_mean',
            'entropy/regularization_loss',
        ):
            if bucket[name]:
                metrics[name] = _aggregate_scalar_list(bucket[name])
        self._metrics[mode] = defaultdict(list)
        return metrics

    def _log_metrics(self, split, metrics):
        if self.accelerator.is_main_process:
            logger.info(
                '%s step=%s reward_mean=%.6f reward_std=%.6f advantage_mean=%.6f ratio_mean=%.6f '
                'clip_ratio/low_mean=%.6f clip_ratio/low_min=%.6f clip_ratio/high_mean=%.6f '
                'clip_ratio/high_max=%.6f clip_ratio/region_mean=%.6f completion_length=%.6f '
                'zero_std_ratio=%.6f valid_fraction=%.6f alert_hit_fraction=%.6f invalid_fraction=%.6f '
                'rewards/qed_mean=%.6f rewards/sa_mean=%.6f rewards/sa_score_mean=%.6f rewards/soft_mean=%.6f '
                'grad_norm=%.6f lr=%.8f%s',
                split,
                metrics['step'],
                metrics['reward_mean'],
                metrics['reward_std'],
                metrics['advantage_mean'],
                metrics['ratio_mean'],
                metrics['clip_ratio/low_mean'],
                metrics['clip_ratio/low_min'],
                metrics['clip_ratio/high_mean'],
                metrics['clip_ratio/high_max'],
                metrics['clip_ratio/region_mean'],
                metrics['completion_length'],
                metrics['zero_std_ratio'],
                metrics['valid_fraction'],
                metrics['alert_hit_fraction'],
                metrics['invalid_fraction'],
                metrics['rewards/qed_mean'],
                metrics['rewards/sa_mean'],
                metrics['rewards/sa_score_mean'],
                metrics['rewards/soft_mean'],
                metrics['grad_norm'],
                metrics['lr'],
                (
                    ''
                    if 'rollout_advantage_mean' not in metrics
                    else (
                        f" rollout_advantage_mean={metrics['rollout_advantage_mean']:.6f}"
                        f" group_advantage_mean={metrics['group_advantage_mean']:.6f}"
                        f" group_reward/diversity_mean={metrics['group_reward/diversity_mean']:.6f}"
                        f" rollout_zero_std_ratio={metrics['rollout_zero_std_ratio']:.6f}"
                        f" group_zero_std_ratio={metrics['group_zero_std_ratio']:.6f}"
                    )
                )
                + (
                    ''
                    if 'diversity_regularizer/advantage_mean' not in metrics
                    else (
                        f" diversity_regularizer/advantage_mean={metrics['diversity_regularizer/advantage_mean']:.6f}"
                        f" diversity_regularizer/group_reward_mean={metrics['diversity_regularizer/group_reward_mean']:.6f}"
                        f" diversity_regularizer/zero_std_ratio={metrics['diversity_regularizer/zero_std_ratio']:.6f}"
                    )
                )
                + (
                    ''
                    if 'diversity_regularizer/loss' not in metrics
                    else f" diversity_regularizer/loss={metrics['diversity_regularizer/loss']:.6f}"
                )
                + (
                    ''
                    if 'candidate_diversity/raw_loo_mean' not in metrics
                    else (
                        f" candidate_diversity/raw_loo_mean="
                        f"{metrics['candidate_diversity/raw_loo_mean']:.6f}"
                        f" candidate_diversity/combined_reward_mean="
                        f"{metrics['candidate_diversity/combined_reward_mean']:.6f}"
                    )
                )
                + ('' if 'kl_mean' not in metrics else f" kl_mean={metrics['kl_mean']:.6f}"),
            )
            write_jsonl(self.metrics_path, metrics)
            with open(self.state_path, 'w') as handle:
                json.dump(
                    {
                        'global_step': self.global_step,
                        'micro_step': self._step,
                        'generation_cycle_idx': self.generation_cycle_idx,
                        'last_metrics': metrics,
                    },
                    handle,
                    sort_keys=True,
                    indent=2,
                )
        if self.config.report_to:
            self.accelerator.log(metrics, step=self.global_step)
        self._flush_text_logs(split)

    def _checkpoint_dir(self):
        return os.path.join(self.output_dir, f'checkpoint-{self.global_step:06d}')

    def _save_checkpoint(self):
        checkpoint_dir = self._checkpoint_dir()
        accelerator_state_dir = os.path.join(checkpoint_dir, 'accelerator_state')
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(accelerator_state_dir)
        self.accelerator.wait_for_everyone()

        if not self.accelerator.is_main_process:
            return

        model_path = os.path.join(checkpoint_dir, 'model.ckpt')
        ref_path = os.path.join(checkpoint_dir, 'reference_backbone.pt')
        trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
        self.policy.save_checkpoint(model_path, step=self.global_step, accelerator=self.accelerator)
        torch.save(self.reference.get_backbone_state_dict(), ref_path)
        with open(trainer_state_path, 'w') as handle:
            json.dump(
                {
                    'global_step': self.global_step,
                    'micro_step': self._step,
                    'generation_cycle_idx': self.generation_cycle_idx,
                    'last_metrics': self._last_train_metrics,
                    'hbd_state': None if self._hbd_memory is None else self._hbd_memory.state_dict(),
                },
                handle,
                sort_keys=True,
                indent=2,
            )
        maybe_trim_checkpoints(self.output_dir, self.config.save_total_limit)

    def _load_checkpoint(self, checkpoint_dir):
        accelerator_state_dir = os.path.join(checkpoint_dir, 'accelerator_state')
        trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
        model_path = os.path.join(checkpoint_dir, 'model.ckpt')
        ref_path = os.path.join(checkpoint_dir, 'reference_backbone.pt')
        ensure_exists(accelerator_state_dir, 'accelerator state')
        ensure_exists(trainer_state_path, 'trainer state')
        ensure_exists(model_path, 'model checkpoint')
        ensure_exists(ref_path, 'reference state')

        with open(trainer_state_path) as handle:
            trainer_state = json.load(handle)

        self.accelerator.load_state(accelerator_state_dir)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.policy.load_ema_state(checkpoint.get('ema'))
        self.reference.load_backbone_state_dict(torch.load(ref_path, map_location='cpu', weights_only=False))

        self.global_step = int(trainer_state['global_step'])
        self._step = int(trainer_state['micro_step'])
        self.generation_cycle_idx = int(trainer_state['generation_cycle_idx'])
        self._last_train_metrics = trainer_state.get('last_metrics')
        hbd_state = trainer_state.get('hbd_state')
        if self._hbd_memory is None:
            if hbd_state is not None:
                raise ValueError('checkpoint contains HBD state but current config has hbd disabled')
        else:
            if hbd_state is None:
                raise ValueError('checkpoint is missing HBD state while current config has hbd enabled')
            self._hbd_memory.load_state_dict(hbd_state)
        self._buffered_inputs = None
        self._buffer_metadata = None

    def train(self, resume_from_checkpoint=None):
        if resume_from_checkpoint is not None:
            logger.info('Resuming from checkpoint: %s', resume_from_checkpoint)
            self._load_checkpoint(resume_from_checkpoint)

        while self.global_step < self.config.max_steps:
            self.optimizer.zero_grad(set_to_none=True)

            for _ in range(self.config.gradient_accumulation_steps):
                inputs = self._prepare_inputs(mode='train')
                loss = self._compute_loss(inputs, mode='train', requires_grad=True)
                self.accelerator.backward(loss / self.config.gradient_accumulation_steps)

            grad_norm = self.accelerator.clip_grad_norm_(
                self.policy.model.backbone.parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.policy.update_ema()
            self.global_step += 1
            self._record_optimizer_metrics('train', grad_norm=float(grad_norm), lr=self.scheduler.get_last_lr()[0])

            if self.config.sync_ref_model and self.global_step % self.config.ref_model_sync_steps == 0:
                self.reference.sync_from(self.policy, alpha=self.config.ref_model_mixup_alpha)

            should_log = self.global_step == 1 and self.config.logging_first_step
            should_log = should_log or (self.global_step % self.config.logging_steps == 0)
            if should_log:
                self._last_train_metrics = self._consume_logged_metrics(
                    'train',
                    step=self.global_step,
                    buffer_cycle=self._buffer_metadata['buffer_cycle'],
                )
                self._log_metrics('train', self._last_train_metrics)

            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint()

        if self._has_pending_metrics('train'):
            self._last_train_metrics = self._consume_logged_metrics(
                'train',
                step=self.global_step,
                buffer_cycle=self._buffer_metadata['buffer_cycle'] if self._buffer_metadata is not None else self.generation_cycle_idx,
            )

        return TrainResult(metrics=self._last_train_metrics or {})

    def evaluate(self):
        current_train_state = self.policy.model.backbone.training
        self._metrics['eval'] = defaultdict(list)
        self._textual_logs['eval'] = []
        self.policy.model.backbone.eval()
        try:
            inputs, metadata = self._generate_and_score_completions(mode='eval')
            for iteration_idx in range(self.config.num_iterations):
                self._compute_loss(
                    inputs,
                    mode='eval',
                    iteration_idx=iteration_idx,
                    requires_grad=False,
                )
            self._record_optimizer_metrics('eval', grad_norm=float('nan'), lr=self.scheduler.get_last_lr()[0])
        finally:
            self.policy.model.backbone.train(current_train_state)
        return self._consume_logged_metrics('eval', step=self.global_step, buffer_cycle=metadata['buffer_cycle'])

    def log_metrics(self, split, metrics):
        self._log_metrics(split, metrics)

    def save_metrics(self, split, metrics):
        if not self.accelerator.is_main_process:
            return
        path = os.path.join(self.output_dir, f'{split}_results.json')
        with open(path, 'w') as handle:
            json.dump(metrics, handle, sort_keys=True, indent=2)

    def save_state(self):
        if not self.accelerator.is_main_process:
            return
        with open(self.state_path, 'w') as handle:
            json.dump(
                {
                    'global_step': self.global_step,
                    'micro_step': self._step,
                    'generation_cycle_idx': self.generation_cycle_idx,
                    'last_metrics': self._last_train_metrics,
                    'hbd_state': None if self._hbd_memory is None else self._hbd_memory.state_dict(),
                },
                handle,
                sort_keys=True,
                indent=2,
            )

    def save_model(self, output_dir):
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            return
        self.policy.save_checkpoint(
            os.path.join(output_dir, 'final_model.ckpt'),
            step=self.global_step,
            accelerator=self.accelerator,
        )

    def close(self):
        self.reward_model.close()
