import bisect
import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime

import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed

from genmol.mm.crossdocked import load_crossdocked_manifest
from genmol.mm.policy import PocketPrefixCpGRPOPolicy
from genmol.mm.reward import (
    MolecularReward,
    RewardRecord,
    compute_internal_diversity,
    compute_internal_diversity_loo_credits,
    normalize_molecular_reward_weights,
)
from genmol.mm.utils import DrugCLIPConfig, UniDockConfig
from genmol.rl.cpgrpo import (
    VALID_GROUP_REWRAD_CREDITS,
    VALID_SGRPO_HIERARCHIES,
    compute_clipped_grpo_loss,
    compute_grouped_advantages,
    compute_group_reward_regularizer_advantages,
    compute_sgrpo_advantages,
    validate_reward_threshold_names,
    split_tensor_dict,
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
from genmol.rl.trainer import (
    _aggregate_scalar_list,
    _build_group_mean_individual_rewards,
    _has_active_individual_reward_thresholds,
    _validate_active_molecular_reward_thresholds,
    _nanmean,
    _nanreduce,
    GENMOL_SGRPO_THRESHOLD_REWARD_NAMES,
    build_scheduler,
    ensure_exists,
    find_last_checkpoint,
    maybe_trim_checkpoints,
    write_jsonl,
)


logger = logging.getLogger(__name__)


@dataclass
class PocketPrefixTrainConfig:
    model_variant: str
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
    drugclip_score: float | None = None
    unidock_score: float | None = None
    individual_reward_thresholds: dict[str, float | None] = field(default_factory=dict)
    group_rewrad_credit: str = 'broadcast'
    group_rewrad_credit_temperature: float = 1.0
    manifest_path: str = ''
    split: str = 'train'
    crossdocked_lmdb_path: str = ''
    crossdocked_split_path: str = ''
    crossdocked_root: str = ''
    drugclip_checkpoint_path: str | None = None
    drugclip_batch_size: int = 64
    drugclip_max_pocket_atoms: int = 256
    drugclip_num_conformers: int = 1
    drugclip_conformer_num_workers: int = 1
    drugclip_use_fp16: bool = True
    unidock_binary_path: str = 'unidock'
    unidock_batch_size: int = 128
    unidock_search_mode: str = 'fast'
    unidock_num_modes: int = 1
    unidock_num_cpu: int = 0
    unidock_max_gpu_memory_mb: int = 0
    unidock_timeout_sec: int = 1800
    unidock_prepare_timeout_sec: int = 300
    pocket_encoder: dict = field(default_factory=dict)
    conditioning: dict = field(default_factory=dict)
    hbd: bool = False
    hbd_bucket_size: int = 25
    hbd_score_threshold_for_memory: float = 0.6
    hbd_similarity_cutoff: float = 0.6
    diverse_minibatch: bool = False
    diverse_minibatch_oversample_factor: int = 2
    entropy_regularization_weight: float = 0.0


@dataclass
class TrainResult:
    metrics: dict


def load_config(path):
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    legacy_rollout_reward_weights = raw.pop('rollout_reward_weights', None)
    if legacy_rollout_reward_weights is not None and not isinstance(legacy_rollout_reward_weights, dict):
        raise TypeError('rollout_reward_weights must be a mapping when present in a saved config')
    config = PocketPrefixTrainConfig(**raw)
    if config.model_variant != 'pocket_prefix_mm':
        raise ValueError(f"Expected model_variant='pocket_prefix_mm', got {config.model_variant!r}")
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
    config.rollout_reward_weights = normalize_molecular_reward_weights(
        {
            'qed': config.qed,
            'sa_score': config.sa_score,
            'drugclip_score': config.drugclip_score,
            'unidock_score': config.unidock_score,
        }
    )
    if config.rollout_reward_weights['drugclip_score'] > 0.0:
        if not config.crossdocked_lmdb_path:
            raise ValueError('crossdocked_lmdb_path is required when drugclip_score reward weight is positive')
        if not config.drugclip_checkpoint_path:
            raise ValueError('drugclip_checkpoint_path is required when drugclip_score reward weight is positive')
        if config.drugclip_batch_size <= 0:
            raise ValueError('drugclip_batch_size must be positive')
        if config.drugclip_max_pocket_atoms <= 0:
            raise ValueError('drugclip_max_pocket_atoms must be positive')
        if config.drugclip_num_conformers <= 0:
            raise ValueError('drugclip_num_conformers must be positive')
        if config.drugclip_conformer_num_workers <= 0:
            raise ValueError('drugclip_conformer_num_workers must be positive')
    if config.rollout_reward_weights['unidock_score'] > 0.0:
        if not config.crossdocked_lmdb_path:
            raise ValueError('crossdocked_lmdb_path is required when unidock_score reward weight is positive')
        if config.unidock_batch_size <= 0:
            raise ValueError('unidock_batch_size must be positive')
        if config.unidock_num_modes <= 0:
            raise ValueError('unidock_num_modes must be positive')
        if config.unidock_num_cpu < 0:
            raise ValueError('unidock_num_cpu must be non-negative')
        if config.unidock_max_gpu_memory_mb < 0:
            raise ValueError('unidock_max_gpu_memory_mb must be non-negative')
        if config.unidock_timeout_sec <= 0:
            raise ValueError('unidock_timeout_sec must be positive')
        if config.unidock_prepare_timeout_sec <= 0:
            raise ValueError('unidock_prepare_timeout_sec must be positive')
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
    if not config.manifest_path:
        raise ValueError('manifest_path is required for pocket_prefix_mm de novo RL')
    if config.split not in {'train', 'val', 'test'}:
        raise ValueError(f'Unsupported split: {config.split!r}')
    if str(config.conditioning.get('overlength_policy')) != 'drop_sample':
        raise ValueError(
            f"Only conditioning.overlength_policy='drop_sample' is supported, got {config.conditioning.get('overlength_policy')!r}"
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
        base_dir = os.path.join(cluster_root, 'runs', 'cpgrpo_denovo_pocket_prefix')
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
        base_dir = os.path.join(repo_root, 'runs', 'cpgrpo_denovo_pocket_prefix')

    config_name = os.path.splitext(os.path.basename(config_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(base_dir, f'{config_name}_{timestamp}')


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


class PocketPrefixCpGRPOTrainer:
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
        ensure_exists(config.manifest_path, 'manifest')
        if config.rollout_reward_weights['drugclip_score'] > 0.0:
            ensure_exists(config.crossdocked_lmdb_path, 'CrossDocked LMDB')
            ensure_exists(config.drugclip_checkpoint_path, 'DrugCLIP checkpoint')
        if config.rollout_reward_weights['unidock_score'] > 0.0:
            ensure_exists(config.crossdocked_lmdb_path, 'CrossDocked LMDB')

        set_seed(config.seed, device_specific=True)

        self.policy = PocketPrefixCpGRPOPolicy(
            checkpoint_path=config.init_ckpt_path,
            device=self.device,
            bf16=config.bf16,
            trainable=True,
        )
        self.reference = PocketPrefixCpGRPOPolicy(
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
        self.policy.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.policy.model,
            optimizer,
            scheduler,
        )

        drugclip_config = None
        if self.config.rollout_reward_weights['drugclip_score'] > 0.0:
            drugclip_config = DrugCLIPConfig(
                checkpoint_path=str(self.config.drugclip_checkpoint_path),
                crossdocked_lmdb_path=str(self.config.crossdocked_lmdb_path),
                device=str(self.device),
                batch_size=int(self.config.drugclip_batch_size),
                max_pocket_atoms=int(self.config.drugclip_max_pocket_atoms),
                num_conformers=int(self.config.drugclip_num_conformers),
                conformer_num_workers=int(self.config.drugclip_conformer_num_workers),
                use_fp16=bool(self.config.drugclip_use_fp16),
                seed=int(self.config.seed),
            )
        unidock_config = None
        if self.config.rollout_reward_weights['unidock_score'] > 0.0:
            unidock_config = UniDockConfig(
                binary_path=str(self.config.unidock_binary_path),
                crossdocked_lmdb_path=str(self.config.crossdocked_lmdb_path),
                device=str(self.device),
                batch_size=int(self.config.unidock_batch_size),
                search_mode=str(self.config.unidock_search_mode),
                scoring='vina',
                num_modes=int(self.config.unidock_num_modes),
                num_cpu=int(self.config.unidock_num_cpu),
                max_gpu_memory_mb=int(self.config.unidock_max_gpu_memory_mb),
                timeout_sec=int(self.config.unidock_timeout_sec),
                prepare_timeout_sec=int(self.config.unidock_prepare_timeout_sec),
            )
        self.reward_model = MolecularReward(
            reward_weights=self.config.rollout_reward_weights,
            drugclip_config=drugclip_config,
            unidock_config=unidock_config,
        )
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

        self.manifest_entries, self.manifest_stats = load_crossdocked_manifest(
            manifest_path=config.manifest_path,
            split=config.split,
        )
        self.max_total_positions = int(config.conditioning.get('max_total_positions', 256))
        self._eligible_entries = sorted(self.manifest_entries, key=lambda item: int(item['residue_count']))
        self._eligible_residue_counts = [int(item['residue_count']) for item in self._eligible_entries]

        if config.report_to:
            init_kwargs = {}
            if 'wandb' in config.report_to:
                init_kwargs['wandb'] = {'name': os.path.basename(output_dir)}
            tracker_config = asdict(config)
            tracker_config['manifest_valid_samples'] = int(self.manifest_stats['valid_samples'])
            self.accelerator.init_trackers('genmol-cpgrpo-pocket-prefix', config=tracker_config, init_kwargs=init_kwargs or None)

        logger.info(
            'process_index=%s device=%s world_size=%s local_sample_count=%s global_sample_count=%s '
            'manifest_split=%s manifest_valid_samples=%s reward_workers=%s',
            self.accelerator.process_index,
            self.device,
            self.world_size,
            self.local_sample_count,
            self.global_sample_count,
            config.split,
            self.manifest_stats['valid_samples'],
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
            'rewards/drugclip_score_mean': float(metadata['rewards/drugclip_score_mean']),
            'rewards/unidock_score_mean': float(metadata['rewards/unidock_score_mean']),
            'rewards/soft_mean': float(metadata['rewards/soft_mean']),
            'reward_score_sec_total': float(metadata['reward_score_sec_total']),
            'reward_base_score_sec': float(metadata['reward_base_score_sec']),
            'reward_drugclip_score_sec': float(metadata['reward_drugclip_score_sec']),
            'reward_unidock_score_sec': float(metadata['reward_unidock_score_sec']),
            'base_valid_count': float(metadata['base_valid_count']),
            'base_valid_fraction': float(metadata['base_valid_fraction']),
            'drugclip_input_count': float(metadata['drugclip_input_count']),
            'drugclip_unique_smiles_count': float(metadata['drugclip_unique_smiles_count']),
            'drugclip_unique_pocket_count': float(metadata['drugclip_unique_pocket_count']),
            'drugclip_molecule_cache_hit_count': float(metadata['drugclip_molecule_cache_hit_count']),
            'drugclip_molecule_cache_miss_count': float(metadata['drugclip_molecule_cache_miss_count']),
            'drugclip_unique_smiles_success_count': float(metadata['drugclip_unique_smiles_success_count']),
            'drugclip_unique_smiles_failure_count': float(metadata['drugclip_unique_smiles_failure_count']),
            'drugclip_score_success_count': float(metadata['drugclip_score_success_count']),
            'drugclip_score_failure_count': float(metadata['drugclip_score_failure_count']),
            'drugclip_score_success_fraction': float(metadata['drugclip_score_success_fraction']),
            'drugclip_pocket_cache_hit_count': float(metadata['drugclip_pocket_cache_hit_count']),
            'drugclip_pocket_cache_miss_count': float(metadata['drugclip_pocket_cache_miss_count']),
            'drugclip_molecule_prepare_sec': float(metadata['drugclip_molecule_prepare_sec']),
            'drugclip_molecule_encode_sec': float(metadata['drugclip_molecule_encode_sec']),
            'drugclip_pocket_prepare_sec': float(metadata['drugclip_pocket_prepare_sec']),
            'drugclip_pocket_encode_sec': float(metadata['drugclip_pocket_encode_sec']),
            'drugclip_score_dot_sec': float(metadata['drugclip_score_dot_sec']),
            'drugclip_fail_smiles_parse_count': float(metadata['drugclip_fail_smiles_parse_count']),
            'drugclip_fail_embed_exception_count': float(metadata['drugclip_fail_embed_exception_count']),
            'drugclip_fail_zero_conformer_count': float(metadata['drugclip_fail_zero_conformer_count']),
            'drugclip_fail_empty_atom_list_count': float(metadata['drugclip_fail_empty_atom_list_count']),
            'unidock_input_count': float(metadata['unidock_input_count']),
            'unidock_unique_smiles_count': float(metadata['unidock_unique_smiles_count']),
            'unidock_unique_pocket_count': float(metadata['unidock_unique_pocket_count']),
            'unidock_score_success_count': float(metadata['unidock_score_success_count']),
            'unidock_score_failure_count': float(metadata['unidock_score_failure_count']),
            'unidock_score_success_fraction': float(metadata['unidock_score_success_fraction']),
            'unidock_cache_hit_count': float(metadata['unidock_cache_hit_count']),
            'unidock_cache_miss_count': float(metadata['unidock_cache_miss_count']),
            'unidock_prepare_sec': float(metadata['unidock_prepare_sec']),
            'unidock_dock_sec': float(metadata['unidock_dock_sec']),
            'unidock_parse_sec': float(metadata['unidock_parse_sec']),
            'unidock_chunk_count': float(metadata['unidock_chunk_count']),
            'unidock_rank_cpu_count': float(metadata['unidock_rank_cpu_count']),
            'unidock_max_gpu_memory_mb': float(metadata['unidock_max_gpu_memory_mb']),
            'unidock_fail_prepare_count': float(metadata['unidock_fail_prepare_count']),
            'unidock_fail_output_missing_count': float(metadata['unidock_fail_output_missing_count']),
            'unidock_fail_parse_count': float(metadata['unidock_fail_parse_count']),
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
        bucket['rewards/drugclip_score_mean'].append(float(metadata['rewards/drugclip_score_mean']))
        bucket['rewards/unidock_score_mean'].append(float(metadata['rewards/unidock_score_mean']))
        bucket['rewards/soft_mean'].append(float(metadata['rewards/soft_mean']))
        bucket['reward_score_sec_total'].append(float(metadata['reward_score_sec_total']))
        bucket['reward_base_score_sec'].append(float(metadata['reward_base_score_sec']))
        bucket['reward_drugclip_score_sec'].append(float(metadata['reward_drugclip_score_sec']))
        bucket['reward_unidock_score_sec'].append(float(metadata['reward_unidock_score_sec']))
        bucket['base_valid_count'].append(float(metadata['base_valid_count']))
        bucket['base_valid_fraction'].append(float(metadata['base_valid_fraction']))
        bucket['drugclip_input_count'].append(float(metadata['drugclip_input_count']))
        bucket['drugclip_unique_smiles_count'].append(float(metadata['drugclip_unique_smiles_count']))
        bucket['drugclip_unique_pocket_count'].append(float(metadata['drugclip_unique_pocket_count']))
        bucket['drugclip_molecule_cache_hit_count'].append(float(metadata['drugclip_molecule_cache_hit_count']))
        bucket['drugclip_molecule_cache_miss_count'].append(float(metadata['drugclip_molecule_cache_miss_count']))
        bucket['drugclip_unique_smiles_success_count'].append(float(metadata['drugclip_unique_smiles_success_count']))
        bucket['drugclip_unique_smiles_failure_count'].append(float(metadata['drugclip_unique_smiles_failure_count']))
        bucket['drugclip_score_success_count'].append(float(metadata['drugclip_score_success_count']))
        bucket['drugclip_score_failure_count'].append(float(metadata['drugclip_score_failure_count']))
        bucket['drugclip_score_success_fraction'].append(float(metadata['drugclip_score_success_fraction']))
        bucket['drugclip_pocket_cache_hit_count'].append(float(metadata['drugclip_pocket_cache_hit_count']))
        bucket['drugclip_pocket_cache_miss_count'].append(float(metadata['drugclip_pocket_cache_miss_count']))
        bucket['drugclip_molecule_prepare_sec'].append(float(metadata['drugclip_molecule_prepare_sec']))
        bucket['drugclip_molecule_encode_sec'].append(float(metadata['drugclip_molecule_encode_sec']))
        bucket['drugclip_pocket_prepare_sec'].append(float(metadata['drugclip_pocket_prepare_sec']))
        bucket['drugclip_pocket_encode_sec'].append(float(metadata['drugclip_pocket_encode_sec']))
        bucket['drugclip_score_dot_sec'].append(float(metadata['drugclip_score_dot_sec']))
        bucket['drugclip_fail_smiles_parse_count'].append(float(metadata['drugclip_fail_smiles_parse_count']))
        bucket['drugclip_fail_embed_exception_count'].append(float(metadata['drugclip_fail_embed_exception_count']))
        bucket['drugclip_fail_zero_conformer_count'].append(float(metadata['drugclip_fail_zero_conformer_count']))
        bucket['drugclip_fail_empty_atom_list_count'].append(float(metadata['drugclip_fail_empty_atom_list_count']))
        bucket['unidock_input_count'].append(float(metadata['unidock_input_count']))
        bucket['unidock_unique_smiles_count'].append(float(metadata['unidock_unique_smiles_count']))
        bucket['unidock_unique_pocket_count'].append(float(metadata['unidock_unique_pocket_count']))
        bucket['unidock_score_success_count'].append(float(metadata['unidock_score_success_count']))
        bucket['unidock_score_failure_count'].append(float(metadata['unidock_score_failure_count']))
        bucket['unidock_score_success_fraction'].append(float(metadata['unidock_score_success_fraction']))
        bucket['unidock_cache_hit_count'].append(float(metadata['unidock_cache_hit_count']))
        bucket['unidock_cache_miss_count'].append(float(metadata['unidock_cache_miss_count']))
        bucket['unidock_prepare_sec'].append(float(metadata['unidock_prepare_sec']))
        bucket['unidock_dock_sec'].append(float(metadata['unidock_dock_sec']))
        bucket['unidock_parse_sec'].append(float(metadata['unidock_parse_sec']))
        bucket['unidock_chunk_count'].append(float(metadata['unidock_chunk_count']))
        bucket['unidock_rank_cpu_count'].append(float(metadata['unidock_rank_cpu_count']))
        bucket['unidock_max_gpu_memory_mb'].append(float(metadata['unidock_max_gpu_memory_mb']))
        bucket['unidock_fail_prepare_count'].append(float(metadata['unidock_fail_prepare_count']))
        bucket['unidock_fail_output_missing_count'].append(float(metadata['unidock_fail_output_missing_count']))
        bucket['unidock_fail_parse_count'].append(float(metadata['unidock_fail_parse_count']))
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
            if name.startswith('diverse_minibatch/'):
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

    def _eligible_manifest_entry(self, add_seq_len, rng):
        allowed_residue_count = self.max_total_positions - (int(add_seq_len) + 2)
        upper_bound = bisect.bisect_right(self._eligible_residue_counts, allowed_residue_count)
        if upper_bound <= 0:
            raise ValueError(
                'No eligible pocket entries for sampled add_seq_len under the current 256-token budget: '
                f'add_seq_len={add_seq_len} allowed_residue_count={allowed_residue_count}'
            )
        sampled_index = rng.randrange(upper_bound)
        return self._eligible_entries[sampled_index]

    def _sample_expanded_pocket_entries(
        self,
        group_specs,
        seed,
        *,
        num_generations=None,
    ):
        group_size = (
            self.config.num_generations
            if num_generations is None
            else int(num_generations)
        )
        if group_size <= 0:
            raise ValueError('num_generations must be positive')
        rng = __import__('random').Random(seed)
        if self.config.rl_algorithm == 'coupled_grpo':
            sampled_group_entries = [
                self._eligible_manifest_entry(spec.add_seq_len, rng)
                for spec in group_specs
            ]
            expanded_entries = []
            for entry in sampled_group_entries:
                expanded_entries.extend([entry] * group_size)
            return expanded_entries

        expanded_entries = []
        for start in range(0, len(group_specs), self.config.supergroup_num_groups):
            supergroup_specs = group_specs[start:start + self.config.supergroup_num_groups]
            if any(spec != supergroup_specs[0] for spec in supergroup_specs[1:]):
                raise ValueError('Expected identical specs within each coupled_sgrpo supergroup')
            sampled_entry = self._eligible_manifest_entry(supergroup_specs[0].add_seq_len, rng)
            expanded_entries.extend(
                [sampled_entry] * (
                    self.config.supergroup_num_groups * group_size
                )
            )
        return expanded_entries

    def _select_diverse_rollout(
        self,
        rollout,
        local_specs,
        local_pocket_entries,
        local_pocket_raw_embeddings,
        local_pocket_mask,
        *,
        seed,
    ):
        if not self.config.diverse_minibatch:
            active_mask = torch.ones(
                len(local_specs),
                device=self.device,
                dtype=torch.bool,
            )
            return (
                rollout,
                local_specs,
                local_pocket_entries,
                local_pocket_raw_embeddings,
                local_pocket_mask,
                active_mask,
                {},
            )

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
        metrics = {
            f'diverse_minibatch/{key}': float(value)
            for key, value in selection.metrics.items()
        }
        metrics['diverse_minibatch/enabled'] = 1.0
        return (
            selected_rollout,
            [local_specs[idx] for idx in selection.indices],
            [local_pocket_entries[idx] for idx in selection.indices],
            local_pocket_raw_embeddings.index_select(0, index_tensor),
            local_pocket_mask.index_select(0, index_tensor),
            active_mask,
            metrics,
        )

    def _score_selected_rollout(
        self,
        rollout,
        local_pocket_entries,
        active_mask,
    ):
        if active_mask.dim() != 1 or active_mask.numel() != len(rollout.smiles):
            raise ValueError('active_mask must match the selected rollout batch')
        active_positions = torch.nonzero(
            active_mask,
            as_tuple=False,
        ).view(-1).tolist()
        if not active_positions:
            raise RuntimeError('selected rollout contains no active samples')
        active_records = self.reward_model.score(
            [rollout.smiles[idx] for idx in active_positions],
            pocket_entries=[
                local_pocket_entries[idx] for idx in active_positions
            ],
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
                drugclip_score=None,
                unidock_score=None,
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
            expanded_pocket_entries = self._sample_expanded_pocket_entries(
                group_specs=group_specs,
                seed=cycle_seed + 777,
                num_generations=self.candidate_group_size,
            )
        else:
            group_specs = []
            expanded_pocket_entries = None

        expanded_specs = broadcast_specs(
            group_specs,
            self.candidate_group_size,
            self.accelerator,
        )
        expanded_pocket_entries = broadcast_object(expanded_pocket_entries, self.accelerator)
        if len(expanded_specs) != self.global_candidate_count:
            raise ValueError(
                f'Expected {self.global_candidate_count} candidate specs, '
                f'got {len(expanded_specs)}'
            )
        if len(expanded_pocket_entries) != self.global_candidate_count:
            raise ValueError(
                f'Expected {self.global_candidate_count} expanded pocket entries, '
                f'got {len(expanded_pocket_entries)}'
            )

        candidate_local_start = (
            self.accelerator.process_index * self.local_candidate_count
        )
        candidate_local_end = candidate_local_start + self.local_candidate_count
        local_specs = expanded_specs[candidate_local_start:candidate_local_end]
        local_pocket_entries = expanded_pocket_entries[
            candidate_local_start:candidate_local_end
        ]
        local_pocket_coords = [entry['pocket_coords'] for entry in local_pocket_entries]
        local_pocket_raw_embeddings, local_pocket_mask = self.policy.get_pocket_raw_embeddings(local_pocket_coords)

        rollout_seed = cycle_seed + self.accelerator.process_index * 1000
        rollout = self.policy.rollout_specs(
            specs=local_specs,
            pocket_raw_embeddings=local_pocket_raw_embeddings,
            pocket_mask=local_pocket_mask,
            generation_batch_size=self.config.generation_batch_size,
            seed=rollout_seed,
        )
        (
            rollout,
            local_specs,
            local_pocket_entries,
            local_pocket_raw_embeddings,
            local_pocket_mask,
            local_active_mask,
            selection_metrics,
        ) = self._select_diverse_rollout(
            rollout,
            local_specs,
            local_pocket_entries,
            local_pocket_raw_embeddings,
            local_pocket_mask,
            seed=cycle_seed + self.accelerator.process_index * 100000 + 991,
        )

        reward_records = self._score_selected_rollout(
            rollout,
            local_pocket_entries,
            local_active_mask,
        )
        local_rewards = torch.tensor([record.reward for record in reward_records], device=self.device, dtype=torch.float32)
        global_rewards = self.accelerator.gather(local_rewards).detach()
        global_active_mask = self.accelerator.gather(local_active_mask).detach()
        local_start = self.accelerator.process_index * self.local_sample_count
        local_end = local_start + self.local_sample_count
        hbd_metrics = {}
        global_group_rewards = None
        global_group_reward_credits = None
        global_group_mean_individual_rewards = None
        if (
            self.config.rl_algorithm == 'coupled_sgrpo'
            or self.config.diversity_regularizer_weight > 0.0
            or self._hbd_memory is not None
        ):
            global_smiles = self._all_gather_objects([record.smiles for record in reward_records])
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
            group_diversities = []
            group_diversity_credits = []
            for group_start in range(0, len(global_smiles), self.config.num_generations):
                group_smiles = global_smiles[group_start:group_start + self.config.num_generations]
                group_diversities.append(
                    compute_internal_diversity(group_smiles)
                )
                if self.config.group_rewrad_credit == 'loo':
                    group_diversity_credits.extend(compute_internal_diversity_loo_credits(group_smiles))
            global_group_rewards = torch.tensor(group_diversities, device=self.device, dtype=torch.float32)
            if self.config.group_rewrad_credit == 'loo':
                global_group_reward_credits = torch.tensor(
                    group_diversity_credits,
                    device=self.device,
                    dtype=torch.float32,
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

        local_valid = torch.tensor([float(record.is_valid) for record in reward_records], device=self.device)
        local_alert = torch.tensor([float(record.alert_hit) for record in reward_records], device=self.device)
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
        local_drugclip_score = torch.tensor(
            [float('nan') if record.drugclip_score is None else float(record.drugclip_score) for record in reward_records],
            device=self.device,
        )
        local_unidock_score = torch.tensor(
            [float('nan') if record.unidock_score is None else float(record.unidock_score) for record in reward_records],
            device=self.device,
        )
        local_soft = torch.tensor(
            [float('nan') if record.soft_reward is None else float(record.soft_reward) for record in reward_records],
            device=self.device,
        )
        local_lengths = rollout.completion_mask.sum(dim=1).float()

        gathered_valid = self.accelerator.gather(local_valid)
        gathered_alert = self.accelerator.gather(local_alert)
        gathered_qed = self.accelerator.gather(local_qed)
        gathered_sa = self.accelerator.gather(local_sa)
        gathered_sa_score = self.accelerator.gather(local_sa_score)
        gathered_drugclip_score = self.accelerator.gather(local_drugclip_score)
        gathered_unidock_score = self.accelerator.gather(local_unidock_score)
        gathered_soft = self.accelerator.gather(local_soft)
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
                pocket_raw_embeddings=local_pocket_raw_embeddings,
                pocket_mask=local_pocket_mask,
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
                pocket_raw_embeddings=local_pocket_raw_embeddings,
                pocket_mask=local_pocket_mask,
            )

        log_rows = []
        for spec, pocket_entry, safe_string, record, final_reward, is_active in zip(
            local_specs,
            local_pocket_entries,
            rollout.safe_strings,
            reward_records,
            local_final_rewards.tolist(),
            local_active_mask.tolist(),
        ):
            if not is_active:
                continue
            log_rows.append(
                {
                    'mode': mode,
                    'buffer_cycle': self.generation_cycle_idx,
                    'step': self.global_step,
                    'spec': spec.__dict__,
                    'source_index': int(pocket_entry['source_index']),
                    'pocket_residue_count': int(pocket_entry['residue_count']),
                    'safe': safe_string,
                    'smiles': record.smiles,
                    'reward': float(final_reward),
                    'qed': record.qed,
                    'sa': record.sa,
                    'sa_score': record.sa_score,
                    'drugclip_score': record.drugclip_score,
                    'unidock_score': record.unidock_score,
                    'soft_reward': record.soft_reward,
                    'is_valid': record.is_valid,
                    'alert_hit': record.alert_hit,
                }
            )

        active_global_rewards = global_rewards[global_active_mask]
        active_global_reward_std = global_reward_std[global_active_mask]
        active_gathered_advantages = gathered_advantages[global_active_mask]
        active_gathered_lengths = gathered_lengths[global_active_mask]
        active_gathered_valid = gathered_valid[global_active_mask]
        active_gathered_alert = gathered_alert[global_active_mask]
        metadata = {
            'buffer_cycle': self.generation_cycle_idx,
            'reward_mean': active_global_rewards.mean().item(),
            'reward_std': active_global_reward_std.mean().item(),
            'advantage_mean': active_gathered_advantages.mean().item(),
            'zero_std_ratio': zero_std_ratio,
            'completion_length': active_gathered_lengths.mean().item(),
            'valid_fraction': active_gathered_valid.mean().item(),
            'alert_hit_fraction': active_gathered_alert.mean().item(),
            'invalid_fraction': 1.0 - active_gathered_valid.mean().item(),
            'rewards/qed_mean': _nanmean(gathered_qed),
            'rewards/sa_mean': _nanmean(gathered_sa),
            'rewards/sa_score_mean': _nanmean(gathered_sa_score),
            'rewards/drugclip_score_mean': _nanmean(gathered_drugclip_score),
            'rewards/unidock_score_mean': _nanmean(gathered_unidock_score),
            'rewards/soft_mean': _nanmean(gathered_soft),
            'reward_score_sec_total': float(self.reward_model.last_stats['reward_score_sec_total']),
            'reward_base_score_sec': float(self.reward_model.last_stats['reward_base_score_sec']),
            'reward_drugclip_score_sec': float(self.reward_model.last_stats['reward_drugclip_score_sec']),
            'reward_unidock_score_sec': float(self.reward_model.last_stats['reward_unidock_score_sec']),
            'base_valid_count': float(self.reward_model.last_stats['base_valid_count']),
            'base_valid_fraction': float(self.reward_model.last_stats['base_valid_fraction']),
            'drugclip_input_count': float(self.reward_model.last_stats['drugclip_input_count']),
            'drugclip_unique_smiles_count': float(self.reward_model.last_stats['drugclip_unique_smiles_count']),
            'drugclip_unique_pocket_count': float(self.reward_model.last_stats['drugclip_unique_pocket_count']),
            'drugclip_molecule_cache_hit_count': float(self.reward_model.last_stats['drugclip_molecule_cache_hit_count']),
            'drugclip_molecule_cache_miss_count': float(self.reward_model.last_stats['drugclip_molecule_cache_miss_count']),
            'drugclip_unique_smiles_success_count': float(
                self.reward_model.last_stats['drugclip_unique_smiles_success_count']
            ),
            'drugclip_unique_smiles_failure_count': float(
                self.reward_model.last_stats['drugclip_unique_smiles_failure_count']
            ),
            'drugclip_score_success_count': float(self.reward_model.last_stats['drugclip_score_success_count']),
            'drugclip_score_failure_count': float(self.reward_model.last_stats['drugclip_score_failure_count']),
            'drugclip_score_success_fraction': float(self.reward_model.last_stats['drugclip_score_success_fraction']),
            'drugclip_pocket_cache_hit_count': float(self.reward_model.last_stats['drugclip_pocket_cache_hit_count']),
            'drugclip_pocket_cache_miss_count': float(self.reward_model.last_stats['drugclip_pocket_cache_miss_count']),
            'drugclip_molecule_prepare_sec': float(self.reward_model.last_stats['drugclip_molecule_prepare_sec']),
            'drugclip_molecule_encode_sec': float(self.reward_model.last_stats['drugclip_molecule_encode_sec']),
            'drugclip_pocket_prepare_sec': float(self.reward_model.last_stats['drugclip_pocket_prepare_sec']),
            'drugclip_pocket_encode_sec': float(self.reward_model.last_stats['drugclip_pocket_encode_sec']),
            'drugclip_score_dot_sec': float(self.reward_model.last_stats['drugclip_score_dot_sec']),
            'drugclip_fail_smiles_parse_count': float(self.reward_model.last_stats['drugclip_fail_smiles_parse_count']),
            'drugclip_fail_embed_exception_count': float(
                self.reward_model.last_stats['drugclip_fail_embed_exception_count']
            ),
            'drugclip_fail_zero_conformer_count': float(self.reward_model.last_stats['drugclip_fail_zero_conformer_count']),
            'drugclip_fail_empty_atom_list_count': float(self.reward_model.last_stats['drugclip_fail_empty_atom_list_count']),
            'unidock_input_count': float(self.reward_model.last_stats['unidock_input_count']),
            'unidock_unique_smiles_count': float(self.reward_model.last_stats['unidock_unique_smiles_count']),
            'unidock_unique_pocket_count': float(self.reward_model.last_stats['unidock_unique_pocket_count']),
            'unidock_score_success_count': float(self.reward_model.last_stats['unidock_score_success_count']),
            'unidock_score_failure_count': float(self.reward_model.last_stats['unidock_score_failure_count']),
            'unidock_score_success_fraction': float(self.reward_model.last_stats['unidock_score_success_fraction']),
            'unidock_cache_hit_count': float(self.reward_model.last_stats['unidock_cache_hit_count']),
            'unidock_cache_miss_count': float(self.reward_model.last_stats['unidock_cache_miss_count']),
            'unidock_prepare_sec': float(self.reward_model.last_stats['unidock_prepare_sec']),
            'unidock_dock_sec': float(self.reward_model.last_stats['unidock_dock_sec']),
            'unidock_parse_sec': float(self.reward_model.last_stats['unidock_parse_sec']),
            'unidock_chunk_count': float(self.reward_model.last_stats['unidock_chunk_count']),
            'unidock_rank_cpu_count': float(self.reward_model.last_stats['unidock_rank_cpu_count']),
            'unidock_max_gpu_memory_mb': float(self.reward_model.last_stats['unidock_max_gpu_memory_mb']),
            'unidock_fail_prepare_count': float(self.reward_model.last_stats['unidock_fail_prepare_count']),
            'unidock_fail_output_missing_count': float(
                self.reward_model.last_stats['unidock_fail_output_missing_count']
            ),
            'unidock_fail_parse_count': float(self.reward_model.last_stats['unidock_fail_parse_count']),
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
            'pocket_raw_embeddings': local_pocket_raw_embeddings,
            'pocket_mask': local_pocket_mask,
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
            pocket_raw_embeddings=inputs['pocket_raw_embeddings'],
            pocket_mask=inputs['pocket_mask'],
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
            'rewards/drugclip_score_mean': _reward_metric('rewards/drugclip_score_mean'),
            'rewards/unidock_score_mean': _reward_metric('rewards/unidock_score_mean'),
            'rewards/soft_mean': _reward_metric('rewards/soft_mean'),
            'reward_score_sec_total': _reward_metric('reward_score_sec_total'),
            'reward_base_score_sec': _reward_metric('reward_base_score_sec'),
            'reward_drugclip_score_sec': _reward_metric('reward_drugclip_score_sec'),
            'reward_unidock_score_sec': _reward_metric('reward_unidock_score_sec'),
            'base_valid_count': _reward_metric('base_valid_count'),
            'base_valid_fraction': _reward_metric('base_valid_fraction'),
            'drugclip_input_count': _reward_metric('drugclip_input_count'),
            'drugclip_unique_smiles_count': _reward_metric('drugclip_unique_smiles_count'),
            'drugclip_unique_pocket_count': _reward_metric('drugclip_unique_pocket_count'),
            'drugclip_molecule_cache_hit_count': _reward_metric('drugclip_molecule_cache_hit_count'),
            'drugclip_molecule_cache_miss_count': _reward_metric('drugclip_molecule_cache_miss_count'),
            'drugclip_unique_smiles_success_count': _reward_metric('drugclip_unique_smiles_success_count'),
            'drugclip_unique_smiles_failure_count': _reward_metric('drugclip_unique_smiles_failure_count'),
            'drugclip_score_success_count': _reward_metric('drugclip_score_success_count'),
            'drugclip_score_failure_count': _reward_metric('drugclip_score_failure_count'),
            'drugclip_score_success_fraction': _reward_metric('drugclip_score_success_fraction'),
            'drugclip_pocket_cache_hit_count': _reward_metric('drugclip_pocket_cache_hit_count'),
            'drugclip_pocket_cache_miss_count': _reward_metric('drugclip_pocket_cache_miss_count'),
            'drugclip_molecule_prepare_sec': _reward_metric('drugclip_molecule_prepare_sec'),
            'drugclip_molecule_encode_sec': _reward_metric('drugclip_molecule_encode_sec'),
            'drugclip_pocket_prepare_sec': _reward_metric('drugclip_pocket_prepare_sec'),
            'drugclip_pocket_encode_sec': _reward_metric('drugclip_pocket_encode_sec'),
            'drugclip_score_dot_sec': _reward_metric('drugclip_score_dot_sec'),
            'drugclip_fail_smiles_parse_count': _reward_metric('drugclip_fail_smiles_parse_count'),
            'drugclip_fail_embed_exception_count': _reward_metric('drugclip_fail_embed_exception_count'),
            'drugclip_fail_zero_conformer_count': _reward_metric('drugclip_fail_zero_conformer_count'),
            'drugclip_fail_empty_atom_list_count': _reward_metric('drugclip_fail_empty_atom_list_count'),
            'unidock_input_count': _reward_metric('unidock_input_count'),
            'unidock_unique_smiles_count': _reward_metric('unidock_unique_smiles_count'),
            'unidock_unique_pocket_count': _reward_metric('unidock_unique_pocket_count'),
            'unidock_score_success_count': _reward_metric('unidock_score_success_count'),
            'unidock_score_failure_count': _reward_metric('unidock_score_failure_count'),
            'unidock_score_success_fraction': _reward_metric('unidock_score_success_fraction'),
            'unidock_cache_hit_count': _reward_metric('unidock_cache_hit_count'),
            'unidock_cache_miss_count': _reward_metric('unidock_cache_miss_count'),
            'unidock_prepare_sec': _reward_metric('unidock_prepare_sec'),
            'unidock_dock_sec': _reward_metric('unidock_dock_sec'),
            'unidock_parse_sec': _reward_metric('unidock_parse_sec'),
            'unidock_chunk_count': _reward_metric('unidock_chunk_count'),
            'unidock_rank_cpu_count': _reward_metric('unidock_rank_cpu_count'),
            'unidock_max_gpu_memory_mb': _reward_metric('unidock_max_gpu_memory_mb'),
            'unidock_fail_prepare_count': _reward_metric('unidock_fail_prepare_count'),
            'unidock_fail_output_missing_count': _reward_metric('unidock_fail_output_missing_count'),
            'unidock_fail_parse_count': _reward_metric('unidock_fail_parse_count'),
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
                'rewards/qed_mean=%.6f rewards/sa_mean=%.6f rewards/sa_score_mean=%.6f '
                'rewards/drugclip_score_mean=%.6f rewards/unidock_score_mean=%.6f rewards/soft_mean=%.6f '
                'reward_score_sec_total=%.6f reward_base_score_sec=%.6f reward_drugclip_score_sec=%.6f '
                'reward_unidock_score_sec=%.6f '
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
                metrics['rewards/drugclip_score_mean'],
                metrics['rewards/unidock_score_mean'],
                metrics['rewards/soft_mean'],
                metrics['reward_score_sec_total'],
                metrics['reward_base_score_sec'],
                metrics['reward_drugclip_score_sec'],
                metrics['reward_unidock_score_sec'],
                metrics['grad_norm'],
                metrics['lr'],
                (
                    ''
                    if metrics['drugclip_input_count'] <= 0.0
                    else (
                        f" base_valid_count={metrics['base_valid_count']:.0f}"
                        f" base_valid_fraction={metrics['base_valid_fraction']:.6f}"
                        f" drugclip_input_count={metrics['drugclip_input_count']:.0f}"
                        f" drugclip_unique_smiles_count={metrics['drugclip_unique_smiles_count']:.0f}"
                        f" drugclip_unique_pocket_count={metrics['drugclip_unique_pocket_count']:.0f}"
                        f" drugclip_molecule_cache_hit_count={metrics['drugclip_molecule_cache_hit_count']:.0f}"
                        f" drugclip_molecule_cache_miss_count={metrics['drugclip_molecule_cache_miss_count']:.0f}"
                        f" drugclip_pocket_cache_hit_count={metrics['drugclip_pocket_cache_hit_count']:.0f}"
                        f" drugclip_pocket_cache_miss_count={metrics['drugclip_pocket_cache_miss_count']:.0f}"
                        f" drugclip_unique_smiles_success_count={metrics['drugclip_unique_smiles_success_count']:.0f}"
                        f" drugclip_unique_smiles_failure_count={metrics['drugclip_unique_smiles_failure_count']:.0f}"
                        f" drugclip_score_success_count={metrics['drugclip_score_success_count']:.0f}"
                        f" drugclip_score_failure_count={metrics['drugclip_score_failure_count']:.0f}"
                        f" drugclip_score_success_fraction={metrics['drugclip_score_success_fraction']:.6f}"
                        f" drugclip_molecule_prepare_sec={metrics['drugclip_molecule_prepare_sec']:.6f}"
                        f" drugclip_molecule_encode_sec={metrics['drugclip_molecule_encode_sec']:.6f}"
                        f" drugclip_pocket_prepare_sec={metrics['drugclip_pocket_prepare_sec']:.6f}"
                        f" drugclip_pocket_encode_sec={metrics['drugclip_pocket_encode_sec']:.6f}"
                        f" drugclip_score_dot_sec={metrics['drugclip_score_dot_sec']:.6f}"
                        f" drugclip_fail_smiles_parse_count={metrics['drugclip_fail_smiles_parse_count']:.0f}"
                        f" drugclip_fail_embed_exception_count={metrics['drugclip_fail_embed_exception_count']:.0f}"
                        f" drugclip_fail_zero_conformer_count={metrics['drugclip_fail_zero_conformer_count']:.0f}"
                        f" drugclip_fail_empty_atom_list_count={metrics['drugclip_fail_empty_atom_list_count']:.0f}"
                    )
                )
                + (
                    ''
                    if metrics['unidock_input_count'] <= 0.0
                    else (
                        f" unidock_input_count={metrics['unidock_input_count']:.0f}"
                        f" unidock_unique_smiles_count={metrics['unidock_unique_smiles_count']:.0f}"
                        f" unidock_unique_pocket_count={metrics['unidock_unique_pocket_count']:.0f}"
                        f" unidock_score_success_count={metrics['unidock_score_success_count']:.0f}"
                        f" unidock_score_failure_count={metrics['unidock_score_failure_count']:.0f}"
                        f" unidock_score_success_fraction={metrics['unidock_score_success_fraction']:.6f}"
                        f" unidock_cache_hit_count={metrics['unidock_cache_hit_count']:.0f}"
                        f" unidock_cache_miss_count={metrics['unidock_cache_miss_count']:.0f}"
                        f" unidock_prepare_sec={metrics['unidock_prepare_sec']:.6f}"
                        f" unidock_dock_sec={metrics['unidock_dock_sec']:.6f}"
                        f" unidock_parse_sec={metrics['unidock_parse_sec']:.6f}"
                        f" unidock_chunk_count={metrics['unidock_chunk_count']:.0f}"
                        f" unidock_rank_cpu_count={metrics['unidock_rank_cpu_count']:.0f}"
                        f" unidock_max_gpu_memory_mb={metrics['unidock_max_gpu_memory_mb']:.0f}"
                        f" unidock_fail_prepare_count={metrics['unidock_fail_prepare_count']:.0f}"
                        f" unidock_fail_output_missing_count={metrics['unidock_fail_output_missing_count']:.0f}"
                        f" unidock_fail_parse_count={metrics['unidock_fail_parse_count']:.0f}"
                    )
                )
                + (
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
        ref_path = os.path.join(checkpoint_dir, 'reference_trainable.pt')
        trainer_state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
        self.policy.save_checkpoint(model_path, step=self.global_step, accelerator=self.accelerator)
        torch.save(self.reference.get_trainable_state_dict(), ref_path)
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
        ref_path = os.path.join(checkpoint_dir, 'reference_trainable.pt')
        ensure_exists(accelerator_state_dir, 'accelerator state')
        ensure_exists(trainer_state_path, 'trainer state')
        ensure_exists(model_path, 'model checkpoint')
        ensure_exists(ref_path, 'reference state')

        with open(trainer_state_path) as handle:
            trainer_state = json.load(handle)

        self.accelerator.load_state(accelerator_state_dir)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.policy.load_ema_state(checkpoint.get('ema'))
        self.reference.load_trainable_state_dict(torch.load(ref_path, map_location='cpu', weights_only=False))

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
                self.policy.trainable_parameters(),
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
        self._metrics['eval'] = defaultdict(list)
        self._textual_logs['eval'] = []
        with self.policy.eval_mode():
            inputs, metadata = self._generate_and_score_completions(mode='eval')
            for iteration_idx in range(self.config.num_iterations):
                self._compute_loss(
                    inputs,
                    mode='eval',
                    iteration_idx=iteration_idx,
                    requires_grad=False,
                )
            self._record_optimizer_metrics('eval', grad_norm=float('nan'), lr=self.scheduler.get_last_lr()[0])
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
