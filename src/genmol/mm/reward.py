from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import torch

from genmol.diversity import (
    compute_internal_diversity,
    compute_internal_diversity_loo_credits,
)
from genmol.mm.utils import DrugCLIPConfig, DrugCLIPScorer, UniDockConfig, UniDockScorer


_PROCESS_KERNEL = None
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


def apply_reward_gate(soft_reward, is_valid, alert_hit):
    if not is_valid:
        return -1.0
    if alert_hit:
        return 0.2 * float(soft_reward)
    return float(soft_reward)


@dataclass(frozen=True)
class RewardRecord:
    reward: float
    is_valid: bool
    alert_hit: bool
    qed: float | None
    sa: float | None
    sa_score: float | None
    drugclip_score: float | None
    unidock_score: float | None
    soft_reward: float | None
    smiles: str | None


class _AlertFilter:
    def __init__(self):
        from rd_filters.rd_filters import RDFilters, read_rules
        import pandas as pd
        import pkg_resources

        alert_file_name = pkg_resources.resource_filename(
            'rd_filters',
            'data/alert_collection.csv',
        )
        rules_file_path = pkg_resources.resource_filename(
            'rd_filters',
            'data/rules.json',
        )
        self._rf = RDFilters(alert_file_name)
        self._pd = pd
        rule_dict = read_rules(rules_file_path)
        rule_dict['Rule_Inpharmatica'] = False
        rule_dict['Rule_PAINS'] = True
        rule_dict['Rule_SureChEMBL'] = True
        rule_dict['Rule_Glaxo'] = True
        for key in ['HBA', 'HBD', 'LogP', 'MW', 'Rot', 'TPSA']:
            rule_dict.pop(key, None)
        rule_list = [
            key.replace('Rule_', '')
            for key, enabled in rule_dict.items()
            if key.startswith('Rule_') and enabled
        ]
        self._rf.build_rule_list(rule_list)

    def __call__(self, input_data):
        if isinstance(input_data, str):
            input_data = [input_data]
        elif not isinstance(input_data, list):
            raise ValueError('Input must be a list of SMILES or one SMILES string')

        indexed_smiles = list(zip(input_data, list(range(len(input_data)))))
        results = [self._rf.evaluate(item) for item in indexed_smiles]
        frame = self._pd.DataFrame(
            results,
            columns=[
                'SMILES',
                'NAME',
                'FILTER',
                'MW',
                'LogP',
                'HBD',
                'HBA',
                'TPSA',
                'Rot',
            ],
        )
        return frame[frame.FILTER == 'OK'].SMILES.values


def _resolve_reward_workers():
    raw_workers = os.environ.get('GENMOL_REWARD_WORKERS')
    if raw_workers is not None:
        try:
            workers = int(raw_workers)
        except ValueError as exc:
            raise ValueError(f'GENMOL_REWARD_WORKERS must be an integer, got: {raw_workers}') from exc
        if workers <= 0:
            raise ValueError(f'GENMOL_REWARD_WORKERS must be positive, got: {workers}')
        return workers

    raw_total_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if raw_total_cpus is None:
        total_cpus = os.cpu_count() or 1
    else:
        try:
            total_cpus = int(raw_total_cpus)
        except ValueError as exc:
            raise ValueError(f'SLURM_CPUS_PER_TASK must be an integer, got: {raw_total_cpus}') from exc
    if total_cpus <= 0:
        raise ValueError(f'Invalid CPU count for reward workers: {total_cpus}')

    raw_local_world_size = os.environ.get('LOCAL_WORLD_SIZE') or os.environ.get('WORLD_SIZE') or '1'
    try:
        local_world_size = int(raw_local_world_size)
    except ValueError as exc:
        raise ValueError(f'LOCAL_WORLD_SIZE/WORLD_SIZE must be an integer, got: {raw_local_world_size}') from exc
    if local_world_size <= 0:
        raise ValueError(f'Invalid local world size for reward workers: {local_world_size}')

    return max(1, (total_cpus // local_world_size) - 1)


class _RewardKernel:
    def __init__(self, reward_weights=None, always_compute_metrics=False):
        from rdkit import Chem
        from rdkit.Chem import QED
        from tdc import Oracle

        self.reward_weights = normalize_molecular_reward_weights(reward_weights)
        self.always_compute_metrics = bool(always_compute_metrics)
        self.compute_qed = self.always_compute_metrics or self.reward_weights['qed'] > 0.0
        self.compute_sa = self.always_compute_metrics or self.reward_weights['sa_score'] > 0.0
        self.require_qed_for_reward = self.reward_weights['qed'] > 0.0
        self.require_sa_for_reward = self.reward_weights['sa_score'] > 0.0
        self._chem = Chem
        self._qed = QED if self.compute_qed else None
        self._sa_oracle = Oracle('sa') if self.compute_sa else None
        self._filter = _AlertFilter()

    def _safe_sa_score(self, smiles):
        if self._sa_oracle is None:
            return None
        try:
            return float(self._sa_oracle([smiles])[0])
        except Exception:
            return None

    def _safe_qed_score(self, mol):
        if self._qed is None:
            return None
        try:
            return float(self._qed.qed(mol))
        except Exception:
            return None

    def _canonicalize(self, smiles):
        if smiles is None:
            return None, None
        try:
            mol = self._chem.MolFromSmiles(smiles, sanitize=True)
        except Exception:
            return None, None
        if mol is None:
            return None, None
        try:
            canonical = self._chem.MolToSmiles(mol)
        except Exception:
            return None, None
        return canonical, mol

    def score_chunk(self, smiles_list):
        canonical_smiles = []
        mols = []
        valid_indices = []
        records = [None] * len(smiles_list)

        for idx, smiles in enumerate(smiles_list):
            canonical, mol = self._canonicalize(smiles)
            if canonical is None or mol is None:
                records[idx] = RewardRecord(
                    reward=-1.0,
                    is_valid=False,
                    alert_hit=False,
                    qed=None,
                    sa=None,
                    sa_score=None,
                    drugclip_score=None,
                    unidock_score=None,
                    soft_reward=None,
                    smiles=None,
                )
                continue
            canonical_smiles.append(canonical)
            mols.append(mol)
            valid_indices.append(idx)

        if valid_indices:
            pass_smiles = set(self._filter(canonical_smiles))
            if self.compute_qed:
                qed_scores = [self._safe_qed_score(mol) for mol in mols]
            else:
                qed_scores = [None] * len(mols)
            if self.compute_sa:
                sa_scores = [self._safe_sa_score(smiles) for smiles in canonical_smiles]
            else:
                sa_scores = [None] * len(canonical_smiles)

            for index, smiles, qed_score, sa_score in zip(valid_indices, canonical_smiles, qed_scores, sa_scores):
                sa_score_value = None if sa_score is None else sa_to_score(sa_score)
                if self.require_qed_for_reward and qed_score is None:
                    records[index] = RewardRecord(
                        reward=-1.0,
                        is_valid=False,
                        alert_hit=False,
                        qed=None,
                        sa=sa_score,
                        sa_score=sa_score_value,
                        drugclip_score=None,
                        unidock_score=None,
                        soft_reward=None,
                        smiles=smiles,
                    )
                    continue
                if self.require_sa_for_reward and sa_score is None:
                    records[index] = RewardRecord(
                        reward=-1.0,
                        is_valid=False,
                        alert_hit=False,
                        qed=qed_score,
                        sa=None,
                        sa_score=None,
                        drugclip_score=None,
                        unidock_score=None,
                        soft_reward=None,
                        smiles=smiles,
                    )
                    continue
                alert_hit = smiles not in pass_smiles
                soft_reward = compute_soft_reward(
                    qed_score,
                    sa_score_value,
                    reward_weights=self.reward_weights,
                )
                records[index] = RewardRecord(
                    reward=apply_reward_gate(soft_reward, is_valid=True, alert_hit=alert_hit),
                    is_valid=True,
                    alert_hit=alert_hit,
                    qed=qed_score,
                    sa=sa_score,
                    sa_score=sa_score_value,
                    drugclip_score=None,
                    unidock_score=None,
                    soft_reward=soft_reward,
                    smiles=smiles,
                )

        return records


def _initialize_process_reward_kernel(reward_weights, always_compute_metrics):
    global _PROCESS_KERNEL
    _PROCESS_KERNEL = _RewardKernel(
        reward_weights=reward_weights,
        always_compute_metrics=always_compute_metrics,
    )


def _score_reward_chunk(smiles_chunk):
    if _PROCESS_KERNEL is None:
        raise RuntimeError('Reward worker kernel is not initialized')
    return _PROCESS_KERNEL.score_chunk(smiles_chunk)


class MolecularReward:
    def __init__(
        self,
        reward_weights=None,
        always_compute_metrics=False,
        drugclip_config: DrugCLIPConfig | None = None,
        unidock_config: UniDockConfig | None = None,
    ):
        self.num_workers = _resolve_reward_workers()
        self.reward_weights = normalize_molecular_reward_weights(reward_weights)
        self.always_compute_metrics = bool(always_compute_metrics)
        self._drugclip = None
        if self.reward_weights['drugclip_score'] > 0.0:
            if drugclip_config is None:
                raise ValueError('drugclip_config is required when drugclip_score reward weight is positive')
            self._drugclip = DrugCLIPScorer(drugclip_config)
        self._unidock = None
        if self.reward_weights['unidock_score'] > 0.0:
            if unidock_config is None:
                raise ValueError('unidock_config is required when unidock_score reward weight is positive')
            self._unidock = UniDockScorer(unidock_config)
        self._base_reward_weights = dict(self.reward_weights)
        self._base_reward_weights['drugclip_score'] = 0.0
        self._base_reward_weights['unidock_score'] = 0.0
        self._kernel = _RewardKernel(
            reward_weights=self._base_reward_weights,
            always_compute_metrics=self.always_compute_metrics,
        )
        self._pool = None
        if self.num_workers > 1:
            self._pool = ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=mp.get_context('spawn'),
                initializer=_initialize_process_reward_kernel,
                initargs=(self._base_reward_weights, self.always_compute_metrics),
            )
        self.last_stats = self._empty_last_stats()

    @staticmethod
    def _empty_last_stats():
        return {
            'reward_score_sec_total': 0.0,
            'reward_base_score_sec': 0.0,
            'reward_drugclip_score_sec': 0.0,
            'reward_unidock_score_sec': 0.0,
            'base_valid_count': 0,
            'base_valid_fraction': 0.0,
            'drugclip_input_count': 0,
            'drugclip_unique_smiles_count': 0,
            'drugclip_unique_pocket_count': 0,
            'drugclip_molecule_cache_hit_count': 0,
            'drugclip_molecule_cache_miss_count': 0,
            'drugclip_unique_smiles_success_count': 0,
            'drugclip_unique_smiles_failure_count': 0,
            'drugclip_score_success_count': 0,
            'drugclip_score_failure_count': 0,
            'drugclip_score_success_fraction': 0.0,
            'drugclip_pocket_cache_hit_count': 0,
            'drugclip_pocket_cache_miss_count': 0,
            'drugclip_molecule_prepare_sec': 0.0,
            'drugclip_molecule_encode_sec': 0.0,
            'drugclip_pocket_prepare_sec': 0.0,
            'drugclip_pocket_encode_sec': 0.0,
            'drugclip_score_dot_sec': 0.0,
            'drugclip_fail_smiles_parse_count': 0,
            'drugclip_fail_embed_exception_count': 0,
            'drugclip_fail_zero_conformer_count': 0,
            'drugclip_fail_empty_atom_list_count': 0,
            'unidock_input_count': 0,
            'unidock_unique_smiles_count': 0,
            'unidock_unique_pocket_count': 0,
            'unidock_score_success_count': 0,
            'unidock_score_failure_count': 0,
            'unidock_score_success_fraction': 0.0,
            'unidock_cache_hit_count': 0,
            'unidock_cache_miss_count': 0,
            'unidock_prepare_sec': 0.0,
            'unidock_dock_sec': 0.0,
            'unidock_parse_sec': 0.0,
            'unidock_chunk_count': 0,
            'unidock_rank_cpu_count': 0,
            'unidock_max_gpu_memory_mb': 0,
            'unidock_fail_prepare_count': 0,
            'unidock_fail_output_missing_count': 0,
            'unidock_fail_parse_count': 0,
        }

    def _reset_last_stats(self):
        self.last_stats = self._empty_last_stats()

    @staticmethod
    def _synchronize_cuda(device):
        if device is None:
            return
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=False)
            self._pool = None
        if self._drugclip is not None:
            self._drugclip.close()
            self._drugclip = None
        if self._unidock is not None:
            self._unidock.close()
            self._unidock = None

    def score(self, smiles_list, pocket_entries=None):
        self._reset_last_stats()
        if not smiles_list:
            return []
        total_start = time.perf_counter()
        base_start = time.perf_counter()
        if self._pool is None or len(smiles_list) == 1:
            records = self._kernel.score_chunk(smiles_list)
        else:
            chunk_count = min(self.num_workers, len(smiles_list))
            chunk_size = math.ceil(len(smiles_list) / chunk_count)
            futures = []
            for start in range(0, len(smiles_list), chunk_size):
                futures.append(self._pool.submit(_score_reward_chunk, smiles_list[start:start + chunk_size]))

            records = []
            for future in futures:
                records.extend(future.result())

            if len(records) != len(smiles_list):
                raise RuntimeError(
                    f'Reward worker returned mismatched record count: expected {len(smiles_list)}, got {len(records)}'
                )
        self.last_stats['reward_base_score_sec'] = time.perf_counter() - base_start

        if self._drugclip is None and self._unidock is None:
            self.last_stats['reward_score_sec_total'] = time.perf_counter() - total_start
            return records

        if pocket_entries is None:
            raise ValueError('pocket_entries are required when molecular docking-style reward weights are positive')
        if len(pocket_entries) != len(smiles_list):
            raise ValueError(
                'pocket_entries must match smiles_list length when docking-style reward scoring is enabled: '
                f'{len(pocket_entries)} vs {len(smiles_list)}'
            )

        active_indices = []
        active_smiles = []
        active_pocket_entries = []
        for index, record in enumerate(records):
            if not record.is_valid or record.smiles is None:
                continue
            active_indices.append(index)
            active_smiles.append(record.smiles)
            active_pocket_entries.append(pocket_entries[index])
        self.last_stats['base_valid_count'] = int(len(active_indices))
        self.last_stats['base_valid_fraction'] = float(len(active_indices) / len(records))

        if not active_indices:
            self.last_stats['reward_score_sec_total'] = time.perf_counter() - total_start
            return records

        updated_records = list(records)
        if self._drugclip is not None:
            self._synchronize_cuda(self._drugclip.device)
            drugclip_start = time.perf_counter()
            drugclip_scores = self._drugclip.score(active_smiles, active_pocket_entries)
            self._synchronize_cuda(self._drugclip.device)
            self.last_stats['reward_drugclip_score_sec'] = time.perf_counter() - drugclip_start
            self.last_stats.update(self._drugclip.last_score_stats)
            if len(drugclip_scores) != len(active_indices):
                raise RuntimeError(
                    'DrugCLIP scorer returned mismatched score count: '
                    f'expected {len(active_indices)}, got {len(drugclip_scores)}'
                )
            for active_index, drugclip_score in zip(active_indices, drugclip_scores):
                record = updated_records[active_index]
                if drugclip_score is None:
                    updated_records[active_index] = replace(
                        record,
                        reward=-1.0,
                        is_valid=False,
                        drugclip_score=None,
                        soft_reward=None,
                    )
                    continue
                updated_records[active_index] = replace(
                    record,
                    drugclip_score=float(drugclip_score),
                )

        if self._unidock is not None:
            unidock_start = time.perf_counter()
            unidock_scores = self._unidock.score(active_smiles, active_pocket_entries)
            self.last_stats['reward_unidock_score_sec'] = time.perf_counter() - unidock_start
            self.last_stats.update(self._unidock.last_score_stats)
            if len(unidock_scores) != len(active_indices):
                raise RuntimeError(
                    'Uni-Dock scorer returned mismatched score count: '
                    f'expected {len(active_indices)}, got {len(unidock_scores)}'
                )
            for active_index, unidock_score in zip(active_indices, unidock_scores):
                record = updated_records[active_index]
                if not record.is_valid:
                    continue
                if unidock_score is None:
                    updated_records[active_index] = replace(
                        record,
                        reward=-1.0,
                        is_valid=False,
                        unidock_score=None,
                        soft_reward=None,
                    )
                    continue
                updated_records[active_index] = replace(
                    record,
                    unidock_score=unidock_affinity_to_score(unidock_score),
                )

        for active_index in active_indices:
            record = updated_records[active_index]
            if not record.is_valid:
                continue
            soft_reward = compute_soft_reward(
                record.qed,
                record.sa_score,
                record.drugclip_score,
                record.unidock_score,
                reward_weights=self.reward_weights,
            )
            updated_records[active_index] = replace(
                record,
                reward=apply_reward_gate(soft_reward, is_valid=True, alert_hit=record.alert_hit),
                soft_reward=soft_reward,
            )
        self.last_stats['reward_score_sec_total'] = time.perf_counter() - total_start
        return updated_records
