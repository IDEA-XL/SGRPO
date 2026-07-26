import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass


_PROCESS_KERNEL = None
MOLECULAR_REWARD_NAME_ORDER = (
    'qed',
    'sa_score',
)
DEFAULT_MOLECULAR_REWARD_WEIGHTS = {
    'qed': 0.6,
    'sa_score': 0.4,
}


def sa_to_score(sa_value):
    return max(0.0, min((6.0 - float(sa_value)) / 5.0, 1.0))


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


def compute_soft_reward(qed_value, sa_score_value, reward_weights=None):
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
    return float(outputs)


def apply_reward_gate(soft_reward, is_valid, alert_hit):
    if not is_valid:
        return -1.0
    if alert_hit:
        return 0.2 * float(soft_reward)
    return float(soft_reward)


def combine_candidate_diversity_reward(
    base_soft_reward,
    raw_loo_credit,
    *,
    weight,
    is_valid,
    alert_hit,
):
    weight = float(weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(
            f'candidate diversity reward weight must be in [0, 1], got {weight}'
        )
    if not is_valid:
        return -1.0
    if base_soft_reward is None:
        raise ValueError(
            'base_soft_reward is required for a valid candidate'
        )
    combined_reward = (
        (1.0 - weight) * float(base_soft_reward)
        + weight * float(raw_loo_credit)
    )
    return apply_reward_gate(
        combined_reward,
        is_valid=True,
        alert_hit=alert_hit,
    )


def _compute_internal_diversity_statistics(smiles_list):
    indexed_fingerprints = _compute_indexed_fingerprints(smiles_list)
    if len(indexed_fingerprints) < 2:
        return 0.0, [0.0 for _ in smiles_list]

    similarity_sum, pair_count, per_fingerprint_similarity_sum = (
        _compute_pairwise_similarity_stats(indexed_fingerprints)
    )
    if pair_count == 0:
        return 0.0, [0.0 for _ in smiles_list]

    full_diversity = 1.0 - (similarity_sum / pair_count)
    credits = [0.0 for _ in smiles_list]
    valid_count = len(indexed_fingerprints)
    reduced_count = valid_count - 1
    reduced_pair_count = reduced_count * (reduced_count - 1) // 2

    for fingerprint_idx, (original_idx, _) in enumerate(indexed_fingerprints):
        if reduced_pair_count == 0:
            reduced_diversity = 0.0
        else:
            reduced_similarity_sum = (
                similarity_sum
                - per_fingerprint_similarity_sum[fingerprint_idx]
            )
            reduced_diversity = (
                1.0 - (reduced_similarity_sum / reduced_pair_count)
            )
        credits[original_idx] = full_diversity - reduced_diversity
    return full_diversity, credits


def compute_internal_diversity(smiles_list):
    diversity, _ = _compute_internal_diversity_statistics(smiles_list)
    return diversity


def compute_internal_diversity_with_loo_credits(smiles_list):
    if len(smiles_list) < 2:
        raise ValueError('LOO diversity credit requires at least two rollouts')
    return _compute_internal_diversity_statistics(smiles_list)


def _compute_indexed_fingerprints(smiles_list):
    from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator

    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    indexed_fingerprints = []
    for original_idx, smiles in enumerate(smiles_list):
        if smiles is None:
            continue
        mol = MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            continue
        indexed_fingerprints.append((original_idx, fingerprint_generator.GetFingerprint(mol)))
    return indexed_fingerprints


def _compute_pairwise_similarity_stats(indexed_fingerprints):
    import numpy as np
    from rdkit import DataStructs

    fingerprints = [fingerprint for _, fingerprint in indexed_fingerprints]
    similarity_sum = 0.0
    pair_count = 0
    per_fingerprint_similarity_sum = np.zeros(
        len(fingerprints),
        dtype=np.float64,
    )
    for left_idx in range(len(fingerprints) - 1):
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(
                fingerprints[left_idx],
                fingerprints[left_idx + 1:],
            ),
            dtype=np.float64,
        )
        row_similarity_sum = float(similarities.sum())
        similarity_sum += row_similarity_sum
        per_fingerprint_similarity_sum[left_idx] += row_similarity_sum
        per_fingerprint_similarity_sum[left_idx + 1:] += similarities
        pair_count += int(similarities.size)
    return (
        similarity_sum,
        pair_count,
        per_fingerprint_similarity_sum.tolist(),
    )


def compute_internal_diversity_loo_credits(smiles_list):
    _, credits = compute_internal_diversity_with_loo_credits(smiles_list)
    return credits


@dataclass(frozen=True)
class RewardRecord:
    reward: float
    is_valid: bool
    alert_hit: bool
    qed: float | None
    sa: float | None
    sa_score: float | None
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

    # Leave one CPU core for the trainer process when possible.
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
    def __init__(self, reward_weights=None, always_compute_metrics=False):
        self.num_workers = _resolve_reward_workers()
        self.reward_weights = normalize_molecular_reward_weights(reward_weights)
        self.always_compute_metrics = bool(always_compute_metrics)
        self._kernel = _RewardKernel(
            reward_weights=self.reward_weights,
            always_compute_metrics=self.always_compute_metrics,
        )
        self._pool = None
        if self.num_workers > 1:
            self._pool = ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=mp.get_context('spawn'),
                initializer=_initialize_process_reward_kernel,
                initargs=(self.reward_weights, self.always_compute_metrics),
            )

    def close(self):
        if self._pool is None:
            return
        self._pool.shutdown(wait=True, cancel_futures=False)
        self._pool = None

    def score(self, smiles_list):
        if not smiles_list:
            return []
        if self._pool is None or len(smiles_list) == 1:
            return self._kernel.score_chunk(smiles_list)

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
        return records
