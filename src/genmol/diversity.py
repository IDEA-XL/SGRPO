from collections import Counter
from collections.abc import Sequence


MORGAN_INTERNAL_DIVERSITY = 'morgan_internal_diversity'
RELATIVE_SCAFFOLD_DIVERSITY = 'relative_scaffold_diversity'
DEFAULT_DIVERSITY_METRIC = MORGAN_INTERNAL_DIVERSITY
VALID_DIVERSITY_METRICS = frozenset(
    {
        MORGAN_INTERNAL_DIVERSITY,
        RELATIVE_SCAFFOLD_DIVERSITY,
    }
)


def validate_diversity_metric(metric):
    if not isinstance(metric, str):
        raise TypeError(f'diversity_metric must be a string, got {type(metric).__name__}')
    if metric not in VALID_DIVERSITY_METRICS:
        raise ValueError(
            f'diversity_metric must be one of {sorted(VALID_DIVERSITY_METRICS)}, got {metric!r}'
        )
    return metric


def compute_molecular_diversity(
    smiles_list: Sequence[str | None],
    *,
    metric: str = DEFAULT_DIVERSITY_METRIC,
) -> float:
    metric = validate_diversity_metric(metric)
    if metric == MORGAN_INTERNAL_DIVERSITY:
        return _compute_morgan_internal_diversity(smiles_list)
    return _compute_relative_scaffold_diversity(smiles_list)


def compute_molecular_diversity_with_loo_credits(
    smiles_list: Sequence[str | None],
    *,
    metric: str = DEFAULT_DIVERSITY_METRIC,
) -> tuple[float, list[float]]:
    if len(smiles_list) < 2:
        raise ValueError('LOO diversity credit requires at least two rollouts')

    metric = validate_diversity_metric(metric)
    if metric == MORGAN_INTERNAL_DIVERSITY:
        return _compute_morgan_internal_diversity_with_loo_credits(smiles_list)
    return _compute_relative_scaffold_diversity_with_loo_credits(smiles_list)


def compute_molecular_diversity_loo_credits(
    smiles_list: Sequence[str | None],
    *,
    metric: str = DEFAULT_DIVERSITY_METRIC,
) -> list[float]:
    _, credits = compute_molecular_diversity_with_loo_credits(
        smiles_list,
        metric=metric,
    )
    return credits


def compute_internal_diversity(smiles_list: Sequence[str | None]) -> float:
    return compute_molecular_diversity(
        smiles_list,
        metric=MORGAN_INTERNAL_DIVERSITY,
    )


def compute_internal_diversity_loo_credits(
    smiles_list: Sequence[str | None],
) -> list[float]:
    return compute_molecular_diversity_loo_credits(
        smiles_list,
        metric=MORGAN_INTERNAL_DIVERSITY,
    )


def _compute_morgan_internal_diversity(smiles_list):
    indexed_fingerprints = _compute_indexed_fingerprints(smiles_list)
    if len(indexed_fingerprints) < 2:
        return 0.0
    similarity_sum, pair_count, _ = _compute_pairwise_similarity_stats(indexed_fingerprints)
    if pair_count == 0:
        return 0.0
    return 1.0 - (similarity_sum / pair_count)


def _compute_morgan_internal_diversity_with_loo_credits(smiles_list):
    indexed_fingerprints = _compute_indexed_fingerprints(smiles_list)
    if len(indexed_fingerprints) < 2:
        return 0.0, [0.0 for _ in smiles_list]

    similarity_sum, pair_count, per_fingerprint_similarity_sum = (
        _compute_pairwise_similarity_stats(indexed_fingerprints)
    )
    full_diversity = 1.0 - (similarity_sum / pair_count)
    credits = [0.0 for _ in smiles_list]
    valid_count = len(indexed_fingerprints)
    reduced_count = valid_count - 1
    reduced_pair_count = reduced_count * (reduced_count - 1) // 2

    for fingerprint_idx, (original_idx, _) in enumerate(indexed_fingerprints):
        if reduced_pair_count == 0:
            reduced_diversity = 0.0
        else:
            reduced_similarity_sum = similarity_sum - per_fingerprint_similarity_sum[fingerprint_idx]
            reduced_diversity = 1.0 - (reduced_similarity_sum / reduced_pair_count)
        credits[original_idx] = full_diversity - reduced_diversity
    return full_diversity, credits


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
    from rdkit import DataStructs

    fingerprints = [fingerprint for _, fingerprint in indexed_fingerprints]
    similarity_sum = 0.0
    pair_count = 0
    per_fingerprint_similarity_sum = [0.0 for _ in fingerprints]
    for left_idx in range(len(fingerprints)):
        for right_idx in range(left_idx + 1, len(fingerprints)):
            similarity = float(
                DataStructs.TanimotoSimilarity(
                    fingerprints[left_idx],
                    fingerprints[right_idx],
                )
            )
            similarity_sum += similarity
            per_fingerprint_similarity_sum[left_idx] += similarity
            per_fingerprint_similarity_sum[right_idx] += similarity
            pair_count += 1
    return similarity_sum, pair_count, per_fingerprint_similarity_sum


def _compute_relative_scaffold_diversity(smiles_list):
    indexed_scaffolds = _compute_indexed_scaffolds(smiles_list)
    valid_count = len(indexed_scaffolds)
    if valid_count == 0:
        return 0.0
    unique_count = len({scaffold for _, scaffold in indexed_scaffolds})
    return unique_count / valid_count


def _compute_relative_scaffold_diversity_with_loo_credits(smiles_list):
    indexed_scaffolds = _compute_indexed_scaffolds(smiles_list)
    valid_count = len(indexed_scaffolds)
    if valid_count == 0:
        return 0.0, [0.0 for _ in smiles_list]

    scaffold_counts = Counter(scaffold for _, scaffold in indexed_scaffolds)
    unique_count = len(scaffold_counts)
    full_diversity = unique_count / valid_count
    credits = [0.0 for _ in smiles_list]

    for original_idx, scaffold in indexed_scaffolds:
        if valid_count == 1:
            reduced_diversity = 0.0
        elif scaffold_counts[scaffold] == 1:
            reduced_diversity = (unique_count - 1) / (valid_count - 1)
        else:
            reduced_diversity = unique_count / (valid_count - 1)
        credits[original_idx] = full_diversity - reduced_diversity
    return full_diversity, credits


def _compute_indexed_scaffolds(smiles_list):
    from rdkit.Chem import MolFromSmiles, MolToSmiles
    from rdkit.Chem.Scaffolds import MurckoScaffold

    indexed_scaffolds = []
    for original_idx, smiles in enumerate(smiles_list):
        if smiles is None:
            continue
        mol = MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            continue
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold = MolToSmiles(
            scaffold_mol,
            canonical=True,
            isomericSmiles=False,
        )
        indexed_scaffolds.append((original_idx, scaffold))
    return indexed_scaffolds
