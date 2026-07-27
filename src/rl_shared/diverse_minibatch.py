from __future__ import annotations

import time
from dataclasses import dataclass


_DPP_DIAGONAL_JITTER = 1.0e-6


@dataclass(frozen=True)
class DiverseMiniBatchSelection:
    indices: tuple[int, ...]
    active_mask: tuple[bool, ...]
    metrics: dict[str, float]


def validate_diverse_minibatch_config(*, enabled, oversample_factor):
    if not isinstance(enabled, bool):
        raise ValueError(f'diverse_minibatch must be boolean, got {enabled!r}')
    if isinstance(oversample_factor, bool):
        raise ValueError('diverse_minibatch_oversample_factor must be an integer')
    try:
        factor = int(oversample_factor)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            'diverse_minibatch_oversample_factor must be an integer'
        ) from exc
    if factor != oversample_factor:
        raise ValueError(
            'diverse_minibatch_oversample_factor must be an integer, '
            f'got {oversample_factor!r}'
        )
    if enabled and factor <= 1:
        raise ValueError(
            'diverse_minibatch_oversample_factor must be greater than 1 '
            'when diverse_minibatch is enabled'
        )
    if not enabled and factor <= 0:
        raise ValueError('diverse_minibatch_oversample_factor must be positive')
    return factor


def optimization_group_size(num_generations, *, enabled, oversample_factor):
    if num_generations <= 1:
        raise ValueError('num_generations must be greater than 1')
    factor = validate_diverse_minibatch_config(
        enabled=enabled,
        oversample_factor=oversample_factor,
    )
    if not enabled:
        return int(num_generations)
    if int(num_generations) % factor != 0:
        raise ValueError(
            'Diverse Mini-Batch GRPO requires num_generations to be divisible '
            f'by the oversample factor: {num_generations} vs {factor}'
        )
    selected_size = int(num_generations) // factor
    if selected_size <= 1:
        raise ValueError(
            'Diverse Mini-Batch GRPO optimization group size must be greater '
            f'than 1, got {selected_size}'
        )
    return selected_size


def select_molecule_groups(
    smiles,
    *,
    candidate_size,
    selected_size,
    seed,
):
    from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator

    if candidate_size <= selected_size:
        raise ValueError(
            f'candidate_size must exceed selected_size, got {candidate_size} and {selected_size}'
        )
    groups = _validate_grouped_items(smiles, candidate_size)
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
    )

    started = time.perf_counter()
    selected_indices = []
    active_mask = []
    valid_count = 0
    dpp_group_count = 0
    fingerprint_sec = 0.0
    kernel_sec = 0.0
    eigensample_sec = 0.0
    raw_kernel_ranks = []
    regularized_group_count = 0.0

    for group_idx, group in enumerate(groups):
        fp_started = time.perf_counter()
        group_valid_indices = []
        fingerprints = []
        for item_idx, item in enumerate(group):
            if not item:
                continue
            molecule = MolFromSmiles(str(item), sanitize=True)
            if molecule is None:
                continue
            group_valid_indices.append(item_idx)
            fingerprints.append(fingerprint_generator.GetFingerprint(molecule))
        fingerprint_sec += time.perf_counter() - fp_started
        valid_count += len(group_valid_indices)
        if not group_valid_indices:
            raise RuntimeError(
                'Diverse Mini-Batch GRPO found no valid molecular candidate in '
                f'group {group_idx}; refusing to fabricate an optimization sample'
            )

        if len(group_valid_indices) <= selected_size:
            group_picks = list(range(len(group_valid_indices)))
        else:
            group_picks, timings = _sample_exact_tanimoto_k_dpp(
                fingerprints,
                size=selected_size,
                seed=int(seed) + group_idx,
            )
            kernel_sec += timings['kernel_sec']
            eigensample_sec += timings['eigensample_sec']
            raw_kernel_ranks.append(timings['raw_kernel_rank'])
            regularized_group_count += timings['regularized']
            dpp_group_count += 1

        original_group_picks = sorted(group_valid_indices[idx] for idx in group_picks)
        padded_picks, group_mask = _pad_group_selection(
            original_group_picks,
            selected_size=selected_size,
        )
        group_offset = group_idx * candidate_size
        selected_indices.extend(group_offset + idx for idx in padded_picks)
        active_mask.extend(group_mask)

    selected_count = sum(active_mask)
    elapsed = time.perf_counter() - started
    metrics = _selection_metrics(
        candidate_count=len(smiles),
        valid_count=valid_count,
        selected_count=selected_count,
        target_count=len(groups) * selected_size,
        elapsed=elapsed,
    )
    metrics.update(
        {
            'fingerprint_sec': float(fingerprint_sec),
            'kernel_sec': float(kernel_sec),
            'eigensample_sec': float(eigensample_sec),
            'exact_dpp_group_count': float(dpp_group_count),
            'raw_kernel_rank_min': float(
                min(raw_kernel_ranks) if raw_kernel_ranks else 0.0
            ),
            'raw_kernel_rank_mean': float(
                sum(raw_kernel_ranks) / len(raw_kernel_ranks)
                if raw_kernel_ranks
                else 0.0
            ),
            'regularized_dpp_group_count': float(regularized_group_count),
            'dpp_diagonal_jitter': float(_DPP_DIAGONAL_JITTER),
        }
    )
    return DiverseMiniBatchSelection(
        indices=tuple(selected_indices),
        active_mask=tuple(active_mask),
        metrics=metrics,
    )


def select_sequence_groups(
    sequences,
    *,
    candidate_size,
    selected_size,
    seed,
):
    from progen2.rewards.common import (
        is_valid_protein_sequence,
        normalize_protein_sequence,
    )

    if candidate_size <= selected_size:
        raise ValueError(
            f'candidate_size must exceed selected_size, got {candidate_size} and {selected_size}'
        )
    groups = _validate_grouped_items(sequences, candidate_size)
    started = time.perf_counter()
    selected_indices = []
    active_mask = []
    valid_count = 0
    selected_valid_count = 0
    distance_sec = 0.0

    for group_idx, group in enumerate(groups):
        normalized_sequences = [
            normalize_protein_sequence(sequence)
            for sequence in group
        ]
        group_valid_flags = [
            is_valid_protein_sequence(sequence)
            for sequence in group
        ]
        valid_count += sum(group_valid_flags)
        distance_started = time.perf_counter()
        group_picks = _maxmin_edit_distance_selection(
            normalized_sequences,
            size=selected_size,
            seed=int(seed) + group_idx,
        )
        distance_sec += time.perf_counter() - distance_started
        group_picks = sorted(group_picks)
        selected_valid_count += sum(group_valid_flags[idx] for idx in group_picks)

        group_offset = group_idx * candidate_size
        selected_indices.extend(group_offset + idx for idx in group_picks)
        active_mask.extend([True] * selected_size)

    selected_count = sum(active_mask)
    metrics = _selection_metrics(
        candidate_count=len(sequences),
        valid_count=valid_count,
        selected_count=selected_count,
        target_count=len(groups) * selected_size,
        elapsed=time.perf_counter() - started,
    )
    metrics['distance_and_maxmin_sec'] = float(distance_sec)
    metrics['selected_valid_count'] = float(selected_valid_count)
    metrics['selected_valid_fraction'] = float(
        selected_valid_count / selected_count
    )
    metrics['selected_invalid_count'] = float(
        selected_count - selected_valid_count
    )
    return DiverseMiniBatchSelection(
        indices=tuple(selected_indices),
        active_mask=tuple(active_mask),
        metrics=metrics,
    )


def _sample_exact_tanimoto_k_dpp(fingerprints, *, size, seed):
    import numpy as np
    from rdkit import DataStructs

    try:
        from dppy.exact_sampling import (
            elementary_symmetric_polynomials,
            k_dpp_eig_vecs_selector,
        )
    except ImportError as exc:
        raise RuntimeError(
            'Exact molecular k-DPP selection requires dppy==0.3.3'
        ) from exc

    if size <= 0 or size > len(fingerprints):
        raise ValueError(
            f'k-DPP size must be in [1, {len(fingerprints)}], got {size}'
        )

    kernel_started = time.perf_counter()
    likelihood_kernel = np.empty(
        (len(fingerprints), len(fingerprints)),
        dtype=np.float64,
    )
    for row_idx, fingerprint in enumerate(fingerprints):
        likelihood_kernel[row_idx] = DataStructs.BulkTanimotoSimilarity(
            fingerprint,
            fingerprints,
        )
    likelihood_kernel = (likelihood_kernel + likelihood_kernel.T) * 0.5
    kernel_sec = time.perf_counter() - kernel_started

    eigensample_started = time.perf_counter()
    eigenvalues, eigenvectors = np.linalg.eigh(likelihood_kernel)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = (
        np.finfo(np.float64).eps
        * max(1, likelihood_kernel.shape[0])
        * scale
        * 10.0
    )
    if float(eigenvalues.min()) < -tolerance:
        raise RuntimeError(
            'Morgan-Tanimoto likelihood kernel is not positive semidefinite '
            f'within numerical tolerance: min_eigenvalue={float(eigenvalues.min()):.6g}, '
            f'tolerance={tolerance:.6g}'
        )
    eigenvalues = np.where(eigenvalues < 0.0, 0.0, eigenvalues)
    rank = int((eigenvalues > tolerance).sum())
    regularized = rank < size
    if regularized:
        # L + eps*I is full rank. Scaling by 1/eps prevents underflow in the
        # fixed-cardinality recursion and leaves the k-DPP distribution unchanged.
        eigenvalues = (
            eigenvalues + _DPP_DIAGONAL_JITTER
        ) / _DPP_DIAGONAL_JITTER

    rng = np.random.RandomState(int(seed))
    elementary_polynomials = elementary_symmetric_polynomials(
        eigenvalues,
        size,
    )
    normalizer = float(elementary_polynomials[size, -1])
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        raise RuntimeError(
            'Exact k-DPP eigenvector-selection normalizer is not finite and '
            f'positive: {normalizer}'
        )
    projection_eigenvectors = k_dpp_eig_vecs_selector(
        eigenvalues,
        eigenvectors,
        size=size,
        E_poly=elementary_polynomials,
        random_state=rng,
    )
    sample = _sample_projection_dpp_gs(
        projection_eigenvectors,
        random_state=rng,
    )
    eigensample_sec = time.perf_counter() - eigensample_started
    picks = [int(idx) for idx in sample]
    if len(picks) != size or len(set(picks)) != size:
        raise RuntimeError(
            'Exact k-DPP returned an invalid sample: '
            f'expected {size} unique indices, got {len(picks)} entries and '
            f'{len(set(picks))} unique entries'
        )
    return picks, {
        'kernel_sec': float(kernel_sec),
        'eigensample_sec': float(eigensample_sec),
        'raw_kernel_rank': float(rank),
        'regularized': float(regularized),
    }


def _sample_projection_dpp_gs(eigenvectors, *, random_state):
    """Sample a projection DPP with numerically normalized GS probabilities."""
    import numpy as np

    vectors = np.asarray(eigenvectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] <= 0:
        raise ValueError(
            'Projection-DPP eigenvectors must be a non-empty matrix, got '
            f'{vectors.shape}'
        )
    if not np.isfinite(vectors).all():
        raise ValueError('Projection-DPP eigenvectors must all be finite')

    ground_size, rank = vectors.shape
    gram_error = float(
        np.max(np.abs(vectors.T @ vectors - np.eye(rank)))
    )
    orthogonality_tolerance = (
        np.finfo(np.float64).eps
        * max(1, ground_size, rank)
        * 1000.0
    )
    if gram_error > orthogonality_tolerance:
        raise RuntimeError(
            'Projection-DPP eigenvectors are not orthonormal within numerical '
            f'tolerance: error={gram_error:.6g}, '
            f'tolerance={orthogonality_tolerance:.6g}'
        )

    available = np.ones(ground_size, dtype=bool)
    contributions = np.zeros((ground_size, rank), dtype=np.float64)
    residual_norms = np.einsum('ij,ij->i', vectors, vectors)
    selected = []
    negative_tolerance = orthogonality_tolerance

    for iteration in range(rank):
        available_indices = np.flatnonzero(available)
        weights = residual_norms[available_indices]
        minimum_weight = float(weights.min())
        if minimum_weight < -negative_tolerance:
            raise RuntimeError(
                'Projection-DPP Gram-Schmidt produced a materially negative '
                f'residual norm at iteration {iteration}: {minimum_weight:.6g}'
            )
        weights = np.maximum(weights, 0.0)
        total_weight = float(weights.sum())
        if not np.isfinite(total_weight) or total_weight <= 0.0:
            raise RuntimeError(
                'Projection-DPP Gram-Schmidt has no positive finite sampling '
                f'mass at iteration {iteration}: {total_weight}'
            )
        probabilities = weights / total_weight
        correction_index = int(np.argmax(probabilities))
        probabilities[correction_index] += 1.0 - float(probabilities.sum())
        if (
            not np.isfinite(probabilities).all()
            or float(probabilities.min()) < 0.0
        ):
            raise RuntimeError(
                'Projection-DPP Gram-Schmidt produced invalid probabilities '
                f'at iteration {iteration}'
            )

        selected_index = int(
            random_state.choice(available_indices, p=probabilities)
        )
        selected.append(selected_index)
        if iteration == rank - 1:
            break

        selected_norm = float(residual_norms[selected_index])
        if selected_norm <= 0.0 or not np.isfinite(selected_norm):
            raise RuntimeError(
                'Projection-DPP selected an item with non-positive residual '
                f'norm at iteration {iteration}: {selected_norm}'
            )
        available[selected_index] = False
        remaining = np.flatnonzero(available)
        contributions[remaining, iteration] = (
            vectors[remaining] @ vectors[selected_index]
            - contributions[remaining, :iteration]
            @ contributions[selected_index, :iteration]
        ) / np.sqrt(selected_norm)
        residual_norms[remaining] -= (
            contributions[remaining, iteration] ** 2
        )
        residual_norms[selected_index] = 0.0

    if len(selected) != rank or len(set(selected)) != rank:
        raise RuntimeError(
            'Projection-DPP Gram-Schmidt returned an invalid sample: '
            f'{len(selected)} entries, {len(set(selected))} unique'
        )
    return selected


def _maxmin_edit_distance_selection(sequences, *, size, seed):
    import numpy as np

    try:
        from rapidfuzz import process as rapidfuzz_process
        from rapidfuzz.distance import Levenshtein
    except ImportError as exc:
        raise RuntimeError(
            'Protein MaxMin selection requires rapidfuzz'
        ) from exc

    if size <= 0 or size > len(sequences):
        raise ValueError(
            f'MaxMin size must be in [1, {len(sequences)}], got {size}'
        )
    similarities = rapidfuzz_process.cdist(
        sequences,
        sequences,
        scorer=Levenshtein.normalized_similarity,
        workers=1,
        dtype=np.float32,
    )
    distances = 1.0 - similarities
    np.fill_diagonal(distances, 0.0)

    rng = np.random.RandomState(int(seed))
    first = int(rng.randint(len(sequences)))
    selected = [first]
    selected_mask = np.zeros(len(sequences), dtype=bool)
    selected_mask[first] = True
    min_distance = distances[first].copy()
    while len(selected) < size:
        scores = np.where(selected_mask, -np.inf, min_distance)
        next_idx = int(np.argmax(scores))
        if selected_mask[next_idx]:
            raise RuntimeError('MaxMin failed to identify an unselected sequence')
        selected.append(next_idx)
        selected_mask[next_idx] = True
        min_distance = np.minimum(min_distance, distances[next_idx])
    return selected


def _validate_grouped_items(items, group_size):
    if group_size <= 0:
        raise ValueError('group_size must be positive')
    if not items:
        raise ValueError('candidate items must be non-empty')
    if len(items) % group_size != 0:
        raise ValueError(
            f'candidate count must be divisible by group_size: {len(items)} vs {group_size}'
        )
    return [
        list(items[start:start + group_size])
        for start in range(0, len(items), group_size)
    ]


def _pad_group_selection(
    indices,
    *,
    selected_size,
    empty_fallback_index=None,
):
    if not indices:
        if empty_fallback_index is None:
            raise RuntimeError('cannot pad an empty selected group')
        if int(empty_fallback_index) != empty_fallback_index or empty_fallback_index < 0:
            raise ValueError(
                'empty_fallback_index must be a non-negative integer, '
                f'got {empty_fallback_index!r}'
            )
        return [int(empty_fallback_index)] * selected_size, [False] * selected_size
    if len(indices) > selected_size:
        raise ValueError(
            f'selected group exceeds target size: {len(indices)} vs {selected_size}'
        )
    active_count = len(indices)
    padded = list(indices)
    padded.extend([indices[0]] * (selected_size - active_count))
    active_mask = [True] * active_count + [False] * (selected_size - active_count)
    return padded, active_mask


def _selection_metrics(
    *,
    candidate_count,
    valid_count,
    selected_count,
    target_count,
    elapsed,
):
    if candidate_count <= 0 or target_count <= 0:
        raise ValueError('candidate_count and target_count must be positive')
    if not 0 <= selected_count <= target_count:
        raise ValueError(
            f'selected_count must be in [0, {target_count}], got {selected_count}'
        )
    return {
        'candidate_count': float(candidate_count),
        'valid_candidate_count': float(valid_count),
        'valid_candidate_fraction': float(valid_count / candidate_count),
        'selected_count': float(selected_count),
        'target_optimization_count': float(target_count),
        'shortfall_count': float(target_count - selected_count),
        'shortfall_fraction': float((target_count - selected_count) / target_count),
        'selection_sec': float(elapsed),
    }
