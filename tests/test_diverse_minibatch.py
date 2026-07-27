import unittest

import numpy as np
import torch

from genmol.rl.cpgrpo import selective_log_softmax
from rl_shared.diverse_minibatch import (
    _sample_k_eigenvectors_logspace,
    _sample_projection_dpp_gs,
    optimization_group_size,
    select_molecule_groups,
    select_sequence_groups,
)
from rl_shared.sgrpo import compute_grouped_advantages
from rl_shared.sgrpo import compute_clipped_grpo_loss


class DiverseMiniBatchTest(unittest.TestCase):
    def test_candidate_count_maps_to_original_optimization_group_size(self):
        self.assertEqual(
            optimization_group_size(
                1024,
                enabled=True,
                oversample_factor=2,
            ),
            512,
        )
        self.assertEqual(
            optimization_group_size(
                512,
                enabled=False,
                oversample_factor=2,
            ),
            512,
        )

    def test_molecule_shortfall_uses_all_valid_candidates_and_masks_padding(self):
        selection = select_molecule_groups(
            ['CCO', None, 'not-a-smiles', 'CCN'],
            candidate_size=4,
            selected_size=3,
            seed=7,
        )
        self.assertEqual(sum(selection.active_mask), 2)
        self.assertEqual(len(selection.indices), 3)
        active_smiles = [
            ['CCO', None, 'not-a-smiles', 'CCN'][idx]
            for idx, active in zip(selection.indices, selection.active_mask)
            if active
        ]
        self.assertEqual(sorted(active_smiles), ['CCN', 'CCO'])
        self.assertEqual(selection.metrics['shortfall_count'], 1.0)

    def test_molecule_all_invalid_group_is_fully_inactive(self):
        selection = select_molecule_groups(
            [None, "not-a-smiles", "", None],
            candidate_size=4,
            selected_size=3,
            seed=7,
        )
        self.assertEqual(selection.indices, (0, 0, 0))
        self.assertEqual(selection.active_mask, (False, False, False))
        self.assertEqual(selection.metrics["selected_count"], 0.0)
        self.assertEqual(selection.metrics["shortfall_count"], 3.0)

    def test_exact_tanimoto_k_dpp_returns_unique_valid_subset(self):
        smiles = [
            'CCO',
            'CCN',
            'CCC',
            'c1ccccc1',
            'CC(=O)O',
            'C1CCCCC1',
        ]
        selection = select_molecule_groups(
            smiles,
            candidate_size=6,
            selected_size=3,
            seed=11,
        )
        self.assertEqual(sum(selection.active_mask), 3)
        self.assertEqual(len(set(selection.indices)), 3)
        self.assertEqual(selection.metrics['exact_dpp_group_count'], 1.0)

    def test_exact_tanimoto_k_dpp_regularizes_rank_deficient_kernel(self):
        selection = select_molecule_groups(
            ['CCO'] * 6,
            candidate_size=6,
            selected_size=3,
            seed=13,
        )
        self.assertEqual(sum(selection.active_mask), 3)
        self.assertEqual(len(set(selection.indices)), 3)
        self.assertEqual(selection.metrics['raw_kernel_rank_min'], 1.0)
        self.assertEqual(
            selection.metrics['regularized_dpp_group_count'],
            1.0,
        )

    def test_projection_dpp_stably_samples_large_fixed_cardinality(self):
        random = np.random.RandomState(17)
        matrix = random.standard_normal((384, 192))
        eigenvectors, _ = np.linalg.qr(matrix, mode='reduced')

        selection = _sample_projection_dpp_gs(
            eigenvectors,
            random_state=np.random.RandomState(23),
        )

        self.assertEqual(len(selection), 192)
        self.assertEqual(len(set(selection)), 192)
        self.assertTrue(all(0 <= index < 384 for index in selection))

    def test_k_dpp_eigenvector_selection_handles_overflow_scale(self):
        eigenvalues = np.full(384, 1.0e8, dtype=np.float64)
        eigenvectors = np.eye(384, dtype=np.float64)

        selection = _sample_k_eigenvectors_logspace(
            eigenvalues,
            eigenvectors,
            size=192,
            random_state=np.random.RandomState(29),
        )

        self.assertEqual(selection.shape, (384, 192))
        selected_indices = np.argmax(selection, axis=0)
        self.assertEqual(len(set(selected_indices.tolist())), 192)
        self.assertTrue(
            np.allclose(
                selection.T @ selection,
                np.eye(192, dtype=np.float64),
            )
        )

    def test_k_dpp_eigenvector_selection_matches_exact_distribution(self):
        eigenvalues = np.array([1.0, 2.0, 4.0], dtype=np.float64)
        eigenvectors = np.eye(3, dtype=np.float64)
        counts = {(0, 1): 0, (0, 2): 0, (1, 2): 0}
        trials = 6000

        for seed in range(trials):
            selection = _sample_k_eigenvectors_logspace(
                eigenvalues,
                eigenvectors,
                size=2,
                random_state=np.random.RandomState(seed),
            )
            selected = tuple(sorted(np.argmax(selection, axis=0).tolist()))
            counts[selected] += 1

        expected = {
            (0, 1): 2.0 / 14.0,
            (0, 2): 4.0 / 14.0,
            (1, 2): 8.0 / 14.0,
        }
        for selected, probability in expected.items():
            self.assertAlmostEqual(
                counts[selected] / trials,
                probability,
                delta=0.02,
            )

    def test_sequence_maxmin_selects_fixed_batch_without_validity_filtering(self):
        selection = select_sequence_groups(
            ['ACDE', '', 'AX', 'WYYY'],
            candidate_size=4,
            selected_size=3,
            seed=5,
        )
        self.assertEqual(selection.active_mask, (True, True, True))
        self.assertEqual(len(selection.indices), 3)
        self.assertEqual(len(set(selection.indices)), 3)
        self.assertEqual(selection.metrics['valid_candidate_count'], 2.0)
        self.assertEqual(selection.metrics['selected_count'], 3.0)
        self.assertEqual(selection.metrics['shortfall_count'], 0.0)
        self.assertGreater(selection.metrics['selected_invalid_count'], 0.0)

    def test_sequence_all_invalid_group_keeps_fixed_active_batch(self):
        selection = select_sequence_groups(
            ['', 'AX', None, ''],
            candidate_size=4,
            selected_size=3,
            seed=5,
        )
        self.assertEqual(len(selection.indices), 3)
        self.assertEqual(len(set(selection.indices)), 3)
        self.assertEqual(selection.active_mask, (True, True, True))
        self.assertEqual(selection.metrics['valid_candidate_count'], 0.0)
        self.assertEqual(selection.metrics['selected_valid_count'], 0.0)
        self.assertEqual(selection.metrics['selected_invalid_count'], 3.0)
        self.assertEqual(selection.metrics['selected_count'], 3.0)
        self.assertEqual(selection.metrics['shortfall_count'], 0.0)

    def test_selected_invalid_reward_receives_negative_group_advantage(self):
        selection = select_sequence_groups(
            ['ACDE', '', 'AX', 'WYYY'],
            candidate_size=4,
            selected_size=3,
            seed=5,
        )
        rewards = torch.tensor([
            1.0 if index in (0, 3) else 0.0
            for index in selection.indices
        ])
        advantages, _, _ = compute_grouped_advantages(
            rewards,
            num_generations=3,
        )
        invalid_positions = [
            position
            for position, index in enumerate(selection.indices)
            if index not in (0, 3)
        ]
        self.assertTrue(invalid_positions)
        self.assertTrue(torch.all(advantages[invalid_positions] < 0.0))

    def test_masked_grouped_advantages_ignore_padding(self):
        rewards = torch.tensor([1.0, 2.0, 100.0, 4.0])
        mask = torch.tensor([True, True, False, True])
        advantages, repeated_std, zero_std_ratio = compute_grouped_advantages(
            rewards,
            num_generations=2,
            sample_mask=mask,
        )
        self.assertTrue(
            torch.allclose(
                advantages,
                torch.tensor([-1.0, 1.0, 0.0, 0.0]),
            )
        )
        self.assertEqual(repeated_std[2].item(), 0.0)
        self.assertAlmostEqual(zero_std_ratio, 1.0 / 3.0)

    def test_masked_grouped_advantages_accept_empty_group(self):
        rewards = torch.tensor([7.0, 9.0, 1.0, 3.0])
        mask = torch.tensor([False, False, True, True])
        advantages, repeated_std, zero_std_ratio = compute_grouped_advantages(
            rewards,
            num_generations=2,
            sample_mask=mask,
        )
        self.assertTrue(
            torch.allclose(
                advantages,
                torch.tensor([0.0, 0.0, -2.0, 2.0]),
            )
        )
        self.assertTrue(
            torch.allclose(
                repeated_std,
                torch.tensor([0.0, 0.0, 2.0 ** 0.5, 2.0 ** 0.5]),
            )
        )
        self.assertEqual(zero_std_ratio, 0.0)

    def test_masked_grouped_advantages_accept_all_inactive(self):
        rewards = torch.tensor([7.0, 9.0])
        mask = torch.tensor([False, False])
        advantages, repeated_std, zero_std_ratio = compute_grouped_advantages(
            rewards,
            num_generations=2,
            sample_mask=mask,
        )
        self.assertTrue(torch.equal(advantages, torch.zeros_like(rewards)))
        self.assertTrue(torch.equal(repeated_std, torch.zeros_like(rewards)))
        self.assertEqual(zero_std_ratio, 1.0)

    def test_zero_active_completion_loss_remains_backwardable(self):
        new_log_probs = torch.zeros(2, 1, 4, requires_grad=True)
        old_log_probs = torch.zeros_like(new_log_probs)
        completion_mask = torch.zeros(2, 4, dtype=torch.bool)
        loss, metrics = compute_clipped_grpo_loss(
            new_log_probs=new_log_probs,
            old_log_probs=old_log_probs,
            advantages=torch.zeros(2),
            completion_mask=completion_mask,
            epsilon=0.2,
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics['ratio_mean'].item(), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(new_log_probs.grad, torch.zeros_like(new_log_probs)))

    def test_coupled_entropy_uses_same_states_without_an_extra_forward(self):
        logits = torch.zeros(3, 2, 4, requires_grad=True)
        targets = torch.tensor([[0, 1]])
        partial_mask = torch.tensor([[True, False]])
        logps, entropy = selective_log_softmax(
            logits,
            targets,
            weights=torch.tensor([1.0, 2.0, 3.0]),
            mask=partial_mask,
            return_normalized_entropy=True,
        )
        self.assertEqual(tuple(logps.shape), (1, 2))
        self.assertEqual(tuple(entropy.shape), (1, 2))
        self.assertTrue(torch.all(entropy > 0.0))
        entropy.mean().backward()
        self.assertIsNotNone(logits.grad)


if __name__ == '__main__':
    unittest.main()
