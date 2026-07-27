import unittest
from unittest import mock

from genmol.diversity import (
    DEFAULT_DIVERSITY_METRIC,
    MORGAN_INTERNAL_DIVERSITY,
    RELATIVE_SCAFFOLD_DIVERSITY,
    compute_internal_diversity,
    compute_internal_diversity_loo_credits,
    compute_molecular_diversity,
    compute_molecular_diversity_with_loo_credits,
    validate_diversity_metric,
)


class MolecularDiversityTest(unittest.TestCase):
    def test_default_metric_is_existing_morgan_internal_diversity(self):
        smiles = ['CCO', 'CCN', 'c1ccccc1']

        self.assertEqual(DEFAULT_DIVERSITY_METRIC, MORGAN_INTERNAL_DIVERSITY)
        self.assertAlmostEqual(
            compute_molecular_diversity(smiles),
            compute_internal_diversity(smiles),
        )

    def test_relative_scaffold_diversity_excludes_invalid_molecules(self):
        smiles = [
            'c1ccccc1',
            'Cc1ccccc1',
            'C1CCCCC1',
            None,
            'not-a-smiles',
        ]

        diversity = compute_molecular_diversity(
            smiles,
            metric=RELATIVE_SCAFFOLD_DIVERSITY,
        )

        self.assertAlmostEqual(diversity, 2.0 / 3.0)

    def test_relative_scaffold_diversity_counts_empty_acyclic_scaffold(self):
        smiles = ['CCO', 'CCC', 'c1ccccc1']

        diversity = compute_molecular_diversity(
            smiles,
            metric=RELATIVE_SCAFFOLD_DIVERSITY,
        )

        self.assertAlmostEqual(diversity, 2.0 / 3.0)

    def test_relative_scaffold_diversity_counts_duplicate_entries_in_denominator(self):
        diversity = compute_molecular_diversity(
            ['c1ccccc1', 'c1ccccc1'],
            metric=RELATIVE_SCAFFOLD_DIVERSITY,
        )

        self.assertAlmostEqual(diversity, 0.5)

    def test_relative_scaffold_diversity_removes_stereochemistry(self):
        from rdkit import Chem

        original_mol_to_smiles = Chem.MolToSmiles

        def assert_stereo_removed(molecule, **kwargs):
            self.assertTrue(
                all(
                    bond.GetStereo() == Chem.BondStereo.STEREONONE
                    for bond in molecule.GetBonds()
                )
            )
            return original_mol_to_smiles(molecule, **kwargs)

        with mock.patch.object(
            Chem,
            'MolToSmiles',
            side_effect=assert_stereo_removed,
        ):
            diversity = compute_molecular_diversity(
                [
                    'c1ccccc1/C=C/c2ccccc2',
                    'c1ccccc1/C=C\\c2ccccc2',
                ],
                metric=RELATIVE_SCAFFOLD_DIVERSITY,
            )

        self.assertAlmostEqual(diversity, 0.5)

    def test_relative_scaffold_diversity_handles_empty_and_singleton_valid_sets(self):
        self.assertEqual(
            compute_molecular_diversity(
                [None, 'not-a-smiles'],
                metric=RELATIVE_SCAFFOLD_DIVERSITY,
            ),
            0.0,
        )
        self.assertEqual(
            compute_molecular_diversity(
                ['CCO', None],
                metric=RELATIVE_SCAFFOLD_DIVERSITY,
            ),
            1.0,
        )

    def test_relative_scaffold_loo_credits_match_naive_formula(self):
        smiles = ['c1ccccc1', 'Cc1ccccc1', 'C1CCCCC1', None]
        full_diversity = compute_molecular_diversity(
            smiles,
            metric=RELATIVE_SCAFFOLD_DIVERSITY,
        )
        expected = []
        for remove_idx in range(len(smiles)):
            reduced = smiles[:remove_idx] + smiles[remove_idx + 1:]
            expected.append(
                full_diversity
                - compute_molecular_diversity(
                    reduced,
                    metric=RELATIVE_SCAFFOLD_DIVERSITY,
                )
            )

        joint_diversity, actual = compute_molecular_diversity_with_loo_credits(
            smiles,
            metric=RELATIVE_SCAFFOLD_DIVERSITY,
        )

        self.assertAlmostEqual(joint_diversity, full_diversity)
        self.assertEqual(len(actual), len(smiles))
        for left, right in zip(actual, expected):
            self.assertAlmostEqual(left, right)

    def test_joint_morgan_computation_preserves_existing_loo_credits(self):
        smiles = ['CCO', 'CCN', 'c1ccccc1', None, 'not-a-smiles']

        diversity, credits = compute_molecular_diversity_with_loo_credits(
            smiles,
            metric=MORGAN_INTERNAL_DIVERSITY,
        )

        self.assertAlmostEqual(diversity, compute_internal_diversity(smiles))
        expected_credits = compute_internal_diversity_loo_credits(smiles)
        for left, right in zip(credits, expected_credits):
            self.assertAlmostEqual(left, right)

    def test_metric_validation_is_strict(self):
        self.assertEqual(
            validate_diversity_metric(RELATIVE_SCAFFOLD_DIVERSITY),
            RELATIVE_SCAFFOLD_DIVERSITY,
        )
        with self.assertRaisesRegex(ValueError, 'diversity_metric must be one of'):
            validate_diversity_metric('scaffold')
        with self.assertRaisesRegex(TypeError, 'diversity_metric must be a string'):
            validate_diversity_metric(None)


if __name__ == '__main__':
    unittest.main()
