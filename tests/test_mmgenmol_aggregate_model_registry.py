import unittest

from scripts.aggregate_mmgenmol_sweep_results import (
    _reward_weights_for_model,
    _uses_unidock_reward,
)


class MmgenmolAggregateModelRegistryTest(unittest.TestCase):
    EXPECTED_WEIGHTS = {
        "qed": 0.3,
        "sa_score": 0.2,
        "drugclip_score": 0.0,
        "unidock_score": 0.5,
    }

    def test_baseline_expansion_models_use_training_reward_weights(self):
        for model_name in ("mmgenmol_dmb_1000", "mmgenmol_entropy_1000"):
            with self.subTest(model_name=model_name):
                self.assertEqual(
                    _reward_weights_for_model(model_name),
                    self.EXPECTED_WEIGHTS,
                )
                self.assertTrue(_uses_unidock_reward(model_name))

    def test_unknown_model_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "No reward-weight mapping registered"):
            _reward_weights_for_model("unregistered_model")


if __name__ == "__main__":
    unittest.main()
