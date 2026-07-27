import importlib.util
import sys
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_MODULE_PATH = REPO_ROOT / "scripts/eval_denovo_sgrpo.py"
SPEC = importlib.util.spec_from_file_location(
    "eval_denovo_sgrpo_for_scaffold_test",
    EVAL_MODULE_PATH,
)
denovo_eval = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = denovo_eval
SPEC.loader.exec_module(denovo_eval)


FORMAL_CONFIGS = {
    "grpo": (
        "cpgrpo_denovo_scaffold_grpo_ng512_bs2048_lr5e-5_beta5e-3_"
        "ni1_gc_ms2000_stl0.yaml"
    ),
    "hbd": (
        "cpgrpo_denovo_scaffold_hbd_ng512_bs2048_lr5e-5_beta5e-3_"
        "ni1_gc_st09_sc04_ms2000_stl0.yaml"
    ),
    "sgrpo": (
        "cpgrpo_denovo_scaffold_sgrpo_ng64_sg8_bs2048_lr5e-5_beta5e-3_"
        "gw09_rewardsum_loo_gc_ms2000_stl0.yaml"
    ),
    "dmb": (
        "cpgrpo_denovo_scaffold_dmb_ng1024_bs2048_lr5e-5_beta5e-3_"
        "ni1_gc_ms2000_stl0.yaml"
    ),
    "entropy": (
        "cpgrpo_denovo_scaffold_entropy001_ng512_bs2048_lr5e-5_beta5e-3_"
        "ni1_gc_ms2000_stl0.yaml"
    ),
}


class DenovoScaffoldExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configs = {
            name: yaml.safe_load((REPO_ROOT / "configs" / filename).read_text())
            for name, filename in FORMAL_CONFIGS.items()
        }

    def test_all_formal_configs_use_requested_four_gpu_settings(self):
        for name, config in self.configs.items():
            with self.subTest(name=name):
                self.assertEqual(config["diversity_metric"], "relative_scaffold_diversity")
                self.assertTrue(config["gradient_checkpointing"])
                self.assertEqual(config["gradient_checkpointing_kwargs"], {"use_reentrant": False})
                self.assertEqual(config["per_device_train_batch_size"], 2048)
                self.assertEqual(config["generation_batch_size"], 2048)
                self.assertEqual(config["gradient_accumulation_steps"], 1)
                self.assertEqual(config["max_steps"], 2000)
                self.assertEqual(config["save_total_limit"], 0)

    def test_four_gpu_global_optimization_batch_matches_original_eight_gpu_batch(self):
        expected_global_batch = 8 * 1024
        for name, config in self.configs.items():
            with self.subTest(name=name):
                actual = (
                    4
                    * config["per_device_train_batch_size"]
                    * config["gradient_accumulation_steps"]
                )
                self.assertEqual(actual, expected_global_batch)

    def test_group_geometry_is_preserved(self):
        global_batch = 4 * 2048
        self.assertEqual(global_batch // self.configs["grpo"]["num_generations"], 16)
        self.assertEqual(global_batch // self.configs["hbd"]["num_generations"], 16)
        self.assertEqual(global_batch // self.configs["entropy"]["num_generations"], 16)

        sgrpo = self.configs["sgrpo"]
        sgrpo_groups = global_batch // sgrpo["num_generations"]
        self.assertEqual(sgrpo_groups, 128)
        self.assertEqual(sgrpo_groups // sgrpo["supergroup_num_groups"], 16)

        dmb = self.configs["dmb"]
        selected_group_size = (
            dmb["num_generations"] // dmb["diverse_minibatch_oversample_factor"]
        )
        optimization_groups = global_batch // selected_group_size
        self.assertEqual(selected_group_size, 512)
        self.assertEqual(optimization_groups, 16)
        self.assertEqual(optimization_groups * dmb["num_generations"], 16384)

    def test_method_specific_settings_are_preserved(self):
        self.assertTrue(self.configs["hbd"]["hbd"])
        self.assertEqual(self.configs["hbd"]["hbd_score_threshold_for_memory"], 0.9)
        self.assertEqual(self.configs["hbd"]["hbd_similarity_cutoff"], 0.4)

        self.assertEqual(self.configs["sgrpo"]["rl_algorithm"], "coupled_sgrpo")
        self.assertEqual(self.configs["sgrpo"]["hierarchy"], "reward_sum")
        self.assertEqual(self.configs["sgrpo"]["group_rewrad_credit"], "loo")
        self.assertEqual(self.configs["sgrpo"]["group_advantage_weight"], 0.9)

        self.assertTrue(self.configs["dmb"]["diverse_minibatch"])
        self.assertEqual(self.configs["dmb"]["diverse_minibatch_oversample_factor"], 2)
        self.assertEqual(self.configs["entropy"]["entropy_regularization_weight"], 0.01)

    def test_eval_metric_labels_are_explicit(self):
        self.assertEqual(
            denovo_eval._diversity_display_name("morgan_internal_diversity"),
            "Internal Diversity",
        )
        self.assertEqual(
            denovo_eval._diversity_display_name("relative_scaffold_diversity"),
            "Scaffold Diversity",
        )
        with self.assertRaisesRegex(ValueError, "diversity_metric must be one of"):
            denovo_eval._diversity_display_name("scaffold")


if __name__ == "__main__":
    unittest.main()
