import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))
MODULE_PATH = SCRIPTS_ROOT / "build_denovo_credit_ablation_sweep_specs.py"
SPEC = importlib.util.spec_from_file_location(
    "build_denovo_credit_ablation_sweep_specs_for_test",
    MODULE_PATH,
)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


class DenovoCreditAblationSweepTest(unittest.TestCase):
    def test_builds_three_model_five_seed_specs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = []
            for index in range(3):
                checkpoint = root / f"checkpoint-{index}.ckpt"
                checkpoint.write_bytes(b"checkpoint")
                checkpoints.append(checkpoint)
            models = tuple(
                builder.ModelSpec(
                    name=f"model_{index}",
                    display_name=f"Model {index}",
                    checkpoint_path=checkpoint,
                )
                for index, checkpoint in enumerate(checkpoints)
            )
            run_root = root / "run"
            manifest_path = builder.build_specs(run_root, models)

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["profile"], "denovo_credit_ablation")
            self.assertEqual(manifest["denovo_tasks"], 15)
            self.assertEqual(manifest["total_expected_raw_rows"], 150_000)
            task_lines = (
                run_root / "specs/denovo/tasks.tsv"
            ).read_text().splitlines()
            self.assertEqual(len(task_lines), 16)
            config = (
                run_root
                / "specs/denovo/seed42/credit_ablation/model_0.yaml"
            ).read_text()
            self.assertIn("num_samples: 1000", config)
            self.assertIn("generation_batch_size: 2048", config)
            self.assertIn(
                "diversity_metric: morgan_internal_diversity",
                config,
            )
            self.assertEqual(config.count("randomness:"), 10)
            self.assertEqual(config.count("generation_temperature:"), 10)

    def test_rejects_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = builder.ModelSpec("missing", "Missing", root / "missing.ckpt")
            with self.assertRaises(FileNotFoundError):
                builder.build_specs(root / "run", (model,))

    def test_rejects_overwriting_specs_without_explicit_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            model = builder.ModelSpec("model", "Model", checkpoint)
            run_root = root / "run"
            builder.build_specs(run_root, (model,))
            with self.assertRaises(FileExistsError):
                builder.build_specs(run_root, (model,))


if __name__ == "__main__":
    unittest.main()
