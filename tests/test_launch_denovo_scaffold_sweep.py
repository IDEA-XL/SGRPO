import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts/launch_denovo_scaffold_sweep.py"
SPEC = importlib.util.spec_from_file_location(
    "launch_denovo_scaffold_sweep_for_test",
    MODULE_PATH,
)
launcher = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


class LaunchDenovoScaffoldSweepTest(unittest.TestCase):
    def test_expected_checkpoint_is_bound_to_training_job(self):
        with tempfile.TemporaryDirectory() as directory:
            original_runs_root = launcher.RUNS_ROOT
            launcher.RUNS_ROOT = Path(directory)
            try:
                checkpoint = launcher._expected_sgrpo_checkpoint(123)
            finally:
                launcher.RUNS_ROOT = original_runs_root

        self.assertEqual(
            checkpoint,
            Path(directory)
            / "cpgrpo_denovo"
            / f"{launcher.SGRPO_CONFIG_STEM}_slurm123"
            / "checkpoint-002000/model.ckpt",
        )


if __name__ == "__main__":
    unittest.main()
