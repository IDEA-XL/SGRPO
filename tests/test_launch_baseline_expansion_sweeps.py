import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts/launch_baseline_expansion_sweeps.py"
SPEC = importlib.util.spec_from_file_location(
    "launch_baseline_expansion_sweeps_for_test",
    MODULE_PATH,
)
launcher = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


class LaunchBaselineExpansionSweepsTest(unittest.TestCase):
    def _molecule_run(self, root, stem, suffix, step, *, complete=True):
        run_dir = root / f"{stem}_{suffix}"
        checkpoint_dir = run_dir / f"checkpoint-{step:06d}"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "model.ckpt").write_bytes(b"model")
        (checkpoint_dir / "trainer_state.json").write_text(
            json.dumps({"global_step": step if complete else step - 1})
        )
        (run_dir / "train_results.json").write_text(
            json.dumps({"step": step if complete else step - 1})
        )
        return run_dir

    def test_resolves_only_complete_molecule_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stem = "config"
            self._molecule_run(root, stem, "partial", 100, complete=False)
            complete = self._molecule_run(root, stem, "complete", 100)

            resolved = launcher._resolve_molecule_checkpoint(
                task_root=root,
                config_stem=stem,
                checkpoint_step=100,
            )

        self.assertEqual(
            resolved,
            complete / "checkpoint-000100/model.ckpt",
        )

    def test_rejects_molecule_checkpoint_without_final_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._molecule_run(root, "config", "partial", 100)
            (run_dir / "train_results.json").unlink()

            with self.assertRaisesRegex(RuntimeError, "found 0 complete"):
                launcher._resolve_molecule_checkpoint(
                    task_root=root,
                    config_stem="config",
                    checkpoint_step=100,
                )

    def test_progen2_checkpoint_requires_nonempty_parseable_files(self):
        with tempfile.TemporaryDirectory() as directory:
            original_runs_root = launcher.RUNS_ROOT
            root = Path(directory)
            launcher.RUNS_ROOT = root
            checkpoint = (
                root
                / "progen2_sgrpo/config_slurm123/checkpoint-000100"
            )
            checkpoint.mkdir(parents=True)
            (checkpoint / "trainer_state.pt").write_bytes(b"state")
            (checkpoint / "model.safetensors").write_bytes(b"model")
            (checkpoint / "config.json").write_text(json.dumps({"model": "p2"}))
            try:
                resolved = launcher._resolve_progen2_checkpoint(
                    config_stem="config",
                    job_id=123,
                    checkpoint_step=100,
                )
            finally:
                launcher.RUNS_ROOT = original_runs_root

        self.assertEqual(resolved, checkpoint)

    def test_progen2_checkpoint_is_bound_to_requested_job(self):
        with tempfile.TemporaryDirectory() as directory:
            original_runs_root = launcher.RUNS_ROOT
            root = Path(directory)
            launcher.RUNS_ROOT = root
            stale = (
                root
                / "progen2_sgrpo/config_slurm111/checkpoint-000100"
            )
            requested = (
                root
                / "progen2_sgrpo/config_slurm222/checkpoint-000100"
            )
            for checkpoint in (stale, requested):
                checkpoint.mkdir(parents=True)
                (checkpoint / "trainer_state.pt").write_bytes(b"state")
                (checkpoint / "model.safetensors").write_bytes(b"model")
                (checkpoint / "config.json").write_text(
                    json.dumps({"model": "p2"})
                )
            try:
                resolved = launcher._resolve_progen2_checkpoint(
                    config_stem="config",
                    job_id=222,
                    checkpoint_step=100,
                )
            finally:
                launcher.RUNS_ROOT = original_runs_root

        self.assertEqual(resolved, requested)


if __name__ == "__main__":
    unittest.main()
