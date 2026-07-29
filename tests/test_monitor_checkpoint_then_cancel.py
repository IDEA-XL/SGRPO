import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import monitor_checkpoint_then_cancel as monitor  # noqa: E402


def _write_checkpoint(root: Path, step: int, ranks: int) -> Path:
    checkpoint = root / f"checkpoint-{step:06d}"
    accelerator = checkpoint / "accelerator_state"
    model_state = accelerator / "pytorch_model"
    model_state.mkdir(parents=True)
    (checkpoint / "model.ckpt").write_bytes(b"model")
    (checkpoint / "reference_backbone.pt").write_bytes(b"reference")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step})
    )
    (accelerator / "latest").write_text("pytorch_model")
    (accelerator / "scheduler.bin").write_bytes(b"x")
    (accelerator / "zero_to_fp32.py").write_text("pass\n")
    (model_state / "mp_rank_00_model_states.pt").write_bytes(b"x")
    for rank in range(ranks):
        (accelerator / f"random_states_{rank}.pkl").write_bytes(b"x")
        (
            model_state
            / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        ).write_bytes(b"x")
    return checkpoint


class MonitorCheckpointThenCancelTest(unittest.TestCase):
    def test_complete_checkpoint_has_signature(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint = _write_checkpoint(Path(temporary_dir), 1000, 2)
            signature = monitor._checkpoint_signature(
                checkpoint,
                target_step=1000,
                expected_ranks=2,
                min_model_bytes=1,
                min_reference_bytes=1,
            )
            self.assertIsNotNone(signature)

    def test_incomplete_checkpoint_returns_none(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint = _write_checkpoint(Path(temporary_dir), 1000, 2)
            (
                checkpoint
                / "accelerator_state/random_states_1.pkl"
            ).unlink()
            self.assertIsNone(
                monitor._checkpoint_signature(
                    checkpoint,
                    target_step=1000,
                    expected_ranks=2,
                    min_model_bytes=1,
                    min_reference_bytes=1,
                )
            )

    def test_wrong_step_fails_fast(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint = _write_checkpoint(Path(temporary_dir), 999, 2)
            with self.assertRaisesRegex(RuntimeError, "step mismatch"):
                monitor._checkpoint_signature(
                    checkpoint,
                    target_step=1000,
                    expected_ranks=2,
                    min_model_bytes=1,
                    min_reference_bytes=1,
                )


if __name__ == "__main__":
    unittest.main()
