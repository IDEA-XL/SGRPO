import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import drive_motif_extension_pipeline as pipeline  # noqa: E402


def _write_checkpoint(run_dir: Path, step: int) -> Path:
    checkpoint_dir = run_dir / f"checkpoint-{step:06d}"
    checkpoint_dir.mkdir(parents=True)
    model_path = checkpoint_dir / "model.ckpt"
    model_path.write_bytes(b"checkpoint")
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps({"global_step": step})
    )
    (run_dir / "train_results.json").write_text(
        json.dumps({"step": step})
    )
    return model_path


class MotifExtensionPipelineTest(unittest.TestCase):
    def test_validate_checkpoint_requires_consistent_final_step(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir)
            model_path = _write_checkpoint(run_dir, 2000)
            self.assertEqual(
                pipeline._validate_checkpoint(run_dir, 2000),
                model_path,
            )

            (run_dir / "train_results.json").write_text(
                json.dumps({"step": 1999})
            )
            with self.assertRaisesRegex(RuntimeError, "Train results"):
                pipeline._validate_checkpoint(run_dir, 2000)

    def test_metrics_require_finite_motif_fractions(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            metrics_path = Path(temporary_dir) / "metrics.jsonl"
            row = {
                "step": 1,
                "reward_mean": 0.5,
                "grad_norm": 0.1,
                "valid_fraction": 0.9,
                "motif_extension/candidate_valid_fraction": 0.95,
                "motif_extension/candidate_retention_fraction": 0.8,
            }
            metrics_path.write_text(json.dumps(row) + "\n")
            metrics = pipeline._read_jsonl_metrics(metrics_path, 2)
            self.assertEqual(metrics["max_step"], 1)

            row["motif_extension/candidate_retention_fraction"] = 1.1
            metrics_path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(
                RuntimeError,
                r"outside \[0, 1\]",
            ):
                pipeline._read_jsonl_metrics(metrics_path, 2)

    def test_summary_validator_checks_hash_and_shape(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "result"
            output_dir.mkdir()
            rows_path = output_dir / "rows.jsonl"
            rows_path.write_text('{"sample_index": 0}\n')
            digest = hashlib.sha256(rows_path.read_bytes()).hexdigest()
            (output_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "experiment": "motif_grpo_2000",
                            "seed": 42,
                            "row_count": 10_000,
                            "motif_count": 10,
                            "samples_per_motif": 100,
                            "rows_sha256": digest,
                        },
                        "results": [{} for _ in range(10)],
                    }
                )
            )
            task = {
                "experiment": "motif_grpo_2000",
                "seed": 42,
                "output_dir": str(output_dir),
            }
            self.assertTrue(pipeline._summary_valid(task))

            rows_path.write_text('{"sample_index": 1}\n')
            self.assertFalse(pipeline._summary_valid(task))

if __name__ == "__main__":
    unittest.main()
