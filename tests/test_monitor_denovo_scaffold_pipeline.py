import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))
MODULE_PATH = SCRIPTS_ROOT / "monitor_denovo_scaffold_pipeline.py"
SPEC = importlib.util.spec_from_file_location(
    "monitor_denovo_scaffold_pipeline_for_test",
    MODULE_PATH,
)
monitor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = monitor
SPEC.loader.exec_module(monitor)


def _metric(step):
    return {
        "step": step,
        "reward_mean": 0.7,
        "grad_norm": 0.1,
        "group_reward/diversity_mean": 0.9,
        "rewards/soft_mean": 0.8,
        "valid_fraction": 0.99,
    }


class MonitorDenovoScaffoldPipelineTest(unittest.TestCase):
    def test_reads_and_validates_scaffold_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(
                json.dumps(_metric(1)) + "\n" + json.dumps(_metric(10)) + "\n"
            )

            result = monitor._read_metrics(path, allow_partial_tail=False)

        self.assertEqual(result["max_step"], 10)
        self.assertTrue(result["first_ten_verified"])

    def test_rejects_out_of_range_scaffold_diversity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            row = _metric(1)
            row["group_reward/diversity_mean"] = 1.1
            path.write_text(json.dumps(row) + "\n")

            with self.assertRaisesRegex(RuntimeError, "outside"):
                monitor._read_metrics(path, allow_partial_tail=False)

    def test_running_job_ignores_partial_final_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text(json.dumps(_metric(1)) + "\n{\"step\": 2,\n")

            result = monitor._read_metrics(path, allow_partial_tail=True)

        self.assertEqual(result["max_step"], 1)

    def test_running_job_retries_transient_interior_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text("placeholder\n")
            partial_snapshot = (
                json.dumps(_metric(1)) + "\n"
                + "temporarily unreadable\n"
                + json.dumps(_metric(2)) + "\n"
            )
            complete_snapshot = (
                json.dumps(_metric(1)) + "\n"
                + json.dumps(_metric(2)) + "\n"
            )
            with (
                mock.patch.object(
                    Path,
                    "read_text",
                    side_effect=[partial_snapshot, complete_snapshot],
                ),
                mock.patch.object(monitor.time, "sleep") as sleep,
            ):
                result = monitor._read_metrics(path, allow_partial_tail=True)

        self.assertEqual(result["max_step"], 2)
        sleep.assert_called_once_with(monitor.METRICS_READ_RETRY_SECONDS)

    def test_running_job_rejects_persistent_interior_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text("placeholder\n")
            invalid_snapshot = (
                json.dumps(_metric(1)) + "\n"
                + "persistently invalid\n"
                + json.dumps(_metric(2)) + "\n"
            )
            with (
                mock.patch.object(
                    Path,
                    "read_text",
                    return_value=invalid_snapshot,
                ),
                mock.patch.object(monitor.time, "sleep") as sleep,
            ):
                with self.assertRaisesRegex(RuntimeError, "Invalid JSON"):
                    monitor._read_metrics(path, allow_partial_tail=True)

        self.assertEqual(sleep.call_count, monitor.METRICS_READ_ATTEMPTS - 1)

    def test_discovers_both_pipeline_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "launcher.out"
            output.write_text("controller_job_id=123\nrender_job_id=456\n")
            controller, render = monitor._discover_pipeline_jobs(
                {"stdout": str(output)}
            )

        self.assertEqual(controller, 123)
        self.assertEqual(render, 456)

    def test_rejects_partial_pipeline_job_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "launcher.out"
            output.write_text("controller_job_id=123\n")
            with self.assertRaisesRegex(RuntimeError, "only one"):
                monitor._discover_pipeline_jobs({"stdout": str(output)})


if __name__ == "__main__":
    unittest.main()
