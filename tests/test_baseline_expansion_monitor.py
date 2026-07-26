import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'scripts'
    / 'monitor_baseline_expansion.py'
)
SPEC = importlib.util.spec_from_file_location(
    'monitor_baseline_expansion',
    MODULE_PATH,
)
monitor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = monitor
SPEC.loader.exec_module(monitor)


def _write_metrics(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


class BaselineExpansionMonitorTest(unittest.TestCase):
    def test_molecular_metrics_do_not_require_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / 'metrics.jsonl'
            base = {
                'reward_mean': 0.5,
                'grad_norm': 0.25,
                'entropy/normalized_mean': 0.75,
            }
            _write_metrics(
                metrics_path,
                [
                    {'step': 1, **base},
                    {'step': 10, **base},
                ],
            )
            result = monitor._read_metrics(self._entropy_spec(metrics_path))

        self.assertEqual(result['max_step'], 10)
        self.assertTrue(result['first_ten_verified'])

    def test_nonfinite_grad_norm_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / 'metrics.jsonl'
            _write_metrics(
                metrics_path,
                [
                    {
                        'step': 1,
                        'reward_mean': 0.5,
                        'grad_norm': float('nan'),
                        'entropy/normalized_mean': 0.75,
                    }
                ],
            )
            with self.assertRaisesRegex(RuntimeError, 'non-finite grad_norm'):
                monitor._read_metrics(self._entropy_spec(metrics_path))

    def test_dmb_valid_candidate_shortfall_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / 'metrics.jsonl'
            _write_metrics(
                metrics_path,
                [
                    {
                        'step': 1,
                        'reward_mean': 0.5,
                        'grad_norm': 0.25,
                        'diverse_minibatch/candidate_count': 384,
                        'diverse_minibatch/valid_candidate_count': 182,
                        'diverse_minibatch/selected_count': 182,
                        'diverse_minibatch/target_optimization_count': 192,
                        'diverse_minibatch/shortfall_count': 10,
                    }
                ],
            )

            result = monitor._read_metrics(self._dmb_spec(metrics_path))

        self.assertEqual(result['max_step'], 1)

    def test_dmb_averaged_fractional_counts_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / 'metrics.jsonl'
            _write_metrics(
                metrics_path,
                [
                    {
                        'step': 1,
                        'reward_mean': 0.5,
                        'grad_norm': 0.25,
                        'diverse_minibatch/candidate_count': 384.0,
                        'diverse_minibatch/valid_candidate_count': 383.75,
                        'diverse_minibatch/selected_count': 192.0,
                        'diverse_minibatch/target_optimization_count': 192.0,
                        'diverse_minibatch/shortfall_count': 0.0,
                    }
                ],
            )

            result = monitor._read_metrics(self._dmb_spec(metrics_path))

        self.assertEqual(result['max_step'], 1)

    def test_dmb_inconsistent_shortfall_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / 'metrics.jsonl'
            _write_metrics(
                metrics_path,
                [
                    {
                        'step': 1,
                        'reward_mean': 0.5,
                        'grad_norm': 0.25,
                        'diverse_minibatch/candidate_count': 384,
                        'diverse_minibatch/valid_candidate_count': 182,
                        'diverse_minibatch/selected_count': 182,
                        'diverse_minibatch/target_optimization_count': 192,
                        'diverse_minibatch/shortfall_count': 0,
                    }
                ],
            )

            with self.assertRaisesRegex(RuntimeError, 'inconsistent shortfall_count'):
                monitor._read_metrics(self._dmb_spec(metrics_path))

    @staticmethod
    def _entropy_spec(metrics_path):
        return monitor.JobSpec(
            name='denovo_entropy',
            job_id=1,
            metrics_glob=str(metrics_path),
            method='entropy',
            expected_final_step=2000,
        )

    @staticmethod
    def _dmb_spec(metrics_path):
        return monitor.JobSpec(
            name='progen2_dmb',
            job_id=1,
            metrics_glob=str(metrics_path),
            method='dmb',
            expected_final_step=100,
            expected_candidate_count=384,
            expected_selected_count=192,
        )


if __name__ == '__main__':
    unittest.main()
