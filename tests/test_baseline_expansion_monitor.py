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
    def test_running_resume_resets_stale_progress_window(self):
        updated = monitor._updated_progress_epoch(
            previous_state='PENDING',
            current_state='RUNNING',
            previous_step=450,
            current_step=450,
            previous_epoch=100.0,
            now=1000.0,
        )

        self.assertEqual(updated, 1000.0)

    def test_unchanged_running_job_keeps_stale_progress_window(self):
        updated = monitor._updated_progress_epoch(
            previous_state='RUNNING',
            current_state='RUNNING',
            previous_step=450,
            current_step=450,
            previous_epoch=100.0,
            now=1000.0,
        )

        self.assertEqual(updated, 100.0)

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

    def test_running_job_ignores_incomplete_final_record(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / 'metrics.jsonl'
            base = {
                'reward_mean': 0.5,
                'grad_norm': 0.25,
                'entropy/normalized_mean': 0.75,
            }
            _write_metrics(metrics_path, [{'step': 1, **base}])
            with metrics_path.open('a') as handle:
                handle.write('{"step": 2,\n')

            result = monitor._read_metrics(
                self._entropy_spec(metrics_path),
                allow_partial_tail=True,
            )

        self.assertEqual(result['max_step'], 1)

    def test_terminal_job_rejects_incomplete_final_record(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / 'metrics.jsonl'
            base = {
                'reward_mean': 0.5,
                'grad_norm': 0.25,
                'entropy/normalized_mean': 0.75,
            }
            _write_metrics(metrics_path, [{'step': 1, **base}])
            with metrics_path.open('a') as handle:
                handle.write('{"step": 2,\n')

            with self.assertRaisesRegex(RuntimeError, 'Invalid JSON'):
                monitor._read_metrics(self._entropy_spec(metrics_path))

    def test_running_job_rejects_invalid_nonfinal_record(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / 'metrics.jsonl'
            base = {
                'reward_mean': 0.5,
                'grad_norm': 0.25,
                'entropy/normalized_mean': 0.75,
            }
            _write_metrics(metrics_path, [{'step': 1, **base}])
            with metrics_path.open('a') as handle:
                handle.write('{"step": 2,\n')
                handle.write(json.dumps({'step': 3, **base}) + '\n')

            with self.assertRaisesRegex(RuntimeError, 'Invalid JSON'):
                monitor._read_metrics(
                    self._entropy_spec(metrics_path),
                    allow_partial_tail=True,
                )

    def test_validated_completed_job_ignores_historical_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            stderr_path = Path(directory) / 'job.err'
            stderr_path.write_text('Traceback (most recent call last):\n')
            record = {
                'state': 'COMPLETED',
                'exit_code': '0:0',
                'stdout': '',
                'stderr': str(stderr_path),
            }
            metrics = {'max_step': 2000}

            errors = monitor._actionable_log_errors(
                record,
                metrics,
                self._entropy_spec(Path(directory) / 'metrics.jsonl'),
            )

        self.assertEqual(errors, [])

    def test_running_job_reports_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            stderr_path = Path(directory) / 'job.err'
            stderr_path.write_text('Traceback (most recent call last):\n')
            record = {
                'state': 'RUNNING',
                'exit_code': '0:0',
                'stdout': '',
                'stderr': str(stderr_path),
            }
            metrics = {'max_step': 100}

            errors = monitor._actionable_log_errors(
                record,
                metrics,
                self._entropy_spec(Path(directory) / 'metrics.jsonl'),
            )

        self.assertEqual(errors, [r'Traceback \(most recent call last\)'])

    def test_molecular_dmb_valid_candidate_shortfall_is_accepted(self):
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

            result = monitor._read_metrics(
                self._molecular_dmb_spec(metrics_path)
            )

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
                        'diverse_minibatch/selected_valid_count': 191.75,
                        'diverse_minibatch/selected_invalid_count': 0.25,
                        'diverse_minibatch/target_optimization_count': 192.0,
                        'diverse_minibatch/shortfall_count': 0.0,
                    }
                ],
            )

            result = monitor._read_metrics(self._dmb_spec(metrics_path))

        self.assertEqual(result['max_step'], 1)

    def test_progen2_dmb_zero_valid_candidates_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / 'metrics.jsonl'
            _write_metrics(
                metrics_path,
                [
                    {
                        'step': 39,
                        'reward_mean': 0.0,
                        'grad_norm': 0.0,
                        'diverse_minibatch/candidate_count': 384,
                        'diverse_minibatch/valid_candidate_count': 0,
                        'diverse_minibatch/selected_count': 192,
                        'diverse_minibatch/selected_valid_count': 0,
                        'diverse_minibatch/selected_invalid_count': 192,
                        'diverse_minibatch/target_optimization_count': 192,
                        'diverse_minibatch/shortfall_count': 0,
                    }
                ],
            )

            with self.assertRaisesRegex(
                RuntimeError,
                'no valid selected candidates',
            ):
                monitor._read_metrics(self._dmb_spec(metrics_path))

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
                monitor._read_metrics(
                    self._molecular_dmb_spec(metrics_path)
                )

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
            selection_includes_invalid_candidates=True,
        )

    @staticmethod
    def _molecular_dmb_spec(metrics_path):
        return monitor.JobSpec(
            name='denovo_dmb',
            job_id=1,
            metrics_glob=str(metrics_path),
            method='dmb',
            expected_final_step=2000,
            expected_candidate_count=384,
            expected_selected_count=192,
        )


if __name__ == '__main__':
    unittest.main()
