import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import drive_rebuttal_dense_sweeps as controller


class RebuttalSweepControllerTest(unittest.TestCase):
    def _state(self):
        return {
            "tasks": {
                "task": {
                    "status": "submitted",
                    "job_id": "123",
                    "array_id": None,
                    "submitted_at_epoch": 1.0,
                }
            }
        }

    def test_completed_job_gets_output_validation_grace(self):
        state = self._state()
        tasks = {
            "task": controller.TaskSpec(
                key="task",
                group="group",
                array_id=None,
                prerequisites=(),
                validator=lambda: False,
            )
        }
        accounting = {("123", None): ("COMPLETED", "0:0")}

        with (
            mock.patch.object(
                controller,
                "_accounting_states",
                return_value=accounting,
            ),
            mock.patch.object(controller.time, "time", return_value=100.0),
        ):
            controller._refresh_task_states(state, tasks, set())

        self.assertEqual(
            state["tasks"]["task"]["output_validation_grace_started_at_epoch"],
            100.0,
        )

        with (
            mock.patch.object(
                controller,
                "_accounting_states",
                return_value=accounting,
            ),
            mock.patch.object(
                controller.time,
                "time",
                return_value=100.0 + controller.COMPLETED_OUTPUT_GRACE_SECONDS,
            ),
            self.assertRaisesRegex(RuntimeError, "after 180s grace"),
        ):
            controller._refresh_task_states(state, tasks, set())

    def test_output_becoming_valid_completes_task_during_grace(self):
        state = self._state()
        output_is_valid = False
        tasks = {
            "task": controller.TaskSpec(
                key="task",
                group="group",
                array_id=None,
                prerequisites=(),
                validator=lambda: output_is_valid,
            )
        }
        accounting = {("123", None): ("COMPLETED", "0:0")}

        with (
            mock.patch.object(
                controller,
                "_accounting_states",
                return_value=accounting,
            ),
            mock.patch.object(controller.time, "time", return_value=100.0),
        ):
            controller._refresh_task_states(state, tasks, set())

        output_is_valid = True
        with (
            mock.patch.object(
                controller,
                "_accounting_states",
                return_value=accounting,
            ),
            mock.patch.object(controller.time, "time", return_value=160.0),
        ):
            controller._refresh_task_states(state, tasks, set())

        self.assertEqual(state["tasks"]["task"]["status"], "complete")

    def test_ready_tasks_wait_for_external_checkpoint(self):
        checkpoint_ready = False
        task = controller.TaskSpec(
            key="task",
            group="group",
            array_id=0,
            prerequisites=(),
            validator=lambda: False,
            readiness_validator=lambda: checkpoint_ready,
        )
        state = {"tasks": {}}

        self.assertEqual(controller._ready_tasks(state, {"task": task}), {})

        checkpoint_ready = True
        self.assertEqual(
            controller._ready_tasks(state, {"task": task}),
            {"group": [task]},
        )

    def test_molecule_checkpoint_requires_complete_final_state(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            checkpoint_dir = run_dir / "checkpoint-000100"
            checkpoint_dir.mkdir(parents=True)
            checkpoint = checkpoint_dir / "model.ckpt"
            checkpoint.write_bytes(b"model")
            (checkpoint_dir / "trainer_state.json").write_text(
                json.dumps({"global_step": 100})
            )
            (run_dir / "train_results.json").write_text(
                json.dumps({"step": 99})
            )

            self.assertFalse(controller._molecule_checkpoint_ready(checkpoint))

            (run_dir / "train_results.json").write_text(
                json.dumps({"step": 100})
            )
            self.assertTrue(controller._molecule_checkpoint_ready(checkpoint))

    def test_progen2_checkpoint_requires_complete_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint-000100"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text(
                json.dumps({"model_type": "progen"})
            )
            (checkpoint / "model.safetensors").write_bytes(b"model")

            self.assertFalse(controller._progen2_checkpoint_ready(checkpoint))

            (checkpoint / "trainer_state.pt").write_bytes(b"state")
            self.assertTrue(controller._progen2_checkpoint_ready(checkpoint))

    def test_gpu_task_selection_starts_from_persisted_cursor(self):
        groups = {
            name: controller.GroupSpec(
                name=name,
                resource="gpu",
                script=Path(f"{name}.sbatch"),
                job_name=name,
                output_pattern=Path(f"{name}.out"),
                error_pattern=Path(f"{name}.err"),
            )
            for name in ("first", "second", "third")
        }
        ready = {
            name: [
                controller.TaskSpec(
                    key=f"{name}:0",
                    group=name,
                    array_id=0,
                    prerequisites=(),
                    validator=lambda: False,
                )
            ]
            for name in groups
        }

        selected = controller._select_gpu_tasks(
            groups,
            ready,
            capacity=1,
            start_group="second",
        )

        self.assertEqual(list(selected), ["second"])

    def test_gpu_scheduler_advances_cursor_after_each_submission(self):
        groups = {
            name: controller.GroupSpec(
                name=name,
                resource="gpu",
                script=Path(f"{name}.sbatch"),
                job_name=name,
                output_pattern=Path(f"{name}.out"),
                error_pattern=Path(f"{name}.err"),
            )
            for name in ("first", "second", "third")
        }
        ready = {
            name: [
                controller.TaskSpec(
                    key=f"{name}:0",
                    group=name,
                    array_id=0,
                    prerequisites=(),
                    validator=lambda: False,
                )
            ]
            for name in groups
        }
        state = {"tasks": {}, "submissions": []}

        with (
            mock.patch.object(
                controller,
                "GPU_MAX_SUBMITTED_JOBS",
                1,
            ),
            mock.patch.object(
                controller,
                "_submit_group",
                return_value=True,
            ) as submit_group,
            mock.patch.object(controller, "_atomic_write_state"),
        ):
            controller._schedule_gpu_tasks(
                state,
                groups,
                ready,
                gpu_submitted_count=0,
            )
            controller._schedule_gpu_tasks(
                state,
                groups,
                ready,
                gpu_submitted_count=0,
            )

        submitted_names = [
            call.args[1].name for call in submit_group.call_args_list
        ]
        self.assertEqual(submitted_names, ["first", "second"])
        self.assertEqual(state["gpu_group_cursor"], "third")


if __name__ == "__main__":
    unittest.main()
