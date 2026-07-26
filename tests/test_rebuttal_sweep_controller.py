import sys
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


if __name__ == "__main__":
    unittest.main()
