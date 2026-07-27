import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


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

    def test_point_reward_validator_matches_only_valid_generation_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation_path = root / "generation.jsonl"
            reward_path = root / "developability.jsonl"
            generation_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"sample_index": 0, "is_valid": True},
                        {"sample_index": 1, "is_valid": False},
                        {"sample_index": 2, "is_valid": True},
                    )
                )
            )
            reward_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"sample_index": 0, "developability": 0.8},
                        {"sample_index": 2, "developability": 0.6},
                    )
                )
            )

            self.assertTrue(
                controller._point_reward_output_valid(
                    generation_path,
                    reward_path,
                    "developability",
                    expected_generation_rows=3,
                )
            )

            reward_path.write_text(
                json.dumps(
                    {"sample_index": 0, "developability": 0.8}
                )
            )
            self.assertFalse(
                controller._point_reward_output_valid(
                    generation_path,
                    reward_path,
                    "developability",
                    expected_generation_rows=3,
                )
            )

    def test_point_reward_validator_accepts_empty_output_when_all_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation_path = root / "generation.jsonl"
            reward_path = root / "foldability.jsonl"
            generation_path.write_text(
                "\n".join(
                    json.dumps(
                        {"sample_index": index, "is_valid": False}
                    )
                    for index in range(3)
                )
            )
            reward_path.write_text("")

            self.assertTrue(
                controller._point_reward_output_valid(
                    generation_path,
                    reward_path,
                    "foldability",
                    expected_generation_rows=3,
                )
            )

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

    def test_split_packed_rewards_depend_only_on_same_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_root = root / "specs"
            denovo_config = spec_root / "denovo/config.yaml"
            denovo_config.parent.mkdir(parents=True)
            denovo_config.write_text(
                yaml.safe_dump(
                    {
                        "output_json_path": str(root / "denovo.json"),
                        "randomness_temperature_pairs": [
                            {
                                "randomness": 0.1,
                                "generation_temperature": 0.5,
                            }
                        ],
                        "experiments": [
                            {
                                "checkpoint_path": str(
                                    root / "denovo-checkpoint/model.ckpt"
                                )
                            }
                        ],
                    }
                )
            )
            with (spec_root / "denovo/tasks.tsv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("task_id", "seed", "config_path"),
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": 0,
                        "seed": 42,
                        "config_path": denovo_config,
                    }
                )

            mm_path = spec_root / "mmgenmol/seed42.tsv"
            mm_path.parent.mkdir(parents=True)
            with mm_path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "task_id",
                        "model_name",
                        "sweep_type",
                        "sweep_value",
                        "checkpoint_path",
                        "output_path",
                    ),
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": 0,
                        "model_name": "model",
                        "sweep_type": "paired",
                        "sweep_value": 1,
                        "checkpoint_path": root / "mm-checkpoint/model.ckpt",
                        "output_path": root / "mm-generated.jsonl",
                    }
                )

            progen2_config_path = spec_root / "progen2/seed42.yaml"
            progen2_config_path.parent.mkdir(parents=True)
            progen2_config = {
                "packed_naturalness_scores_path": str(
                    root / "naturalness/naturalness.rows.jsonl"
                ),
                "packed_stability_scores_path": str(
                    root / "stability/stability.rows.jsonl"
                ),
                "output_json_path": str(root / "progen2.json"),
                "experiments": [
                    {"name": "dmb"},
                    {"name": "entropy"},
                ],
            }
            progen2_config_path.write_text(yaml.safe_dump(progen2_config))
            p2_tasks_path = spec_root / "progen2/seed42_tasks.tsv"
            with p2_tasks_path.open("w", newline="") as handle:
                fieldnames = (
                    "task_id",
                    "experiment",
                    "checkpoint_dir",
                    "generation_rows_path",
                    "foldability_scores_path",
                    "developability_scores_path",
                    "diversity_scores_path",
                )
                writer = csv.DictWriter(
                    handle,
                    fieldnames=fieldnames,
                    delimiter="\t",
                )
                writer.writeheader()
                for task_id, experiment in enumerate(("dmb", "entropy")):
                    writer.writerow(
                        {
                            "task_id": task_id,
                            "experiment": experiment,
                            "checkpoint_dir": root / f"{experiment}-checkpoint",
                            "generation_rows_path": root
                            / f"{experiment}-generated.jsonl",
                            "foldability_scores_path": root
                            / f"{experiment}-foldability.jsonl",
                            "developability_scores_path": root
                            / f"{experiment}-developability.jsonl",
                            "diversity_scores_path": root
                            / f"{experiment}-diversity.json",
                        }
                    )

            with (
                mock.patch.object(controller, "SEEDS", (42,)),
                mock.patch.object(controller, "SPEC_ROOT", spec_root),
                mock.patch.object(controller, "RUN_OUTPUT_ROOT", root),
                mock.patch.object(controller, "LOG_ROOT", root / "logs"),
                mock.patch.object(controller, "REPO_REMOTE_ROOT", root / "repo"),
                mock.patch.dict(
                    "os.environ",
                    {controller.PROGEN2_PACKED_BY_EXPERIMENT_ENV: "1"},
                ),
            ):
                groups, tasks, _ = controller._build_dag()

            self.assertEqual(
                tasks["p2:42:naturalness:dmb"].prerequisites,
                ("p2:42:generation:0",),
            )
            self.assertEqual(
                tasks["p2:42:naturalness:entropy"].prerequisites,
                ("p2:42:generation:1",),
            )
            self.assertEqual(
                set(tasks["p2:42:packed_merge"].prerequisites),
                {
                    "p2:42:naturalness:dmb",
                    "p2:42:stability:dmb",
                    "p2:42:naturalness:entropy",
                    "p2:42:stability:entropy",
                },
            )
            self.assertEqual(
                groups["progen2_naturalness_42_entropy"].exports[-1],
                ("EXPERIMENT_NAME", "entropy"),
            )
            self.assertNotIn("p2:42:naturalness", tasks)


if __name__ == "__main__":
    unittest.main()
