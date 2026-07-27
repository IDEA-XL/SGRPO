import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import progen2_sweep_pipeline as pipeline


class ProGen2SweepIncrementalTest(unittest.TestCase):
    def _write_config_and_tasks(self, directory):
        root = Path(directory)
        official_code_dir = root / "official"
        official_code_dir.mkdir()
        tokenizer_path = root / "tokenizer.json"
        tokenizer_path.write_text("{}")
        prompt_path = root / "prompts.txt"
        prompt_path.write_text("1\\n")
        ready_checkpoint = root / "ready-checkpoint"
        ready_checkpoint.mkdir()
        missing_checkpoint = root / "missing-checkpoint"
        tasks_path = root / "tasks.tsv"
        output_path = root / "generated.jsonl"

        payload = {
            "tasks_path": str(tasks_path),
            "generation_output_root": str(root / "generation"),
            "foldability_output_root": str(root / "foldability"),
            "developability_output_root": str(root / "developability"),
            "diversity_output_root": str(root / "diversity"),
            "packed_naturalness_scores_path": str(root / "naturalness.jsonl"),
            "packed_stability_scores_path": str(root / "stability.jsonl"),
            "output_markdown_path": str(root / "results.md"),
            "output_json_path": str(root / "results.json"),
            "output_rows_path": str(root / "rows.jsonl"),
            "output_naturalness_diversity_plot_path": str(root / "naturalness.png"),
            "output_foldability_diversity_plot_path": str(root / "foldability.png"),
            "output_stability_diversity_plot_path": str(root / "stability.png"),
            "output_developability_diversity_plot_path": str(root / "developability.png"),
            "output_soft_reward_diversity_plot_path": str(root / "soft_reward.png"),
            "official_code_dir": str(official_code_dir),
            "tokenizer_path": str(tokenizer_path),
            "prompt_path": str(prompt_path),
            "rewards": {},
            "num_samples": 1,
            "generation_prompt_batch_size": 1,
            "num_return_sequences": 1,
            "temperature_values": [0.8],
            "experiments": [
                {
                    "name": "pending",
                    "checkpoint_dir": str(missing_checkpoint),
                    "naturalness": 0.25,
                    "foldability": 0.3,
                    "stability": 0.2,
                    "developability": 0.25,
                },
                {
                    "name": "ready",
                    "checkpoint_dir": str(ready_checkpoint),
                    "naturalness": 0.25,
                    "foldability": 0.3,
                    "stability": 0.2,
                    "developability": 0.25,
                },
            ],
        }
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(payload))

        task = {
            "task_id": 1,
            "experiment": "ready",
            "display_name": "Ready",
            "checkpoint_dir": str(ready_checkpoint),
            "checkpoint_subdir": "",
            "naturalness_weight": 0.25,
            "foldability_weight": 0.3,
            "stability_weight": 0.2,
            "developability_weight": 0.25,
            "temperature": 0.8,
            "generation_rows_path": str(output_path),
            "foldability_scores_path": str(root / "foldability.jsonl"),
            "developability_scores_path": str(root / "developability.jsonl"),
            "diversity_scores_path": str(root / "diversity.json"),
        }
        with tasks_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=pipeline.POINT_TASK_FIELDNAMES,
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerow(task)
        return config_path, output_path

    def test_generate_task_ignores_unrelated_pending_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path, output_path = self._write_config_and_tasks(directory)
            generated = {
                "sample_index": 0,
                "prompt_text": "1",
                "decoded_text": "A",
                "raw_sequence": "A",
                "sequence": "A",
                "is_valid": True,
                "invalid_reason": None,
            }

            with (
                mock.patch.object(pipeline, "load_prompt_texts", return_value=["1"]),
                mock.patch.object(pipeline, "resolve_device", return_value=object()),
                mock.patch.object(pipeline, "_instantiate_policy", return_value=object()),
                mock.patch.object(pipeline, "_generate_rows", return_value=[generated]),
            ):
                pipeline.cmd_generate_task(
                    SimpleNamespace(config=str(config_path), task_id=1)
                )

            rows = [
                json.loads(line)
                for line in output_path.read_text().splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["experiment"], "ready")

    def test_default_config_load_still_requires_every_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path, _ = self._write_config_and_tasks(directory)

            with self.assertRaisesRegex(
                FileNotFoundError,
                "missing-checkpoint",
            ):
                pipeline.load_config(config_path)


if __name__ == "__main__":
    unittest.main()
