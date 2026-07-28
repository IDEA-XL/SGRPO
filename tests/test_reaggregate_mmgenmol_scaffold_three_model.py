import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts/reaggregate_mmgenmol_scaffold_three_model.py"
SPEC = importlib.util.spec_from_file_location(
    "reaggregate_mmgenmol_scaffold_three_model_for_test",
    MODULE_PATH,
)
reaggregate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = reaggregate
SPEC.loader.exec_module(reaggregate)


class ReaggregateMmgenmolScaffoldThreeModelTest(unittest.TestCase):
    SEED = 7
    SWEEP = ((0.1, 0.5), (0.2, 0.65))
    SOURCES = (
        reaggregate.ModelSource("original", False),
        reaggregate.ModelSource("unidock", True),
    )

    @staticmethod
    def _generated_rows():
        return [
            {"source_index": 10, "smiles": "c1ccccc1"},
            {"source_index": 10, "smiles": "Cc1ccccc1"},
            {"source_index": 20, "smiles": "C1CCCCC1"},
            {"source_index": 20, "smiles": "CC1CCCCC1"},
        ]

    @staticmethod
    def _docking_rows():
        rows = []
        for row_idx, (source_index, success) in enumerate(
            ((10, True), (10, False), (20, True), (20, True))
        ):
            record = {"is_success": success}
            if success:
                record.update(
                    {
                        "dock_affinity": -7.0 - row_idx,
                        "score_only_affinity": -6.0 - row_idx,
                        "minimize_affinity": -6.5 - row_idx,
                    }
                )
            rows.append(
                {
                    "row_idx": row_idx,
                    "source_index": source_index,
                    "record": record,
                }
            )
        return rows

    @staticmethod
    def _mean(values):
        return sum(values) / len(values)

    @classmethod
    def _summary(cls, model, task_id, randomness, temperature, generated, docking, uses_unidock):
        valid = []
        dock_values = []
        score_values = []
        min_values = []
        success_flags = []
        for row_idx, row in enumerate(generated):
            record = docking[row_idx]["record"]
            final_valid = not uses_unidock or record["is_success"]
            if not final_valid:
                continue
            valid.append(row["smiles"])
            success_flags.append(record["is_success"])
            if record["is_success"]:
                dock_values.append(record["dock_affinity"])
                score_values.append(record["score_only_affinity"])
                min_values.append(record["minimize_affinity"])
        ordered = sorted(dock_values)
        median = ordered[len(ordered) // 2] if len(ordered) % 2 else (
            ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]
        ) / 2
        return {
            "task_id": task_id,
            "model_name": model,
            "sweep_type": "paired",
            "sweep_value": float(task_id + 1),
            "randomness": randomness,
            "temperature": temperature,
            "checkpoint_path": f"/checkpoint/{model}",
            "reward_weights": {
                "qed": 0.3 if uses_unidock else 0.6,
                "sa_score": 0.2 if uses_unidock else 0.4,
                "drugclip_score": 0.0,
                "unidock_score": 0.5 if uses_unidock else 0.0,
            },
            "num_rows": 4,
            "num_pockets": 2,
            "samples_per_pocket": 2,
            "valid_count": len(valid),
            "valid_fraction": len(valid) / 4,
            "unique_valid_count": len(set(valid)),
            "duplicate_fraction": 1.0 - len(set(valid)) / len(valid),
            "qed_mean": 0.5,
            "sa_mean": 2.0,
            "sa_score_mean": 0.8,
            "soft_reward_mean": 0.7,
            "diversity": 0.5,
            "vina_dock_success_fraction": sum(success_flags) / len(success_flags),
            "vina_dock_num_docked": len(dock_values),
            "vina_dock_mean": cls._mean(dock_values),
            "vina_dock_median": median,
            "vina_score_mean": cls._mean(score_values),
            "vina_min_mean": cls._mean(min_values),
        }

    def _build_inputs(self, root):
        run_root = root / "run"
        task_rows = []
        summaries = []
        task_id = 0
        for source in self.SOURCES:
            for randomness, temperature in self.SWEEP:
                generated = self._generated_rows()
                docking = self._docking_rows()
                generated_path = (
                    run_root
                    / "mmgenmol"
                    / f"seed{self.SEED}"
                    / "generation"
                    / source.model_name
                    / f"paired_{task_id + 1}"
                    / "generated.rows.jsonl"
                )
                docking_path = (
                    run_root
                    / "mmgenmol"
                    / f"seed{self.SEED}"
                    / "docking"
                    / source.model_name
                    / f"paired_{task_id + 1}"
                    / "docking.records.jsonl"
                )
                generated_path.parent.mkdir(parents=True)
                docking_path.parent.mkdir(parents=True)
                generated_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in generated)
                )
                docking_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in docking)
                )
                task_rows.append(
                    {
                        "task_id": task_id,
                        "model_name": source.model_name,
                        "sweep_type": "paired",
                        "sweep_value": task_id + 1,
                        "randomness": randomness,
                        "temperature": temperature,
                        "checkpoint_path": f"/checkpoint/{source.model_name}",
                        "output_path": generated_path,
                    }
                )
                summary = self._summary(
                    source.model_name,
                    task_id,
                    randomness,
                    temperature,
                    generated,
                    docking,
                    source.uses_unidock_reward,
                )
                summary["generated_rows_path"] = str(generated_path)
                summary["docking_records_path"] = str(docking_path)
                summaries.append(summary)
                task_id += 1

        task_path = run_root / f"specs/mmgenmol/seed{self.SEED}.tsv"
        task_path.parent.mkdir(parents=True)
        with task_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=task_rows[0].keys(),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(task_rows)
        summary_path = (
            run_root
            / "mmgenmol"
            / f"seed{self.SEED}"
            / "aggregate/mmgenmol_dense.json"
        )
        summary_path.parent.mkdir(parents=True)
        summary_path.write_text(json.dumps(summaries))
        return run_root

    def test_reuses_utility_and_recomputes_per_pocket_scaffold_diversity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = self._build_inputs(root)
            output_root = root / "output"

            manifest_path = reaggregate.reaggregate(
                run_root=run_root,
                output_root=output_root,
                seeds=(self.SEED,),
                sweep=self.SWEEP,
                model_sources=self.SOURCES,
                expected_num_pockets=2,
                samples_per_pocket=2,
            )

            manifest = json.loads(manifest_path.read_text())
            rows = json.loads(
                (output_root / f"mmgenmol/seed{self.SEED}.json").read_text()
            )
            self.assertEqual(manifest["total_generated_rows"], 16)
            self.assertEqual(manifest["total_docking_rows"], 16)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["soft_reward_mean"] == 0.7 for row in rows))
            self.assertTrue(
                all(
                    row["diversity_metric"]
                    == reaggregate.RELATIVE_SCAFFOLD_DIVERSITY
                    for row in rows
                )
            )
            diversity_by_model = {
                row["model_name"]: row["diversity"] for row in rows
            }
            self.assertEqual(diversity_by_model["original"], 0.5)
            self.assertEqual(diversity_by_model["unidock"], 0.75)

    def test_rejects_source_valid_count_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = self._build_inputs(root)
            summary_path = (
                run_root
                / "mmgenmol"
                / f"seed{self.SEED}"
                / "aggregate/mmgenmol_dense.json"
            )
            summaries = json.loads(summary_path.read_text())
            summaries[0]["valid_count"] -= 1
            summary_path.write_text(json.dumps(summaries))

            with self.assertRaisesRegex(ValueError, "valid_count mismatch"):
                reaggregate.reaggregate(
                    run_root=run_root,
                    output_root=root / "output",
                    seeds=(self.SEED,),
                    sweep=self.SWEEP,
                    model_sources=self.SOURCES,
                    expected_num_pockets=2,
                    samples_per_pocket=2,
                )

    def test_rejects_duplicate_docking_row_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = self._build_inputs(root)
            docking_path = next(
                (
                    run_root
                    / "mmgenmol"
                    / f"seed{self.SEED}"
                    / "docking"
                ).glob("**/docking.records.jsonl")
            )
            rows = [json.loads(line) for line in docking_path.read_text().splitlines()]
            rows[1]["row_idx"] = rows[0]["row_idx"]
            docking_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )

            with self.assertRaisesRegex(ValueError, "duplicate docking row_idx"):
                reaggregate.reaggregate(
                    run_root=run_root,
                    output_root=root / "output",
                    seeds=(self.SEED,),
                    sweep=self.SWEEP,
                    model_sources=self.SOURCES,
                    expected_num_pockets=2,
                    samples_per_pocket=2,
                )


if __name__ == "__main__":
    unittest.main()
