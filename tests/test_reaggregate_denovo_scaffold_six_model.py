import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts/reaggregate_denovo_scaffold_six_model.py"
SPEC = importlib.util.spec_from_file_location(
    "reaggregate_denovo_scaffold_six_model_for_test",
    MODULE_PATH,
)
reaggregate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = reaggregate
SPEC.loader.exec_module(reaggregate)


class ReaggregateDenovoScaffoldSixModelTest(unittest.TestCase):
    SEED = 7
    SWEEP = ((0.1, 0.5), (0.2, 0.65))
    SOURCES = (
        reaggregate.ModelSource("base_model", "base", "main"),
        reaggregate.ModelSource("new_model", "expansion", "baseline"),
    )

    @staticmethod
    def _rows(experiment, randomness, temperature):
        smiles = ("c1ccccc1", "Cc1ccccc1", "CC")
        rows = []
        for sample_index, value in enumerate((0.6, 0.7, 0.8)):
            rows.append(
                {
                    "experiment": experiment,
                    "display_name": experiment,
                    "checkpoint_path": f"/checkpoint/{experiment}",
                    "qed_weight": 0.6,
                    "sa_score_weight": 0.4,
                    "sample_index": sample_index,
                    "sweep_axis": "randomness_temperature_pair",
                    "sweep_value": float(
                        1 if (randomness, temperature) == (0.1, 0.5) else 2
                    ),
                    "sweep_label": f"r={randomness},t={temperature}",
                    "generation_temperature": temperature,
                    "randomness": randomness,
                    "reward": value,
                    "is_valid": True,
                    "alert_hit": False,
                    "qed": value,
                    "sa": 2.0,
                    "sa_score": 0.75,
                    "soft_reward": value,
                    "smiles": smiles[sample_index],
                }
            )
        return rows

    @classmethod
    def _summary(cls, rows):
        first = rows[0]
        return {
            "experiment": first["experiment"],
            "display_name": first["display_name"],
            "checkpoint_path": first["checkpoint_path"],
            "qed_weight": 0.6,
            "sa_score_weight": 0.4,
            "num_samples": len(rows),
            "sweep_axis": first["sweep_axis"],
            "sweep_value": first["sweep_value"],
            "sweep_label": first["sweep_label"],
            "generation_temperature": first["generation_temperature"],
            "randomness": first["randomness"],
            "diversity_metric": "morgan_internal_diversity",
            "reward_mean": 0.7,
            "qed_mean": 0.7,
            "sa_mean": 2.0,
            "sa_score_mean": 0.75,
            "soft_reward_mean": 0.7,
            "diversity": 0.5,
            "valid_fraction": 1.0,
            "alert_hit_fraction": 0.0,
            "invalid_fraction": 0.0,
        }

    def _build_inputs(self, root):
        roots = {
            "base": root / "base",
            "expansion": root / "expansion",
        }
        for source in self.SOURCES:
            aggregate = (
                roots[source.run_kind]
                / "denovo"
                / f"seed{self.SEED}"
                / source.category
                / source.experiment
                / "aggregate"
            )
            aggregate.mkdir(parents=True)
            all_rows = []
            summaries = []
            for randomness, temperature in self.SWEEP:
                rows = self._rows(source.experiment, randomness, temperature)
                all_rows.extend(rows)
                summaries.append(self._summary(rows))
            (aggregate / "dense.rows.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in all_rows)
            )
            (aggregate / "dense.json").write_text(json.dumps(summaries))
        return roots

    def test_reuses_utility_and_recomputes_scaffold_diversity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = self._build_inputs(root)
            output_root = root / "output"

            manifest_path = reaggregate.reaggregate(
                base_run_root=roots["base"],
                expansion_run_root=roots["expansion"],
                output_root=output_root,
                seeds=(self.SEED,),
                sweep=self.SWEEP,
                samples_per_point=3,
                model_sources=self.SOURCES,
            )

            manifest = json.loads(manifest_path.read_text())
            rows = json.loads(
                (output_root / f"denovo/seed{self.SEED}.json").read_text()
            )
            self.assertEqual(manifest["total_raw_rows"], 12)
            self.assertEqual(len(rows), 4)
            self.assertTrue(
                all(row["soft_reward_mean"] == 0.7 for row in rows)
            )
            self.assertTrue(
                all(
                    row["diversity_metric"]
                    == reaggregate.RELATIVE_SCAFFOLD_DIVERSITY
                    for row in rows
                )
            )
            self.assertTrue(all(row["diversity"] == 2 / 3 for row in rows))

    def test_rejects_source_utility_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = self._build_inputs(root)
            summary_path = (
                roots["base"]
                / "denovo"
                / f"seed{self.SEED}"
                / "main/base_model/aggregate/dense.json"
            )
            summaries = json.loads(summary_path.read_text())
            summaries[0]["soft_reward_mean"] = 0.71
            summary_path.write_text(json.dumps(summaries))

            with self.assertRaisesRegex(ValueError, "soft_reward_mean mismatch"):
                reaggregate.reaggregate(
                    base_run_root=roots["base"],
                    expansion_run_root=roots["expansion"],
                    output_root=root / "output",
                    seeds=(self.SEED,),
                    sweep=self.SWEEP,
                    samples_per_point=3,
                    model_sources=self.SOURCES,
                )

    def test_rejects_duplicate_sample_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = self._build_inputs(root)
            rows_path = (
                roots["expansion"]
                / "denovo"
                / f"seed{self.SEED}"
                / "baseline/new_model/aggregate/dense.rows.jsonl"
            )
            rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
            rows[1]["sample_index"] = rows[0]["sample_index"]
            rows_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )

            with self.assertRaisesRegex(ValueError, "duplicate sample_index"):
                reaggregate.reaggregate(
                    base_run_root=roots["base"],
                    expansion_run_root=roots["expansion"],
                    output_root=root / "output",
                    seeds=(self.SEED,),
                    sweep=self.SWEEP,
                    samples_per_point=3,
                    model_sources=self.SOURCES,
                )


if __name__ == "__main__":
    unittest.main()
