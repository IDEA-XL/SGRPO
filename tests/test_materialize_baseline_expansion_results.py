import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts/materialize_baseline_expansion_results.py"
SPEC = importlib.util.spec_from_file_location(
    "materialize_baseline_expansion_results_for_test",
    MODULE_PATH,
)
materializer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = materializer
SPEC.loader.exec_module(materializer)


def _write_tsv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=rows[0].keys(),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


class MaterializeBaselineExpansionResultsTest(unittest.TestCase):
    def _build_run_root(self, root):
        run_root = root / "run"
        spec_root = run_root / "specs"
        denovo_rows = []
        for seed in materializer.SEEDS:
            config_path = spec_root / f"denovo/seed{seed}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                "randomness_temperature_pairs:\n"
                "  - randomness: 0.1\n"
                "    generation_temperature: 0.5\n"
                "  - randomness: 0.2\n"
                "    generation_temperature: 0.65\n"
            )
            denovo_rows.append(
                {
                    "task_id": seed,
                    "seed": seed,
                    "category": "baseline",
                    "experiment": "method",
                    "config_path": config_path,
                }
            )
            _write_tsv(
                spec_root / f"mmgenmol/seed{seed}.tsv",
                [{"task_id": 0}, {"task_id": 1}, {"task_id": 2}],
            )
            _write_tsv(
                spec_root / f"progen2/seed{seed}_tasks.tsv",
                [{"task_id": 0}, {"task_id": 1}, {"task_id": 2}, {"task_id": 3}],
            )
            outputs = {
                "denovo": [{"row": index} for index in range(2)],
                "mmgenmol": [{"row": index} for index in range(3)],
                "progen2": {
                    "results": [{"row": index} for index in range(4)]
                },
            }
            for domain, value in outputs.items():
                source = materializer._source_path(run_root, domain, seed)
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text(json.dumps(value))
        _write_tsv(spec_root / "denovo/tasks.tsv", denovo_rows)
        (run_root / "COMPLETE").write_text("2026-07-27T00:00:00+0000\n")
        (run_root / "controller_state.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "active_domains": list(materializer.REQUIRED_DOMAINS),
                }
            )
        )
        return run_root

    def test_materializes_all_seed_summaries_with_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = self._build_run_root(root)
            output_root = root / "output"
            manifest_path = materializer.materialize(run_root, output_root)
            manifest = json.loads(manifest_path.read_text())

            self.assertEqual(len(manifest["files"]), 15)
            self.assertTrue(
                all(len(record["sha256"]) == 64 for record in manifest["files"])
            )
            self.assertEqual(
                json.loads((output_root / "progen2/seed42.json").read_text())[
                    "results"
                ][-1]["row"],
                3,
            )

    def test_rejects_wrong_aggregate_row_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = self._build_run_root(root)
            source = materializer._source_path(run_root, "mmgenmol", 42)
            source.write_text(json.dumps([{"row": 0}]))

            with self.assertRaisesRegex(ValueError, "has 1 rows; expected 3"):
                materializer.materialize(run_root, root / "output")

    def test_requires_controller_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = self._build_run_root(root)
            (run_root / "COMPLETE").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "completion marker"):
                materializer.materialize(run_root, root / "output")

    def test_rejects_partial_domain_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = self._build_run_root(root)
            (run_root / "controller_state.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "active_domains": ["denovo", "mmgenmol"],
                    }
                )
            )

            with self.assertRaisesRegex(RuntimeError, "requires all domains"):
                materializer.materialize(run_root, root / "output")


if __name__ == "__main__":
    unittest.main()
