import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VIS_CODE = REPO_ROOT / "vis-code"
sys.path.insert(0, str(VIS_CODE))
MODULE_PATH = VIS_CODE / "render_denovo_credit_ablation_results.py"
SPEC = importlib.util.spec_from_file_location(
    "render_denovo_credit_ablation_results_for_test",
    MODULE_PATH,
)
render = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = render
SPEC.loader.exec_module(render)


class RenderDenovoCreditAblationResultsTest(unittest.TestCase):
    def test_panels_match_the_two_requested_comparison_groups(self):
        self.assertEqual(
            [
                [model.source_id for model in panel.models]
                for panel in render.PANELS
            ],
            [
                [
                    "genmol_denovo_sgrpo_rewardsum_loo_2000",
                    "denovo_raw_loo_diversity_2000",
                ],
                [
                    "genmol_denovo_sgrpo_rewardsum_loo_2000",
                    "denovo_mean_baseline_2000",
                    "denovo_mean_baseline_std_2000",
                ],
            ],
        )

    def test_upsert_preserves_existing_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expanded.md"
            path.write_text(
                "# Results\n\n"
                "<!-- BEGIN DENOVO SCAFFOLD DIVERSITY RESULTS -->\n"
                "preserve training result\n"
                "<!-- END DENOVO SCAFFOLD DIVERSITY RESULTS -->\n\n"
                "<!-- BEGIN MMGENMOL THREE-MODEL SCAFFOLD REANALYSIS RESULTS -->\n"
                "preserve mmGenMol result\n"
                "<!-- END MMGENMOL THREE-MODEL SCAFFOLD REANALYSIS RESULTS -->\n"
            )
            render._upsert_section(path, "new credit-ablation result\n")
            text = path.read_text()

        self.assertEqual(text.count(render.SECTION_BEGIN), 1)
        self.assertEqual(text.count(render.SECTION_END), 1)
        self.assertIn("new credit-ablation result", text)
        self.assertIn("preserve training result", text)
        self.assertIn("preserve mmGenMol result", text)

    def test_upsert_rejects_unbalanced_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expanded.md"
            path.write_text(render.SECTION_BEGIN + "\nvalue\n")
            with self.assertRaisesRegex(RuntimeError, "Malformed"):
                render._upsert_section(path, "replacement")


if __name__ == "__main__":
    unittest.main()
