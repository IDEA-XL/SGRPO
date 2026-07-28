import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VIS_CODE = REPO_ROOT / "vis-code"
sys.path.insert(0, str(VIS_CODE))
MODULE_PATH = VIS_CODE / "render_mmgenmol_scaffold_three_model_results.py"
SPEC = importlib.util.spec_from_file_location(
    "render_mmgenmol_scaffold_three_model_results_for_test",
    MODULE_PATH,
)
render = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = render
SPEC.loader.exec_module(render)


class RenderMmgenmolScaffoldThreeModelResultsTest(unittest.TestCase):
    def test_panel_contains_only_paper_models(self):
        self.assertEqual(
            [model.source_id for model in render.PANEL.models],
            [
                "original_5500",
                "grpo_unidock_1000",
                "sgrpo_unidock_rewardsum_loo_1000",
            ],
        )

    def test_upsert_preserves_every_existing_result_section(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expanded.md"
            path.write_text(
                "# Results\n\n"
                "<!-- BEGIN DENOVO SCAFFOLD DIVERSITY RESULTS -->\n"
                "preserve scaffold training\n"
                "<!-- END DENOVO SCAFFOLD DIVERSITY RESULTS -->\n\n"
                "<!-- BEGIN DENOVO SIX-MODEL SCAFFOLD REANALYSIS RESULTS -->\n"
                "preserve denovo reanalysis\n"
                "<!-- END DENOVO SIX-MODEL SCAFFOLD REANALYSIS RESULTS -->\n\n"
                f"{render.SECTION_BEGIN}\n"
                "replace only this section\n"
                f"{render.SECTION_END}\n"
            )

            render._upsert_section(path, "new mmGenMol scaffold section\n")
            text = path.read_text()

        self.assertEqual(text.count(render.SECTION_BEGIN), 1)
        self.assertEqual(text.count(render.SECTION_END), 1)
        self.assertNotIn("replace only this section", text)
        self.assertIn("new mmGenMol scaffold section", text)
        self.assertIn("preserve scaffold training", text)
        self.assertIn("preserve denovo reanalysis", text)

    def test_upsert_rejects_unbalanced_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expanded.md"
            path.write_text(render.SECTION_BEGIN + "\nvalue\n")
            with self.assertRaisesRegex(RuntimeError, "Malformed"):
                render._upsert_section(path, "replacement")


if __name__ == "__main__":
    unittest.main()
