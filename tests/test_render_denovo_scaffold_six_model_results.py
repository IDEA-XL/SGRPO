import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VIS_CODE = REPO_ROOT / "vis-code"
sys.path.insert(0, str(VIS_CODE))
MODULE_PATH = VIS_CODE / "render_denovo_scaffold_six_model_results.py"
SPEC = importlib.util.spec_from_file_location(
    "render_denovo_scaffold_six_model_results_for_test",
    MODULE_PATH,
)
render = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = render
SPEC.loader.exec_module(render)


class RenderDenovoScaffoldSixModelResultsTest(unittest.TestCase):
    def test_panel_contains_all_six_models(self):
        self.assertEqual(
            [model.source_id for model in render.PANEL.models],
            [
                "original_genmol_v2",
                "genmol_denovo_grpo_2000",
                "genmol_denovo_grpo_hbd_2000",
                "denovo_dmb_2000",
                "denovo_entropy_2000",
                "genmol_denovo_sgrpo_rewardsum_loo_2000",
            ],
        )

    def test_upsert_replaces_only_six_model_reanalysis_section(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expanded.md"
            path.write_text(
                "# Results\n\n"
                "<!-- BEGIN DENOVO SCAFFOLD DIVERSITY RESULTS -->\n"
                "preserve earlier scaffold-training result\n"
                "<!-- END DENOVO SCAFFOLD DIVERSITY RESULTS -->\n\n"
                f"{render.SECTION_BEGIN}\n"
                "old six-model reanalysis section\n"
                f"{render.SECTION_END}\n\n"
                "<!-- BEGIN OTHER RESULTS -->\n"
                "preserve me\n"
                "<!-- END OTHER RESULTS -->\n"
            )

            render._upsert_section(path, "new six-model section\n")
            text = path.read_text()

        self.assertEqual(text.count(render.SECTION_BEGIN), 1)
        self.assertEqual(text.count(render.SECTION_END), 1)
        self.assertNotIn("old six-model reanalysis section", text)
        self.assertIn("new six-model section", text)
        self.assertIn("preserve earlier scaffold-training result", text)
        self.assertIn("preserve me", text)

    def test_upsert_rejects_unbalanced_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expanded.md"
            path.write_text(render.SECTION_BEGIN + "\nvalue\n")
            with self.assertRaisesRegex(RuntimeError, "Malformed"):
                render._upsert_section(path, "replacement")


if __name__ == "__main__":
    unittest.main()
