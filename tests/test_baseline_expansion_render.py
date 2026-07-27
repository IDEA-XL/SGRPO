import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VIS_CODE = REPO_ROOT / "vis-code"
sys.path.insert(0, str(VIS_CODE))
MODULE_PATH = VIS_CODE / "render_baseline_expansion_tables.py"
SPEC = importlib.util.spec_from_file_location(
    "render_baseline_expansion_tables_for_test",
    MODULE_PATH,
)
render = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = render
SPEC.loader.exec_module(render)


class BaselineExpansionRenderTest(unittest.TestCase):
    def test_preserves_all_well_formed_result_sections(self):
        text = """# Existing

<!-- BEGIN FIRST RESULTS -->
first
<!-- END FIRST RESULTS -->

<!-- BEGIN SECOND RESULTS -->
second
<!-- END SECOND RESULTS -->
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.md"
            path.write_text(text)
            sections = render._preserved_result_sections(path)

        self.assertEqual(len(sections), 2)
        self.assertIn("first", sections[0])
        self.assertIn("second", sections[1])

    def test_rejects_unbalanced_result_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.md"
            path.write_text(
                "<!-- BEGIN FIRST RESULTS -->\n"
                "value\n"
                "<!-- END SECOND RESULTS -->\n"
            )
            with self.assertRaisesRegex(RuntimeError, "Malformed"):
                render._preserved_result_sections(path)

    def test_missing_output_has_no_preserved_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.md"
            self.assertEqual(render._preserved_result_sections(path), [])


if __name__ == "__main__":
    unittest.main()
