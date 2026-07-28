import copy
import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "vis-code/render_motif_extension_results.py"
SPEC = importlib.util.spec_from_file_location(
    "render_motif_extension_results",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _rows_by_motif():
    return {
        motif_index: [
            {
                "sample_index": sample_index,
                "smiles": "CC",
                "raw_smiles": "CC",
                "motif_retained": True,
                "is_valid": True,
                "alert_hit": False,
                "soft_reward": 0.5,
                "qed": 0.5,
                "sa": 2.0,
                "sa_score": 0.8,
            }
            for sample_index in range(100)
        ]
        for motif_index in range(10)
    }


def _summary_row(rows_by_motif):
    model = MODULE.PANEL.models[0]
    checkpoint_path = "/checkpoints/model.ckpt"
    summary = {
        "experiment": model.source_id,
        "display_name": model.label,
        "checkpoint_path": checkpoint_path,
        "seed": 42,
        "point_index": 0,
        "sweep_axis": "randomness_temperature_pair",
        "sweep_value": 1.0,
        "sweep_label": "r=0.1,t=0.5",
        "randomness": 0.1,
        "generation_temperature": 0.5,
        "num_motifs": 10,
        "samples_per_motif": 100,
        "num_samples": 1_000,
        "diversity_metric": MODULE.MORGAN_INTERNAL_DIVERSITY,
    }
    summary.update(
        MODULE._recompute_point_summary(
            rows_by_motif,
            context="test",
        )
    )
    return model, checkpoint_path, summary


class MotifExtensionRenderTest(unittest.TestCase):
    def test_raw_summary_validation_accepts_matching_aggregates(self):
        rows_by_motif = _rows_by_motif()
        model, checkpoint_path, summary = _summary_row(rows_by_motif)

        MODULE._validate_point_summary(
            summary,
            rows_by_motif,
            model=model,
            seed=42,
            point_index=0,
            checkpoint_path=checkpoint_path,
        )

    def test_raw_summary_validation_rejects_tampered_utility(self):
        rows_by_motif = _rows_by_motif()
        model, checkpoint_path, summary = _summary_row(rows_by_motif)
        tampered = copy.deepcopy(summary)
        tampered["soft_reward_mean"] += 0.1

        with self.assertRaisesRegex(ValueError, "raw-to-summary mismatch"):
            MODULE._validate_point_summary(
                tampered,
                rows_by_motif,
                model=model,
                seed=42,
                point_index=0,
                checkpoint_path=checkpoint_path,
            )


if __name__ == "__main__":
    unittest.main()
