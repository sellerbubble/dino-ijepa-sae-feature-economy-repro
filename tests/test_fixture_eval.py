import json
import tempfile
import unittest
from pathlib import Path

from feature_economy.artifacts import validate_probe_summary
from feature_economy.probes import evaluate_native_fixture, evaluate_sae_fixture


FIXTURE_DIR = Path(__file__).parent / "fixtures"


class FixtureEvalTests(unittest.TestCase):
    def test_evaluate_count_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = evaluate_native_fixture(
                manifest_path=FIXTURE_DIR / "tiny_clevr_count_eval_manifest.jsonl",
                task_type="count_classification",
                task_id="clevr_count",
                model_id="dino_v2_base",
                output_dir=tmpdir,
                expected_split="val",
                seed=42,
            )
            record = json.loads(summary_path.read_text())
            validate_probe_summary(record)
            self.assertAlmostEqual(record["metrics"]["accuracy"], 2 / 3)
            self.assertTrue(record["fixture"])

    def test_evaluate_classification_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = evaluate_native_fixture(
                manifest_path=FIXTURE_DIR / "tiny_imagenet_eval_manifest.jsonl",
                task_type="classification",
                task_id="imagenet_1k",
                model_id="ijepa_vit_h14",
                output_dir=tmpdir,
                expected_split="val",
            )
            record = json.loads(summary_path.read_text())
            self.assertAlmostEqual(record["metrics"]["top1"], 1 / 3)
            self.assertAlmostEqual(record["metrics"]["top3"], 1.0)

    def test_probe_native_command_name_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluate_native_fixture(
                manifest_path=FIXTURE_DIR / "tiny_imagenet_eval_manifest.jsonl",
                task_type="classification",
                task_id="imagenet_1k",
                model_id="dino_v2_base",
                output_dir=tmpdir,
                expected_split="val",
                command_name="feature-economy probe-native",
            )
            manifest = json.loads((Path(tmpdir) / "run_manifest.json").read_text())
            self.assertEqual(manifest["command"], "feature-economy probe-native")

    def test_evaluate_sae_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = evaluate_sae_fixture(
                manifest_path=FIXTURE_DIR / "tiny_clevr_count_eval_manifest.jsonl",
                task_type="count_classification",
                task_id="clevr_count",
                model_id="ijepa_vit_h14",
                sae_id="ijepa_l31_topk32_exp4",
                output_dir=tmpdir,
                expected_split="val",
                seed=7,
            )
            record = json.loads(summary_path.read_text())
            validate_probe_summary(record)
            self.assertEqual(record["record_type"], "sae_probe_summary")
            self.assertEqual(record["sae_id"], "ijepa_l31_topk32_exp4")
            self.assertAlmostEqual(record["metrics"]["accuracy"], 2 / 3)

    def test_evaluate_depth_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = evaluate_native_fixture(
                manifest_path=FIXTURE_DIR / "tiny_nyuv2_eval_manifest.jsonl",
                task_type="dense_depth",
                task_id="nyuv2_depth",
                model_id="dino_v2_base",
                output_dir=tmpdir,
                expected_split="val",
            )
            record = json.loads(summary_path.read_text())
            self.assertIn("rmse", record["metrics"])
            self.assertIn("abs_rel", record["metrics"])
            self.assertIn("delta1", record["metrics"])
            self.assertGreater(record["metrics"]["rmse"], 0.0)

    def test_evaluate_segmentation_fixture(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = evaluate_native_fixture(
                manifest_path=FIXTURE_DIR / "tiny_ade20k_eval_manifest.jsonl",
                task_type="dense_segmentation",
                task_id="ade20k_segmentation",
                model_id="ijepa_vit_h14",
                output_dir=tmpdir,
                expected_split="val",
                num_classes=2,
            )
            record = json.loads(summary_path.read_text())
            self.assertIn("miou", record["metrics"])
            self.assertIn("pixel_accuracy", record["metrics"])
            self.assertGreater(record["metrics"]["pixel_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
