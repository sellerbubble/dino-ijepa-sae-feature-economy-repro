import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PREP_SCRIPT = REPO_ROOT / "scripts" / "prepare_real_artifact_slice.py"


class PrepareRealArtifactSliceTests(unittest.TestCase):
    def test_prepare_dino_imagenet_trial_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "dino_imagenet_trial"
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT / "src")

            subprocess.run(
                [
                    sys.executable,
                    str(PREP_SCRIPT),
                    "--config-root",
                    str(REPO_ROOT / "configs"),
                    "--output-dir",
                    str(output_dir),
                    "--profile",
                    "dino_imagenet_v1_trial",
                ],
                check=True,
                cwd=REPO_ROOT,
                env=env,
            )

            run_plan = json.loads((output_dir / "reproduction_run_plan.json").read_text())
            manifest = json.loads((output_dir / "artifact_export_manifest.json").read_text())
            with (output_dir / "artifact_export_manifest.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(run_plan["num_rows"], 15)
            self.assertEqual(manifest["num_rows"], 15)
            self.assertEqual(len(rows), 15)
            self.assertTrue((output_dir / "full_reproduction_run_plan.json").exists())
            self.assertTrue((output_dir / "README.md").exists())
            self.assertTrue(all(row["model_id"] == "dino_v2_base" for row in rows))
            self.assertIn(
                "features.npz;feature_extraction_summary.json;run_manifest.json",
                {row["required_files"] for row in rows},
            )
            statuses = {row["stage"]: row["export_status"] for row in rows}
            self.assertEqual(statuses["feature_extraction"], "TODO")
            self.assertEqual(statuses["contribution_scores"], "TODO")
            self.assertEqual(statuses["sae_probe"], "NEEDS_CONVERSION")
            self.assertEqual(statuses["feature_ablation"], "NEEDS_CONVERSION")
            ranking_methods = {row["ranking_method"] for row in rows if row["stage"] == "feature_ranking"}
            self.assertEqual(ranking_methods, {"probe_weight", "validation_contribution", "hybrid"})
            self.assertIn("stage_artifact_bundle_from_manifest.sh", (output_dir / "README.md").read_text())

    def test_prepare_ijepa_imagenet_trial_workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "ijepa_imagenet_trial"
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT / "src")

            subprocess.run(
                [
                    sys.executable,
                    str(PREP_SCRIPT),
                    "--config-root",
                    str(REPO_ROOT / "configs"),
                    "--output-dir",
                    str(output_dir),
                    "--profile",
                    "ijepa_imagenet_v1_trial",
                ],
                check=True,
                cwd=REPO_ROOT,
                env=env,
            )

            run_plan = json.loads((output_dir / "reproduction_run_plan.json").read_text())
            with (output_dir / "artifact_export_manifest.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(run_plan["num_rows"], 15)
            self.assertEqual(len(rows), 15)
            self.assertTrue(all(row["model_id"] == "ijepa_vit_h14" for row in rows))
            self.assertIn("ijepa_l31_topk32_exp4", {row["sae_id"] for row in rows})
            statuses = {row["stage"]: row["export_status"] for row in rows}
            self.assertEqual(statuses["feature_extraction"], "TODO")
            self.assertEqual(statuses["contribution_scores"], "TODO")
            self.assertEqual(statuses["sae_probe"], "NEEDS_CONVERSION")

    def test_prepare_task_level_trial_profiles(self):
        expected = {
            "imagenet_v1_trial": (30, {"imagenet_1k", "imagenet_1k_val"}),
            "nyuv2_v1_trial": (28, {"nyuv2_depth"}),
            "dino_nyuv2_v1_trial": (14, {"nyuv2_depth"}),
            "ijepa_nyuv2_v1_trial": (14, {"nyuv2_depth"}),
            "ade20k_v1_trial": (28, {"ade20k_segmentation"}),
            "dino_ade20k_v1_trial": (14, {"ade20k_segmentation"}),
            "ijepa_ade20k_v1_trial": (14, {"ade20k_segmentation"}),
            "clevr_count_v1_trial": (28, {"clevr_count"}),
            "dino_clevr_count_v1_trial": (14, {"clevr_count"}),
            "ijepa_clevr_count_v1_trial": (14, {"clevr_count"}),
            "layer_sweep_v1_trial": (27, {"imagenet_1k_val"}),
        }
        for profile, (expected_rows, expected_tasks) in expected.items():
            with self.subTest(profile=profile):
                with tempfile.TemporaryDirectory() as tmpdir:
                    output_dir = Path(tmpdir) / profile
                    env = dict(os.environ)
                    env["PYTHONPATH"] = str(REPO_ROOT / "src")

                    subprocess.run(
                        [
                            sys.executable,
                            str(PREP_SCRIPT),
                            "--config-root",
                            str(REPO_ROOT / "configs"),
                            "--output-dir",
                            str(output_dir),
                            "--profile",
                            profile,
                        ],
                        check=True,
                        cwd=REPO_ROOT,
                        env=env,
                    )

                    run_plan = json.loads((output_dir / "reproduction_run_plan.json").read_text())
                    with (output_dir / "artifact_export_manifest.csv").open(newline="") as handle:
                        rows = list(csv.DictReader(handle))

                    self.assertEqual(run_plan["num_rows"], expected_rows)
                    self.assertEqual(len(rows), expected_rows)
                    self.assertEqual({row["task_id"] for row in rows}, expected_tasks)
                    expected_models = {"dino_v2_base", "ijepa_vit_h14"}
                    if profile.startswith("dino_"):
                        expected_models = {"dino_v2_base"}
                    elif profile.startswith("ijepa_"):
                        expected_models = {"ijepa_vit_h14"}
                    self.assertEqual({row["model_id"] for row in rows}, expected_models)
                    self.assertIn("TODO", {row["export_status"] for row in rows})
                    self.assertIn("NEEDS_CONVERSION", {row["export_status"] for row in rows})


if __name__ == "__main__":
    unittest.main()
