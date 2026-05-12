import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"
FILTER_SCRIPT = REPO_ROOT / "scripts" / "filter_reproduction_run_plan.py"


class FilterReproductionRunPlanTests(unittest.TestCase):
    def test_filter_dino_imagenet_vertical_slice(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            full_plan = tmpdir / "full_plan.json"
            subset_plan = tmpdir / "subset_plan.json"
            env = dict(os.environ)
            env["PYTHONPATH"] = str(REPO_ROOT / "src")

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "feature_economy.cli.main",
                    "plan-runs",
                    "--config-root",
                    str(CONFIG_ROOT),
                    "--output-json",
                    str(full_plan),
                ],
                check=True,
                cwd=REPO_ROOT,
                env=env,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(FILTER_SCRIPT),
                    "--input-json",
                    str(full_plan),
                    "--output-json",
                    str(subset_plan),
                    "--model-id",
                    "dino_v2_base",
                    "--task-id",
                    "imagenet_1k",
                    "--task-id",
                    "imagenet_1k_val",
                    "--sae-id",
                    "",
                    "--sae-id",
                    "dino_l11_topk32_exp4",
                ],
                check=True,
                cwd=REPO_ROOT,
                env=env,
            )

            payload = json.loads(subset_plan.read_text(encoding="utf-8"))
            self.assertEqual(payload["record_type"], "reproduction_run_plan")
            self.assertEqual(payload["num_rows"], 9)
            self.assertEqual(
                set(payload["stages"]),
                {
                    "availability",
                    "feature_ablation",
                    "feature_extraction",
                    "feature_ranking",
                    "contribution_scores",
                    "native_probe",
                    "sae_code_extraction",
                    "sae_probe",
                    "subset_usage",
                },
            )
            self.assertTrue(all(row["model_id"] == "dino_v2_base" for row in payload["rows"]))


if __name__ == "__main__":
    unittest.main()
