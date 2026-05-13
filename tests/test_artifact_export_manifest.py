import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"


class ArtifactExportManifestTests(unittest.TestCase):
    def test_create_export_manifest_from_public_run_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            run_plan = tmpdir / "reproduction_run_plan.json"
            output_csv = tmpdir / "artifact_export_manifest.csv"
            output_json = tmpdir / "artifact_export_manifest.json"
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
                    str(run_plan),
                ],
                check=True,
                cwd=REPO_ROOT,
                env=env,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "create_artifact_export_manifest.py"),
                    "--run-plan-json",
                    str(run_plan),
                    "--output-csv",
                    str(output_csv),
                    "--output-json",
                    str(output_json),
                ],
                check=True,
                cwd=REPO_ROOT,
                env=env,
            )

            with output_csv.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["record_type"], "artifact_export_manifest_template")
            self.assertEqual(payload["num_rows"], 114)
            self.assertEqual(len(rows), 114)
            self.assertEqual(rows[0]["export_status"], "TODO")
            self.assertIn("run_manifest.json", rows[0]["required_files"])
            self.assertTrue(
                any(
                    row["stage"] == "feature_ablation"
                    and row["task_id"] == "nyuv2_depth"
                    and row["model_id"] == "ijepa_vit_h14"
                    and row["ranking_method"] == "hybrid"
                    for row in rows
                )
            )


if __name__ == "__main__":
    unittest.main()
