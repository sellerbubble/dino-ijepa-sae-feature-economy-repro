import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README_SCRIPT = REPO_ROOT / "scripts" / "create_artifact_bundle_readme.py"


class ArtifactBundleReadmeTests(unittest.TestCase):
    def test_create_readme_from_run_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            run_plan = tmpdir / "run_plan.json"
            readme = tmpdir / "README.md"
            run_plan.write_text(
                json.dumps(
                    {
                        "record_type": "reproduction_run_plan",
                        "num_rows": 2,
                        "rows": [
                            {
                                "stage": "native_probe",
                                "task_id": "imagenet_1k",
                                "model_id": "dino_v2_base",
                                "sae_id": "",
                            },
                            {
                                "stage": "feature_ranking",
                                "task_id": "imagenet_1k",
                                "model_id": "dino_v2_base",
                                "sae_id": "dino_l11_topk32_exp4",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(README_SCRIPT),
                    "--run-plan-json",
                    str(run_plan),
                    "--output-readme",
                    str(readme),
                    "--bundle-name",
                    "feature_economy_test_bundle",
                    "--public-repo-commit",
                    "abc123",
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            text = readme.read_text(encoding="utf-8")
            self.assertIn("# feature_economy_test_bundle", text)
            self.assertIn("Public reproduction repo commit: `abc123`", text)
            self.assertIn("| `native_probe` | 1 |", text)
            self.assertIn("| `feature_ranking` | 1 |", text)
            self.assertIn("check-bundle", text)
            self.assertIn("artifact-first saved-array reproduction", text)


if __name__ == "__main__":
    unittest.main()
