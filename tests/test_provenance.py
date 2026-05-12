import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feature_economy.artifacts import current_git_commit
from feature_economy.models import extract_fixture_features


PUBLIC_REPRO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).parent / "fixtures"


class ProvenanceTests(unittest.TestCase):
    def test_current_git_commit_uses_environment_override(self):
        with patch.dict(os.environ, {"FEATURE_ECONOMY_GIT_COMMIT": "test-commit"}, clear=False):
            self.assertEqual(current_git_commit(), "test-commit")

    def test_run_manifest_records_git_commit_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(
                os.environ,
                {"FEATURE_ECONOMY_GIT_COMMIT": "fixture-commit"},
                clear=False,
            ):
                extract_fixture_features(
                    config_root=PUBLIC_REPRO_ROOT / "configs",
                    manifest_path=FIXTURE_DIR / "tiny_imagenet_eval_manifest.jsonl",
                    task_type="classification",
                    model_id="dino_v2_base",
                    output_dir=tmpdir,
                    expected_split="val",
                )
            manifest = json.loads((Path(tmpdir) / "run_manifest.json").read_text())
            self.assertEqual(manifest["git_commit"], "fixture-commit")


if __name__ == "__main__":
    unittest.main()
