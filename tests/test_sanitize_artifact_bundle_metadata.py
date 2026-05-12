import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "sanitize_artifact_bundle_metadata.py"


class SanitizeArtifactBundleMetadataTests(unittest.TestCase):
    def test_sanitize_metadata_replaces_private_prefixes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata = root / "run_manifest.json"
            metadata.write_text(
                json.dumps(
                    {
                        "input": "/private/source/artifacts/bundle/features/x.npz",
                        "output": "/private/source/models/checkpoint.pt",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = root / "sanitize_report.json"

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--artifact-root",
                    str(root),
                    "--replace",
                    "/private/source/artifacts/bundle/=${ARTIFACT_ROOT}/",
                    "--replace",
                    "/private/source/models/=${PRIVATE_MODEL_ROOT}/",
                    "--output-json",
                    str(report),
                    "--require-no-private-paths",
                ],
                check=True,
            )

            rewritten = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertEqual(rewritten["input"], "${ARTIFACT_ROOT}/features/x.npz")
            self.assertEqual(rewritten["output"], "${PRIVATE_MODEL_ROOT}/checkpoint.pt")
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["changed_files"], 1)
            self.assertEqual(payload["remaining_private_path_hits"], 0)
            self.assertEqual(payload["artifact_root"], "${ARTIFACT_ROOT}")
            self.assertEqual(payload["replacement_count"], 2)
            self.assertEqual(payload["replacements"][0]["old"], "<redacted>")

    def test_sanitize_metadata_supports_scan_only_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata = root / "summary.json"
            metadata.write_text('{"path": "${ARTIFACT_ROOT}/features/x.npz"}\n', encoding="utf-8")
            report = root / "sanitize_report.json"

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--artifact-root",
                    str(root),
                    "--output-json",
                    str(report),
                    "--require-no-private-paths",
                ],
                check=True,
            )

            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["changed_files"], 0)
            self.assertEqual(payload["replacement_count"], 0)
            self.assertEqual(payload["remaining_private_path_hits"], 0)


if __name__ == "__main__":
    unittest.main()
