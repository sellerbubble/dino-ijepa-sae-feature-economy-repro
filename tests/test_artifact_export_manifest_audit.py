import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = REPO_ROOT / "scripts" / "audit_artifact_export_manifest.py"


class ArtifactExportManifestAuditTests(unittest.TestCase):
    def test_audit_reports_ready_source_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            source.mkdir()
            for filename in ["native_probe_summary.json", "probe_logits.npz", "run_manifest.json"]:
                (source / filename).write_text("{}", encoding="utf-8")
            manifest_csv = root / "manifest.csv"
            audit_json = root / "audit.json"
            _write_manifest(
                manifest_csv,
                export_status="READY",
                source_artifact_dir=str(source),
            )

            subprocess.run(
                [
                    sys.executable,
                    str(AUDIT_SCRIPT),
                    "--manifest-csv",
                    str(manifest_csv),
                    "--artifact-root",
                    str(root / "public"),
                    "--output-json",
                    str(audit_json),
                    "--require-ready",
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            report = json.loads(audit_json.read_text(encoding="utf-8"))
            self.assertTrue(report["ready"])
            self.assertEqual(report["ready_rows"], 1)
            self.assertEqual(report["problem_rows"], 0)
            self.assertTrue(report["rows"][0]["resolved_public_artifact_dir"].endswith("/public/probes/native/dino/imagenet/val"))

    def test_audit_reports_empty_source_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_csv = root / "manifest.csv"
            audit_json = root / "audit.json"
            _write_manifest(
                manifest_csv,
                export_status="TODO",
                source_artifact_dir="",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(AUDIT_SCRIPT),
                    "--manifest-csv",
                    str(manifest_csv),
                    "--output-json",
                    str(audit_json),
                    "--require-ready",
                ],
                cwd=REPO_ROOT,
            )

            self.assertNotEqual(result.returncode, 0)
            report = json.loads(audit_json.read_text(encoding="utf-8"))
            self.assertFalse(report["ready"])
            self.assertIn("source_artifact_dir is empty", report["rows"][0]["problems"])


def _write_manifest(path: Path, *, export_status: str, source_artifact_dir: str) -> None:
    columns = [
        "row_id",
        "export_status",
        "stage",
        "experiment_id",
        "task_id",
        "task_type",
        "model_id",
        "sae_id",
        "command",
        "public_artifact_dir",
        "required_files",
        "source_artifact_dir",
        "source_artifact_note",
        "conversion_needed",
        "validation_command",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow(
            {
                "row_id": "row_001",
                "export_status": export_status,
                "stage": "native_probe",
                "experiment_id": "paper_v0_native",
                "task_id": "imagenet_1k",
                "task_type": "classification",
                "model_id": "dino_v2_base",
                "sae_id": "",
                "command": "probe-native",
                "public_artifact_dir": "${ARTIFACT_ROOT}/probes/native/dino/imagenet/val",
                "required_files": "native_probe_summary.json;probe_logits.npz;run_manifest.json",
                "source_artifact_dir": source_artifact_dir,
                "source_artifact_note": "",
                "conversion_needed": "no",
                "validation_command": "index-artifacts --require-valid",
                "notes": "",
            }
        )


if __name__ == "__main__":
    unittest.main()
