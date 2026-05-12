import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COPY_SCRIPT = REPO_ROOT / "scripts" / "copy_artifact_export_manifest.py"


class ArtifactExportManifestCopyTests(unittest.TestCase):
    def test_copy_ready_row_copies_required_files_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            source.mkdir()
            for filename in ["native_probe_summary.json", "probe_logits.npz", "run_manifest.json"]:
                (source / filename).write_text(filename, encoding="utf-8")
            (source / "extra_private_note.txt").write_text("do not copy", encoding="utf-8")
            manifest_csv = root / "manifest.csv"
            artifact_root = root / "public"
            report_json = root / "copy_report.json"
            _write_manifest(manifest_csv, export_status="READY", source_artifact_dir=str(source))

            subprocess.run(
                [
                    sys.executable,
                    str(COPY_SCRIPT),
                    "--manifest-csv",
                    str(manifest_csv),
                    "--artifact-root",
                    str(artifact_root),
                    "--output-json",
                    str(report_json),
                ],
                check=True,
                cwd=REPO_ROOT,
            )

            dest = artifact_root / "probes" / "native" / "dino" / "imagenet" / "val"
            self.assertTrue((dest / "native_probe_summary.json").exists())
            self.assertTrue((dest / "probe_logits.npz").exists())
            self.assertTrue((dest / "run_manifest.json").exists())
            self.assertFalse((dest / "extra_private_note.txt").exists())
            report = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(report["copied_rows"], 1)
            self.assertEqual(report["copied_files"], 3)

    def test_copy_refuses_to_overwrite_without_force(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            source.mkdir()
            for filename in ["native_probe_summary.json", "probe_logits.npz", "run_manifest.json"]:
                (source / filename).write_text(filename, encoding="utf-8")
            artifact_root = root / "public"
            dest = artifact_root / "probes" / "native" / "dino" / "imagenet" / "val"
            dest.mkdir(parents=True)
            (dest / "run_manifest.json").write_text("existing", encoding="utf-8")
            manifest_csv = root / "manifest.csv"
            report_json = root / "copy_report.json"
            _write_manifest(manifest_csv, export_status="READY", source_artifact_dir=str(source))

            result = subprocess.run(
                [
                    sys.executable,
                    str(COPY_SCRIPT),
                    "--manifest-csv",
                    str(manifest_csv),
                    "--artifact-root",
                    str(artifact_root),
                    "--output-json",
                    str(report_json),
                ],
                cwd=REPO_ROOT,
            )

            self.assertNotEqual(result.returncode, 0)
            report = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(report["problem_rows"], 1)
            self.assertTrue(any("--force was not set" in problem for problem in report["rows"][0]["problems"]))


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
