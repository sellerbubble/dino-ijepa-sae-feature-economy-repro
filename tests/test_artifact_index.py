import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_economy.artifacts import build_artifact_index
from feature_economy.artifacts.arrays import validate_array_contract
from feature_economy.probes import write_smoke_probe_artifacts


PUBLIC_REPRO_ROOT = Path(__file__).resolve().parents[1]


class ArtifactIndexTests(unittest.TestCase):
    def test_index_smoke_probe_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            artifact_dir = tmpdir / "artifacts"
            index_json = tmpdir / "index" / "artifact_index.json"
            index_csv = tmpdir / "index" / "artifact_index.csv"
            write_smoke_probe_artifacts(
                config_root=PUBLIC_REPRO_ROOT / "configs",
                experiment_id="paper_v0_native",
                output_dir=artifact_dir,
            )

            index = build_artifact_index(artifact_dir, index_json, index_csv)

            self.assertEqual(index["record_type"], "artifact_index")
            self.assertEqual(index["num_records"], 9)
            self.assertEqual(index["num_valid"], 9)
            self.assertTrue(index_json.exists())
            self.assertTrue(index_csv.exists())
            records = json.loads(index_json.read_text())["records"]
            self.assertEqual(
                {record["record_type"] for record in records},
                {"native_probe_summary", "run_manifest"},
            )

    def test_index_records_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            artifact_dir = tmpdir / "artifacts"
            artifact_dir.mkdir()
            (artifact_dir / "broken.json").write_text("{not json", encoding="utf-8")
            index = build_artifact_index(
                artifact_dir,
                tmpdir / "artifact_index.json",
            )
            self.assertEqual(index["num_records"], 1)
            self.assertEqual(index["num_valid"], 0)
            self.assertFalse(index["records"][0]["valid"])

    def test_index_array_contract_validation_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            artifact_dir = tmpdir / "artifacts"
            artifact_dir.mkdir()
            features_path = artifact_dir / "features.npz"
            manifest_path = artifact_dir / "manifest.jsonl"
            np.savez_compressed(features_path, features=np.ones((1, 2), dtype=np.float32))
            manifest_path.write_text(
                json.dumps({"image": "a.jpg", "split": "val", "label": 0}) + "\n",
                encoding="utf-8",
            )
            validate_array_contract(
                npz_path=features_path,
                kind="features",
                task_type="classification",
                manifest_path=manifest_path,
                expected_split="val",
                write_json=artifact_dir / "array_validation.json",
            )

            index = build_artifact_index(
                artifact_dir,
                tmpdir / "artifact_index.json",
                require_valid=True,
            )
            self.assertEqual(index["num_records"], 1)
            self.assertEqual(index["records"][0]["record_type"], "array_contract_validation")

    def test_require_valid_rejects_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            artifact_dir = tmpdir / "artifacts"
            artifact_dir.mkdir()
            (artifact_dir / "broken.json").write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_artifact_index(
                    artifact_dir,
                    tmpdir / "artifact_index.json",
                    require_valid=True,
                )

    def test_indexer_skips_existing_output_json_on_rerun(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            artifact_dir = tmpdir / "artifacts"
            index_json = artifact_dir / "index" / "artifact_index.json"
            index_csv = artifact_dir / "index" / "artifact_index.csv"
            write_smoke_probe_artifacts(
                config_root=PUBLIC_REPRO_ROOT / "configs",
                experiment_id="paper_v0_native",
                output_dir=artifact_dir,
            )

            first = build_artifact_index(
                artifact_dir,
                index_json,
                index_csv,
                require_valid=True,
            )
            second = build_artifact_index(
                artifact_dir,
                index_json,
                index_csv,
                require_valid=True,
            )

            self.assertEqual(first["num_records"], second["num_records"])
            self.assertEqual(second["num_valid"], second["num_records"])

    def test_indexer_skips_packaged_index_release_manifest_and_sanitization_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            artifact_dir = tmpdir / "artifacts"
            write_smoke_probe_artifacts(
                config_root=PUBLIC_REPRO_ROOT / "configs",
                experiment_id="paper_v0_native",
                output_dir=artifact_dir,
            )
            (artifact_dir / "index").mkdir()
            (artifact_dir / "index" / "artifact_index.json").write_text(
                json.dumps({"record_type": "artifact_index", "records": []}) + "\n",
                encoding="utf-8",
            )
            (artifact_dir / "bundle_release_manifest.json").write_text(
                json.dumps({"record_type": "artifact_release_manifest"}) + "\n",
                encoding="utf-8",
            )
            (artifact_dir / "artifact_export_manifest.json").write_text(
                json.dumps({"record_type": "artifact_export_manifest_template"}) + "\n",
                encoding="utf-8",
            )
            (artifact_dir / "index" / "artifact_metadata_sanitization_report.json").write_text(
                json.dumps({"record_type": "artifact_metadata_sanitization_report"}) + "\n",
                encoding="utf-8",
            )

            index = build_artifact_index(
                artifact_dir,
                tmpdir / "new_index" / "artifact_index.json",
                require_valid=True,
            )

            self.assertEqual(index["num_records"], 9)
            self.assertEqual(index["num_valid"], 9)


if __name__ == "__main__":
    unittest.main()
