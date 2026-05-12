import json
import tempfile
import unittest
from pathlib import Path

from feature_economy.artifacts import check_artifact_bundle


class ArtifactBundleTests(unittest.TestCase):
    def test_check_complete_bundle_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_path = root / "run_plan.json"
            artifact_dir = root / "probes" / "native" / "dino" / "imagenet" / "val"
            artifact_dir.mkdir(parents=True)
            for filename in [
                "native_probe_summary.json",
                "probe_logits.npz",
                "run_manifest.json",
            ]:
                (artifact_dir / filename).write_text("{}", encoding="utf-8")
            _write_plan(
                plan_path,
                stage="native_probe",
                artifact_dir="${ARTIFACT_ROOT}/probes/native/dino/imagenet/val",
            )

            report = check_artifact_bundle(
                run_plan_json=plan_path,
                artifact_root=root,
                output_json=root / "bundle_check.json",
            )
            self.assertTrue(report["complete"])
            self.assertEqual(report["complete_rows"], 1)
            self.assertTrue((root / "bundle_check.json").exists())

    def test_check_bundle_reports_missing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plan_path = root / "run_plan.json"
            (root / "analysis" / "ranking" / "dino").mkdir(parents=True)
            _write_plan(
                plan_path,
                stage="feature_ranking",
                artifact_dir="${ARTIFACT_ROOT}/analysis/ranking/dino",
            )

            report = check_artifact_bundle(run_plan_json=plan_path, artifact_root=root)
            self.assertFalse(report["complete"])
            self.assertEqual(report["missing_rows"], 1)
            self.assertEqual(
                set(report["rows"][0]["missing_files"]),
                {"task_feature_ranking.json", "run_manifest.json"},
            )


def _write_plan(path: Path, *, stage: str, artifact_dir: str) -> None:
    payload = {
        "record_type": "reproduction_run_plan",
        "num_rows": 1,
        "rows": [
            {
                "stage": stage,
                "task_id": "imagenet_1k",
                "model_id": "dino_v2_base",
                "sae_id": "",
                "artifact_dir": artifact_dir,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
