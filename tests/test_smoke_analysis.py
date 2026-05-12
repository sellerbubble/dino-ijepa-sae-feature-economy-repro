import json
import tempfile
import unittest
from pathlib import Path

from feature_economy.analysis import write_smoke_analysis_artifacts
from feature_economy.artifacts import build_artifact_index


PUBLIC_REPRO_ROOT = Path(__file__).resolve().parents[1]


class SmokeAnalysisTests(unittest.TestCase):
    def test_write_smoke_analysis_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            artifact_dir = tmpdir / "analysis"
            written = write_smoke_analysis_artifacts(
                config_root=PUBLIC_REPRO_ROOT / "configs",
                output_dir=artifact_dir,
            )
            self.assertEqual(len(written), 27)
            index = build_artifact_index(
                artifact_dir,
                tmpdir / "artifact_index.json",
            )
            self.assertEqual(index["num_records"], 27)
            self.assertEqual(index["num_valid"], 27)
            record_types = {record["record_type"] for record in index["records"]}
            self.assertEqual(
                record_types,
                {
                    "availability_summary",
                    "task_feature_ranking",
                    "subset_usage_summary",
                    "feature_ablation_summary",
                    "run_manifest",
                },
            )
            manifest = json.loads((artifact_dir / "run_manifest.json").read_text())
            self.assertTrue(manifest["smoke"])

    def test_write_smoke_analysis_artifact_subsets(self):
        cases = [
            ({"availability"}, {"availability_summary", "run_manifest"}, 3),
            ({"ranking"}, {"task_feature_ranking", "run_manifest"}, 9),
            ({"subset_usage"}, {"subset_usage_summary", "run_manifest"}, 9),
            ({"ablation"}, {"feature_ablation_summary", "run_manifest"}, 9),
        ]
        for include, expected_types, expected_count in cases:
            with self.subTest(include=include):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir = Path(tmpdir)
                    artifact_dir = tmpdir / "analysis"
                    written = write_smoke_analysis_artifacts(
                        config_root=PUBLIC_REPRO_ROOT / "configs",
                        output_dir=artifact_dir,
                        include=include,
                    )
                    self.assertEqual(len(written), expected_count)
                    index = build_artifact_index(
                        artifact_dir,
                        tmpdir / "artifact_index.json",
                    )
                    self.assertEqual(index["num_valid"], expected_count)
                    self.assertEqual(
                        {record["record_type"] for record in index["records"]},
                        expected_types,
                    )


if __name__ == "__main__":
    unittest.main()
