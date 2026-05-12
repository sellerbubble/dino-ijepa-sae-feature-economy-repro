import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from feature_economy.artifacts import build_artifact_index
from feature_economy.analysis import write_smoke_analysis_artifacts
from feature_economy.paper import (
    write_all_tables_from_index,
    write_probe_score_table,
    write_probe_score_table_from_index,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
PUBLIC_REPRO_ROOT = Path(__file__).resolve().parents[1]


class PaperTableTests(unittest.TestCase):
    def test_write_probe_score_table_from_fixture_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            artifact_dir = tmpdir / "artifacts"
            output_dir = tmpdir / "tables"
            artifact_dir.mkdir()
            shutil.copy(FIXTURE_DIR / "tiny_probe_summary.json", artifact_dir / "sae.json")
            shutil.copy(
                FIXTURE_DIR / "tiny_native_probe_summary.json",
                artifact_dir / "native.json",
            )

            row_count = write_probe_score_table(
                input_dir=artifact_dir,
                output_csv=output_dir / "probe_scores.csv",
            )

            self.assertEqual(row_count, 6)
            with (output_dir / "probe_scores.csv").open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertEqual(
                {row["record_type"] for row in rows},
                {"native_probe_summary", "sae_probe_summary"},
            )
            self.assertEqual({row["smoke"] for row in rows}, {"False"})
            self.assertIn("source_path", rows[0])

    def test_write_probe_score_table_from_artifact_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            artifact_dir = tmpdir / "artifacts"
            table_dir = tmpdir / "tables"
            index_json = tmpdir / "index" / "artifact_index.json"
            artifact_dir.mkdir()
            shutil.copy(FIXTURE_DIR / "tiny_probe_summary.json", artifact_dir / "sae.json")
            shutil.copy(
                FIXTURE_DIR / "tiny_native_probe_summary.json",
                artifact_dir / "native.json",
            )
            build_artifact_index(artifact_dir, index_json)

            row_count = write_probe_score_table_from_index(
                index_json=index_json,
                output_csv=table_dir / "probe_scores.csv",
            )

            self.assertEqual(row_count, 6)
            with (table_dir / "probe_scores.csv").open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)

    def test_write_all_tables_from_analysis_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            artifact_dir = tmpdir / "analysis"
            index_json = tmpdir / "index" / "artifact_index.json"
            table_dir = tmpdir / "tables"
            write_smoke_analysis_artifacts(PUBLIC_REPRO_ROOT / "configs", artifact_dir)
            build_artifact_index(artifact_dir, index_json)

            counts = write_all_tables_from_index(index_json, table_dir)

            self.assertEqual(counts["probe_scores"], 0)
            self.assertEqual(counts["availability_summary"], 8)
            self.assertEqual(counts["subset_usage"], 16)
            self.assertEqual(counts["ablation_summary"], 16)
            self.assertTrue((table_dir / "availability_summary.csv").exists())
            self.assertTrue((table_dir / "subset_usage_summary.csv").exists())
            self.assertTrue((table_dir / "ablation_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
