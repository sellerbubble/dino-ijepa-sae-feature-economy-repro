import json
import tempfile
import unittest
from pathlib import Path

from feature_economy.artifacts import validate_probe_summary, validate_run_manifest
from feature_economy.probes import write_smoke_probe_artifacts


PUBLIC_REPRO_ROOT = Path(__file__).resolve().parents[1]


class SmokeProbeTests(unittest.TestCase):
    def test_write_native_smoke_probe_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            written = write_smoke_probe_artifacts(
                config_root=PUBLIC_REPRO_ROOT / "configs",
                experiment_id="paper_v0_native",
                output_dir=tmpdir,
            )
            self.assertEqual(len(written), 9)
            summaries = [path for path in written if path.name != "run_manifest.json"]
            self.assertEqual(len(summaries), 8)
            for path in summaries:
                validate_probe_summary(json.loads(path.read_text()))
            validate_run_manifest(json.loads((Path(tmpdir) / "run_manifest.json").read_text()))

    def test_write_sae_smoke_probe_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            written = write_smoke_probe_artifacts(
                config_root=PUBLIC_REPRO_ROOT / "configs",
                experiment_id="paper_v0_sae",
                output_dir=tmpdir,
            )
            summaries = [path for path in written if path.name != "run_manifest.json"]
            self.assertEqual(len(summaries), 8)
            for path in summaries:
                record = json.loads(path.read_text())
                self.assertEqual(record["record_type"], "sae_probe_summary")
                self.assertIn("sae_id", record)
                self.assertTrue(record["smoke"])


if __name__ == "__main__":
    unittest.main()
