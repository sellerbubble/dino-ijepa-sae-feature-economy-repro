import tempfile
import unittest
from pathlib import Path

from feature_economy.data import ManifestError, validate_manifest


FIXTURE_DIR = Path(__file__).parent / "fixtures"


class ManifestTests(unittest.TestCase):
    def test_validate_dense_depth_fixture(self):
        count = validate_manifest(
            FIXTURE_DIR / "tiny_nyuv2_manifest.jsonl",
            task_type="dense_depth",
            expected_split="val",
        )
        self.assertEqual(count, 2)

    def test_rejects_missing_task_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            path.write_text('{"image": "x.jpg", "split": "val"}\n', encoding="utf-8")
            with self.assertRaises(ManifestError):
                validate_manifest(path, task_type="dense_depth", expected_split="val")

    def test_rejects_wrong_split(self):
        with self.assertRaises(ManifestError):
            validate_manifest(
                FIXTURE_DIR / "tiny_nyuv2_manifest.jsonl",
                task_type="dense_depth",
                expected_split="train",
            )


if __name__ == "__main__":
    unittest.main()
