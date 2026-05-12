import json
import tempfile
import unittest
from pathlib import Path

from feature_economy.models import inspect_manifest_images


PUBLIC_REPRO_ROOT = Path(__file__).resolve().parents[1]


class ImageInspectionTests(unittest.TestCase):
    def test_inspect_manifest_images_preprocesses_relative_paths(self):
        try:
            from PIL import Image
        except Exception:  # pragma: no cover - depends on optional dependency
            self.skipTest("Pillow is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_dir = tmpdir / "images"
            image_dir.mkdir()
            Image.new("RGB", (320, 240), color=(100, 120, 140)).save(
                image_dir / "sample.jpg"
            )
            manifest = tmpdir / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "image": "images/sample.jpg",
                        "split": "val",
                        "label": 3,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_json = tmpdir / "inspection.json"
            report = inspect_manifest_images(
                config_root=PUBLIC_REPRO_ROOT / "configs",
                manifest_path=manifest,
                task_type="classification",
                model_id="dino_v2_base",
                expected_split="val",
                output_json=output_json,
                limit=1,
            )
            self.assertEqual(report["artifact_type"], "image_manifest_inspection")
            self.assertEqual(report["num_inspected"], 1)
            self.assertEqual(report["records"][0]["shape"], [3, 224, 224])
            self.assertTrue(output_json.exists())


if __name__ == "__main__":
    unittest.main()
