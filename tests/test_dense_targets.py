import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_economy.data.targets import export_dense_targets


class DenseTargetExportTests(unittest.TestCase):
    def test_export_depth_targets_infers_shape_from_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            depth_dir = tmpdir / "depth"
            depth_dir.mkdir()
            np.save(depth_dir / "a.npy", np.arange(16, dtype=np.float32).reshape(4, 4))
            np.save(depth_dir / "b.npy", np.ones((4, 4), dtype=np.float32))
            manifest_path = tmpdir / "manifest.jsonl"
            rows = [
                {"image": "a.jpg", "depth": "depth/a.npy", "split": "val"},
                {"image": "b.jpg", "depth": "depth/b.npy", "split": "val"},
            ]
            manifest_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            features_path = tmpdir / "features.npz"
            np.savez_compressed(
                features_path,
                features=np.ones((2, 2, 2, 3), dtype=np.float32),
            )

            summary_path = export_dense_targets(
                manifest_path=manifest_path,
                task_type="dense_depth",
                output_dir=tmpdir / "targets",
                expected_split="val",
                features_npz=features_path,
            )

            record = json.loads(summary_path.read_text())
            self.assertEqual(record["target_shape"], [2, 2, 2])
            arrays = np.load(summary_path.parent / "targets.npz")
            self.assertEqual(arrays["targets"].shape, (2, 2, 2))
            self.assertTrue(np.issubdtype(arrays["targets"].dtype, np.floating))

    def test_export_segmentation_targets_uses_integer_dtype(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            seg_dir = tmpdir / "seg"
            seg_dir.mkdir()
            np.save(seg_dir / "a.npy", np.asarray([[0, 1], [2, 3]], dtype=np.int64))
            manifest_path = tmpdir / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps({"image": "a.jpg", "segmentation": "seg/a.npy", "split": "val"})
                + "\n",
                encoding="utf-8",
            )

            summary_path = export_dense_targets(
                manifest_path=manifest_path,
                task_type="dense_segmentation",
                output_dir=tmpdir / "targets",
                expected_split="val",
                target_shape=(2, 2),
            )

            arrays = np.load(summary_path.parent / "targets.npz")
            self.assertEqual(arrays["targets"].shape, (1, 2, 2))
            self.assertTrue(np.issubdtype(arrays["targets"].dtype, np.integer))

    def test_export_segmentation_targets_can_remap_official_ade20k_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            seg_dir = tmpdir / "seg"
            seg_dir.mkdir()
            np.save(seg_dir / "a.npy", np.asarray([[0, 1], [2, 150]], dtype=np.int64))
            manifest_path = tmpdir / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps({"image": "a.jpg", "segmentation": "seg/a.npy", "split": "val"})
                + "\n",
                encoding="utf-8",
            )

            summary_path = export_dense_targets(
                manifest_path=manifest_path,
                task_type="dense_segmentation",
                output_dir=tmpdir / "targets",
                expected_split="val",
                target_shape=(2, 2),
                segmentation_ignore_value=0,
                segmentation_label_offset=-1,
                segmentation_output_ignore_index=255,
            )

            arrays = np.load(summary_path.parent / "targets.npz")
            np.testing.assert_array_equal(
                arrays["targets"],
                np.asarray([[[255, 0], [1, 149]]], dtype=np.int64),
            )
            record = json.loads(summary_path.read_text())
            self.assertEqual(record["segmentation_ignore_value"], 0)
            self.assertEqual(record["segmentation_label_offset"], -1)
            self.assertEqual(record["segmentation_output_ignore_index"], 255)


if __name__ == "__main__":
    unittest.main()
