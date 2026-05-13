import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_economy.artifacts import build_artifact_index, validate_probe_summary
from feature_economy.probes import (
    train_native_linear_probe,
    train_native_paper_scale_probe,
    train_sae_linear_probe,
    train_sae_paper_scale_probe,
)

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class LinearProbeTests(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "paper-scale torch probe tests require PyTorch")
    def test_train_native_paper_scale_probe_classification(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            train_features_path = tmpdir / "train_features.npz"
            val_features_path = tmpdir / "val_features.npz"
            train_manifest_path = tmpdir / "train_manifest.jsonl"
            val_manifest_path = tmpdir / "val_manifest.jsonl"
            features = np.asarray(
                [
                    [3.0, 0.0],
                    [2.5, 0.1],
                    [0.0, 3.0],
                    [0.1, 2.5],
                ],
                dtype=np.float32,
            )
            np.savez(train_features_path, features=features)
            np.savez(val_features_path, features=features)
            train_rows = [
                {"image": "a.jpg", "split": "train", "label": 0},
                {"image": "b.jpg", "split": "train", "label": 0},
                {"image": "c.jpg", "split": "train", "label": 1},
                {"image": "d.jpg", "split": "train", "label": 1},
            ]
            val_rows = [{**row, "split": "val"} for row in train_rows]
            train_manifest_path.write_text(
                "\n".join(json.dumps(row) for row in train_rows) + "\n",
                encoding="utf-8",
            )
            val_manifest_path.write_text(
                "\n".join(json.dumps(row) for row in val_rows) + "\n",
                encoding="utf-8",
            )
            summary_path = train_native_paper_scale_probe(
                train_features_npz=train_features_path,
                train_manifest_path=train_manifest_path,
                val_features_npz=val_features_path,
                val_manifest_path=val_manifest_path,
                task_type="classification",
                task_id="tiny_cls",
                model_id="dino_v2_base",
                output_dir=tmpdir / "probe",
                epochs=20,
                batch_size=2,
                lr=0.05,
                device="cpu",
            )
            record = json.loads(summary_path.read_text())
            validate_probe_summary(record)
            self.assertEqual(record["backend"], "paper_scale_torch")
            self.assertEqual(record["selection"]["checkpoint_rule"], "best_validation_top1")
            self.assertAlmostEqual(record["metrics"]["top1"], 1.0)
            self.assertTrue((tmpdir / "probe" / "probe.pt").exists())
            self.assertTrue((tmpdir / "probe" / "probe_outputs.npz").exists())
            self.assertTrue((tmpdir / "probe" / "probe_logits.npz").exists())

    @unittest.skipUnless(HAS_TORCH, "paper-scale torch probe tests require PyTorch")
    def test_train_sae_paper_scale_probe_counting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            train_codes_path = tmpdir / "train_codes.npz"
            val_codes_path = tmpdir / "val_codes.npz"
            train_manifest_path = tmpdir / "train_manifest.jsonl"
            val_manifest_path = tmpdir / "val_manifest.jsonl"
            codes = np.asarray(
                [
                    [[3.0, 0.0], [3.0, 0.0]],
                    [[2.5, 0.1], [2.5, 0.1]],
                    [[0.0, 3.0], [0.0, 3.0]],
                    [[0.1, 2.5], [0.1, 2.5]],
                ],
                dtype=np.float32,
            )
            np.savez(train_codes_path, codes=codes)
            np.savez(val_codes_path, codes=codes)
            train_rows = [
                {"image": "a.jpg", "split": "train", "label": 0},
                {"image": "b.jpg", "split": "train", "label": 0},
                {"image": "c.jpg", "split": "train", "label": 1},
                {"image": "d.jpg", "split": "train", "label": 1},
            ]
            val_rows = [{**row, "split": "val"} for row in train_rows]
            train_manifest_path.write_text(
                "\n".join(json.dumps(row) for row in train_rows) + "\n",
                encoding="utf-8",
            )
            val_manifest_path.write_text(
                "\n".join(json.dumps(row) for row in val_rows) + "\n",
                encoding="utf-8",
            )
            summary_path = train_sae_paper_scale_probe(
                train_codes_npz=train_codes_path,
                train_manifest_path=train_manifest_path,
                val_codes_npz=val_codes_path,
                val_manifest_path=val_manifest_path,
                task_type="count_classification",
                task_id="tiny_count",
                model_id="ijepa_vit_h14",
                sae_id="ijepa_l31_topk32_exp4",
                output_dir=tmpdir / "probe",
                epochs=20,
                batch_size=2,
                lr=0.05,
                device="cpu",
            )
            record = json.loads(summary_path.read_text())
            validate_probe_summary(record)
            self.assertEqual(record["record_type"], "sae_probe_summary")
            self.assertEqual(record["backend"], "paper_scale_torch")
            self.assertEqual(record["selection"]["checkpoint_rule"], "best_validation_accuracy")
            self.assertAlmostEqual(record["metrics"]["accuracy"], 1.0)
            probe = np.load(tmpdir / "probe" / "probe_outputs.npz")
            self.assertEqual(probe["weights"].shape[0], codes.shape[-1] + 1)

    def test_train_sae_linear_probe_classification(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path = tmpdir / "codes.npz"
            manifest_path = tmpdir / "manifest.jsonl"
            np.savez(
                codes_path,
                codes=np.asarray(
                    [
                        [2.0, 0.0],
                        [1.5, 0.1],
                        [0.0, 2.0],
                        [0.1, 1.5],
                    ],
                    dtype=np.float32,
                ),
            )
            rows = [
                {"image": "a.jpg", "split": "val", "label": 0},
                {"image": "b.jpg", "split": "val", "label": 0},
                {"image": "c.jpg", "split": "val", "label": 1},
                {"image": "d.jpg", "split": "val", "label": 1},
            ]
            manifest_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            summary_path = train_sae_linear_probe(
                codes_npz=codes_path,
                manifest_path=manifest_path,
                task_type="classification",
                task_id="tiny_cls",
                model_id="dino_v2_base",
                sae_id="dino_l11_topk32_exp4",
                output_dir=tmpdir / "probe",
                expected_split="val",
            )
            record = json.loads(summary_path.read_text())
            validate_probe_summary(record)
            self.assertEqual(record["record_type"], "sae_probe_summary")
            self.assertAlmostEqual(record["metrics"]["top1"], 1.0)
            self.assertFalse(record["fixture"])
            index = build_artifact_index(
                tmpdir / "probe",
                tmpdir / "artifact_index.json",
                require_valid=True,
            )
            self.assertEqual(index["num_valid"], 2)

    def test_train_native_linear_probe_classification(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            features_path = tmpdir / "features.npz"
            manifest_path = tmpdir / "manifest.jsonl"
            np.savez(
                features_path,
                features=np.asarray(
                    [
                        [2.0, 0.0],
                        [1.5, 0.1],
                        [0.0, 2.0],
                        [0.1, 1.5],
                    ],
                    dtype=np.float32,
                ),
            )
            rows = [
                {"image": "a.jpg", "split": "val", "label": 0},
                {"image": "b.jpg", "split": "val", "label": 0},
                {"image": "c.jpg", "split": "val", "label": 1},
                {"image": "d.jpg", "split": "val", "label": 1},
            ]
            manifest_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            summary_path = train_native_linear_probe(
                features_npz=features_path,
                manifest_path=manifest_path,
                task_type="classification",
                task_id="tiny_cls",
                model_id="dino_v2_base",
                output_dir=tmpdir / "probe",
                expected_split="val",
            )
            record = json.loads(summary_path.read_text())
            validate_probe_summary(record)
            self.assertEqual(record["record_type"], "native_probe_summary")
            self.assertNotIn("sae_id", record)
            self.assertAlmostEqual(record["metrics"]["top1"], 1.0)

    def test_train_sae_linear_probe_pools_token_codes_for_classification(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path = tmpdir / "codes.npz"
            manifest_path = tmpdir / "manifest.jsonl"
            codes = np.asarray(
                [
                    [[2.0, 0.0], [2.0, 0.0]],
                    [[1.5, 0.1], [1.5, 0.1]],
                    [[0.0, 2.0], [0.0, 2.0]],
                    [[0.1, 1.5], [0.1, 1.5]],
                ],
                dtype=np.float32,
            )
            np.savez(codes_path, codes=codes)
            rows = [
                {"image": "a.jpg", "split": "val", "label": 0},
                {"image": "b.jpg", "split": "val", "label": 0},
                {"image": "c.jpg", "split": "val", "label": 1},
                {"image": "d.jpg", "split": "val", "label": 1},
            ]
            manifest_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            summary_path = train_sae_linear_probe(
                codes_npz=codes_path,
                manifest_path=manifest_path,
                task_type="classification",
                task_id="tiny_cls",
                model_id="dino_v2_base",
                sae_id="dino_l11_topk32_exp4",
                output_dir=tmpdir / "probe",
                expected_split="val",
            )
            record = json.loads(summary_path.read_text())
            self.assertAlmostEqual(record["metrics"]["top1"], 1.0)
            probe = np.load(tmpdir / "probe" / "probe_logits.npz")
            self.assertEqual(probe["weights"].shape[0], codes.shape[-1] + 1)

    def test_train_sae_linear_probe_dense_depth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path = tmpdir / "codes.npz"
            targets_path = tmpdir / "targets.npz"
            manifest_path = tmpdir / "manifest.jsonl"
            targets = np.asarray(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[2.0, 3.0], [4.0, 5.0]],
                ],
                dtype=np.float32,
            )
            codes = targets[..., None]
            np.savez(codes_path, codes=codes)
            np.savez(targets_path, targets=targets)
            rows = [
                {"image": "a.jpg", "depth": "a.npy", "split": "val"},
                {"image": "b.jpg", "depth": "b.npy", "split": "val"},
            ]
            manifest_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            summary_path = train_sae_linear_probe(
                codes_npz=codes_path,
                manifest_path=manifest_path,
                task_type="dense_depth",
                task_id="tiny_depth",
                model_id="dino_v2_base",
                sae_id="dino_l11_topk32_exp4",
                output_dir=tmpdir / "probe",
                expected_split="val",
                targets_npz=targets_path,
                ridge=0.0,
            )
            record = json.loads(summary_path.read_text())
            validate_probe_summary(record)
            self.assertLess(record["metrics"]["rmse"], 1e-4)

    def test_train_native_linear_probe_dense_segmentation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            features_path = tmpdir / "features.npz"
            targets_path = tmpdir / "targets.npz"
            manifest_path = tmpdir / "manifest.jsonl"
            targets = np.asarray(
                [
                    [[0, 1], [0, 1]],
                    [[1, 0], [1, 0]],
                ],
                dtype=np.int64,
            )
            features = np.stack([1 - targets, targets], axis=-1).astype(np.float32)
            np.savez(features_path, features=features)
            np.savez(targets_path, targets=targets)
            rows = [
                {"image": "a.jpg", "segmentation": "a.png", "split": "val"},
                {"image": "b.jpg", "segmentation": "b.png", "split": "val"},
            ]
            manifest_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            summary_path = train_native_linear_probe(
                features_npz=features_path,
                manifest_path=manifest_path,
                task_type="dense_segmentation",
                task_id="tiny_seg",
                model_id="dino_v2_base",
                output_dir=tmpdir / "probe",
                expected_split="val",
                targets_npz=targets_path,
                num_classes=2,
            )
            record = json.loads(summary_path.read_text())
            validate_probe_summary(record)
            self.assertAlmostEqual(record["metrics"]["pixel_accuracy"], 1.0)
            self.assertAlmostEqual(record["metrics"]["miou"], 1.0)

    def test_linear_probe_rejects_row_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path = tmpdir / "codes.npz"
            manifest_path = tmpdir / "manifest.jsonl"
            np.savez(codes_path, codes=np.ones((2, 3), dtype=np.float32))
            manifest_path.write_text(
                json.dumps({"image": "a.jpg", "split": "val", "label": 0}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                train_sae_linear_probe(
                    codes_npz=codes_path,
                    manifest_path=manifest_path,
                    task_type="classification",
                    task_id="tiny_cls",
                    model_id="dino_v2_base",
                    sae_id="dino_l11_topk32_exp4",
                    output_dir=tmpdir / "probe",
                    expected_split="val",
                )


if __name__ == "__main__":
    unittest.main()
