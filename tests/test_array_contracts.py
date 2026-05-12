import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_economy.artifacts import validate_array_contract
from feature_economy.artifacts.schemas import SchemaError


class ArrayContractTests(unittest.TestCase):
    def test_validate_features_with_manifest_alignment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            npz_path = tmpdir / "features.npz"
            manifest_path = _write_manifest(
                tmpdir,
                [
                    {"image": "a.jpg", "split": "val", "label": 0},
                    {"image": "b.jpg", "split": "val", "label": 1},
                ],
            )
            np.savez_compressed(npz_path, features=np.ones((2, 3), dtype=np.float32))

            report = validate_array_contract(
                npz_path=npz_path,
                kind="features",
                task_type="classification",
                manifest_path=manifest_path,
                expected_split="val",
            )
            self.assertTrue(report["valid"])
            self.assertEqual(report["arrays"]["features"], [2, 3])
            self.assertEqual(report["num_manifest_rows"], 2)

    def test_validate_codes_rejects_manifest_row_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            npz_path = tmpdir / "codes.npz"
            manifest_path = _write_manifest(
                tmpdir,
                [{"image": "a.jpg", "split": "val", "label": 0}],
            )
            np.savez_compressed(npz_path, codes=np.ones((2, 4), dtype=np.float32))

            with self.assertRaises(SchemaError):
                validate_array_contract(
                    npz_path=npz_path,
                    kind="codes",
                    task_type="classification",
                    manifest_path=manifest_path,
                    expected_split="val",
                )

    def test_validate_classification_probe_logits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            npz_path = tmpdir / "probe_logits.npz"
            np.savez_compressed(
                npz_path,
                logits=np.ones((3, 2), dtype=np.float32),
                weights=np.ones((5, 2), dtype=np.float32),
                classes=np.asarray([0, 1], dtype=np.int64),
                labels=np.asarray([0, 1, 1], dtype=np.int64),
            )

            report = validate_array_contract(
                npz_path=npz_path,
                kind="probe_logits",
                task_type="classification",
            )
            self.assertEqual(report["arrays"]["weights"], [5, 2])

    def test_validate_dense_targets_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            npz_path = tmpdir / "targets.npz"
            manifest_path = _write_manifest(
                tmpdir,
                [
                    {"image": "a.jpg", "split": "val", "depth": "a.npy"},
                    {"image": "b.jpg", "split": "val", "depth": "b.npy"},
                ],
            )
            np.savez_compressed(npz_path, targets=np.ones((2, 4, 4), dtype=np.float32))

            report = validate_array_contract(
                npz_path=npz_path,
                kind="targets",
                task_type="dense_depth",
                manifest_path=manifest_path,
                expected_split="val",
            )
            self.assertEqual(report["arrays"]["targets"], [2, 4, 4])

    def test_validate_dense_segmentation_probe_logits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            npz_path = tmpdir / "probe_logits.npz"
            np.savez_compressed(
                npz_path,
                logits=np.ones((2, 3, 3, 4), dtype=np.float32),
                weights=np.ones((6, 4), dtype=np.float32),
                classes=np.arange(4, dtype=np.int64),
                targets=np.ones((2, 3, 3), dtype=np.int64),
            )

            report = validate_array_contract(
                npz_path=npz_path,
                kind="probe_logits",
                task_type="dense_segmentation",
            )
            self.assertEqual(report["arrays"]["logits"], [2, 3, 3, 4])

    def test_validate_contribution_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            npz_path = tmpdir / "contribution_scores.npz"
            np.savez_compressed(
                npz_path,
                validation_contribution_score=np.asarray([0.0, 0.2, -0.1], dtype=np.float32),
            )

            report = validate_array_contract(
                npz_path=npz_path,
                kind="contribution_scores",
            )
            self.assertEqual(report["arrays"]["validation_contribution_score"], [3])


def _write_manifest(root: Path, rows: list[dict[str, object]]) -> Path:
    manifest_path = root / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return manifest_path


if __name__ == "__main__":
    unittest.main()
