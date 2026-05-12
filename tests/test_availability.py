import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_economy.analysis import compute_availability_from_codes
from feature_economy.artifacts import build_artifact_index, validate_availability_summary


class AvailabilityTests(unittest.TestCase):
    def test_compute_availability_from_codes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path = tmpdir / "codes.npz"
            np.savez(
                codes_path,
                codes=np.asarray(
                    [
                        [1.0, 0.0, 0.0, 2.0],
                        [0.5, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 3.0],
                    ],
                    dtype=np.float32,
                ),
            )
            summary_path = compute_availability_from_codes(
                codes_npz=codes_path,
                model_id="dino_v2_base",
                sae_id="dino_l11_topk32_exp4",
                dataset_id="tiny",
                split="val",
                output_dir=tmpdir / "availability",
            )
            record = json.loads(summary_path.read_text())
            validate_availability_summary(record)
            self.assertEqual(record["total_features"], 4)
            self.assertEqual(record["fired_features"], 2)
            self.assertEqual(record["dead_features"], 2)
            self.assertEqual(record["buckets"]["dead"], 2)
            self.assertEqual(record["buckets"]["rare"], 2)
            index = build_artifact_index(
                tmpdir / "availability",
                tmpdir / "artifact_index.json",
                require_valid=True,
            )
            self.assertEqual(index["num_valid"], 2)

    def test_compute_availability_requires_codes_array(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            bad_path = tmpdir / "bad.npz"
            np.savez(bad_path, features=np.ones((2, 3), dtype=np.float32))
            with self.assertRaises(ValueError):
                compute_availability_from_codes(
                    codes_npz=bad_path,
                    model_id="dino_v2_base",
                    sae_id="dino_l11_topk32_exp4",
                    dataset_id="tiny",
                    split="val",
                    output_dir=tmpdir / "availability",
                )


if __name__ == "__main__":
    unittest.main()
