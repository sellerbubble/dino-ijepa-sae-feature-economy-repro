import unittest
from pathlib import Path

from feature_economy.data import ManifestDataset, ManifestError


FIXTURE_DIR = Path(__file__).parent / "fixtures"


class DatasetAdapterTests(unittest.TestCase):
    def test_manifest_dataset_reads_depth_records(self):
        dataset = ManifestDataset(
            FIXTURE_DIR / "tiny_nyuv2_manifest.jsonl",
            task_type="dense_depth",
            expected_split="val",
        )
        self.assertEqual(len(dataset), 2)
        first = dataset[0]
        self.assertEqual(first.image, "images/nyu_0001.jpg")
        self.assertEqual(first.depth, "depth/nyu_0001.npy")
        self.assertEqual(first.split, "val")
        self.assertIsNone(first.label)
        self.assertEqual(
            dataset.image_path(0),
            FIXTURE_DIR / "images/nyu_0001.jpg",
        )

    def test_manifest_dataset_reuses_validation(self):
        with self.assertRaises(ManifestError):
            ManifestDataset(
                FIXTURE_DIR / "tiny_nyuv2_manifest.jsonl",
                task_type="dense_depth",
                expected_split="train",
            )


if __name__ == "__main__":
    unittest.main()
