import unittest

import numpy as np

from feature_economy.probes import accuracy, depth_metrics, segmentation_metrics, topk_accuracy


class MetricTests(unittest.TestCase):
    def test_topk_accuracy(self):
        logits = np.array(
            [
                [0.9, 0.1, 0.0],
                [0.1, 0.2, 0.7],
                [0.4, 0.5, 0.1],
            ]
        )
        labels = np.array([0, 1, 2])
        scores = topk_accuracy(logits, labels, topk=(1, 2))
        self.assertAlmostEqual(scores["top1"], 1 / 3)
        self.assertAlmostEqual(scores["top2"], 2 / 3)

    def test_accuracy(self):
        self.assertAlmostEqual(
            accuracy(np.array([1, 2, 3]), np.array([1, 0, 3])),
            2 / 3,
        )

    def test_depth_metrics(self):
        prediction = np.array([[1.0, 2.0], [3.0, 4.0]])
        target = np.array([[1.0, 1.0], [3.0, 2.0]])
        scores = depth_metrics(prediction, target)
        self.assertAlmostEqual(scores["rmse"], np.sqrt(1.25))
        self.assertAlmostEqual(scores["abs_rel"], 0.5)
        self.assertAlmostEqual(scores["delta1"], 0.5)

    def test_segmentation_metrics(self):
        prediction = np.array([[0, 1], [1, 1]])
        target = np.array([[0, 1], [0, 1]])
        scores = segmentation_metrics(prediction, target, num_classes=2)
        self.assertAlmostEqual(scores["pixel_accuracy"], 0.75)
        self.assertAlmostEqual(scores["miou"], (0.5 + (2 / 3)) / 2)

    def test_segmentation_ignore_index(self):
        prediction = np.array([[0, 1], [1, 1]])
        target = np.array([[0, 1], [255, 1]])
        scores = segmentation_metrics(prediction, target, num_classes=2, ignore_index=255)
        self.assertAlmostEqual(scores["pixel_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
