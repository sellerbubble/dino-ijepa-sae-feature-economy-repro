import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_economy.analysis import compute_linear_probe_contribution_scores
from feature_economy.artifacts import (
    build_artifact_index,
    validate_array_contract,
    validate_contribution_scores_summary,
)


class ContributionTests(unittest.TestCase):
    def test_compute_classification_contribution_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path = tmpdir / "codes.npz"
            probe_path = tmpdir / "probe_logits.npz"
            np.savez(
                codes_path,
                codes=np.asarray(
                    [
                        [2.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 2.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            )
            np.savez(
                probe_path,
                weights=np.asarray(
                    [
                        [2.0, -2.0],
                        [-2.0, 2.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                logits=np.zeros((4, 2), dtype=np.float32),
                classes=np.asarray([0, 1], dtype=np.int64),
                labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
            )

            summary_path = compute_linear_probe_contribution_scores(
                codes_npz=codes_path,
                probe_logits_npz=probe_path,
                task_type="classification",
                task_id="tiny_cls",
                model_id="dino_v2_base",
                sae_id="dino_l11_topk32_exp4",
                output_dir=tmpdir / "contribution",
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            validate_contribution_scores_summary(summary)
            self.assertEqual(summary["primary_metric"], "top1")
            self.assertEqual(summary["num_features"], 3)
            scores_path = tmpdir / "contribution" / "contribution_scores.npz"
            scores = np.load(scores_path)["validation_contribution_score"]
            self.assertEqual(scores.shape, (3,))
            self.assertGreater(float(np.max(scores[:2])), float(scores[2]))
            report = validate_array_contract(
                npz_path=scores_path,
                kind="contribution_scores",
            )
            self.assertTrue(report["valid"])
            index = build_artifact_index(
                tmpdir / "contribution",
                tmpdir / "artifact_index.json",
                require_valid=True,
            )
            self.assertEqual(index["num_valid"], 2)

    def test_compute_fast_classification_contribution_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path = tmpdir / "codes.npz"
            probe_path = tmpdir / "probe_logits.npz"
            np.savez(
                codes_path,
                codes=np.asarray(
                    [
                        [2.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 2.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
            )
            np.savez(
                probe_path,
                weights=np.asarray(
                    [
                        [2.0, -2.0],
                        [-2.0, 2.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                logits=np.zeros((4, 2), dtype=np.float32),
                classes=np.asarray([0, 1], dtype=np.int64),
                labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
            )

            summary_path = compute_linear_probe_contribution_scores(
                codes_npz=codes_path,
                probe_logits_npz=probe_path,
                task_type="classification",
                task_id="tiny_cls",
                model_id="dino_v2_base",
                sae_id="dino_l11_topk32_exp4",
                output_dir=tmpdir / "contribution",
                scoring_method="true_class_logit_drop",
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["scoring_method"], "true_class_logit_drop")
            scores = np.load(tmpdir / "contribution" / "contribution_scores.npz")[
                "validation_contribution_score"
            ]
            self.assertEqual(scores.shape, (3,))
            self.assertGreater(float(scores[0]), 0.0)
            self.assertGreater(float(scores[1]), 0.0)
            self.assertEqual(float(scores[2]), 0.0)

    def test_compute_classification_contribution_scores_with_pooled_token_codes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path = tmpdir / "codes.npz"
            probe_path = tmpdir / "probe_logits.npz"
            codes = np.asarray(
                [
                    [[2.0, 0.0], [2.0, 0.0]],
                    [[2.0, 0.0], [2.0, 0.0]],
                    [[0.0, 2.0], [0.0, 2.0]],
                    [[0.0, 2.0], [0.0, 2.0]],
                ],
                dtype=np.float32,
            )
            weights = np.asarray(
                [[2.0, -2.0], [-2.0, 2.0], [0.0, 0.0]],
                dtype=np.float32,
            )
            np.savez(codes_path, codes=codes)
            np.savez(
                probe_path,
                weights=weights,
                logits=np.zeros((4, 2), dtype=np.float32),
                classes=np.asarray([0, 1], dtype=np.int64),
                labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
            )

            summary_path = compute_linear_probe_contribution_scores(
                codes_npz=codes_path,
                probe_logits_npz=probe_path,
                task_type="classification",
                task_id="tiny_cls",
                model_id="dino_v2_base",
                sae_id="dino_l11_topk32_exp4",
                output_dir=tmpdir / "contribution",
                scoring_method="true_class_logit_drop",
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["scored_features"], 2)
            scores = np.load(tmpdir / "contribution" / "contribution_scores.npz")[
                "validation_contribution_score"
            ]
            self.assertGreater(float(scores[0]), 0.0)
            self.assertGreater(float(scores[1]), 0.0)

    def test_compute_dense_weight_activation_contribution_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path = tmpdir / "codes.npz"
            probe_path = tmpdir / "probe_logits.npz"
            codes = np.zeros((2, 2, 2, 3), dtype=np.float32)
            codes[..., 0] = 2.0
            codes[..., 1] = 0.5
            np.savez(codes_path, codes=codes)
            np.savez(
                probe_path,
                weights=np.asarray([3.0, 1.0, 0.0, 0.1], dtype=np.float32),
                prediction=np.zeros((2, 2, 2), dtype=np.float32),
                targets=np.ones((2, 2, 2), dtype=np.float32),
            )

            summary_path = compute_linear_probe_contribution_scores(
                codes_npz=codes_path,
                probe_logits_npz=probe_path,
                task_type="dense_depth",
                task_id="nyuv2_depth",
                model_id="dino_v2_base",
                sae_id="dino_l11_topk32_exp4",
                output_dir=tmpdir / "contribution",
                scoring_method="weight_activation",
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["scoring_method"], "weight_activation")
            scores = np.load(tmpdir / "contribution" / "contribution_scores.npz")[
                "validation_contribution_score"
            ]
            self.assertEqual(scores.shape, (3,))
            self.assertGreater(float(scores[0]), float(scores[1]))
            self.assertEqual(float(scores[2]), 0.0)


if __name__ == "__main__":
    unittest.main()
