import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_economy.analysis import (
    compute_subset_usage_from_ranking,
    rank_features_from_linear_probe,
)
from feature_economy.artifacts import (
    build_artifact_index,
    validate_feature_ranking,
    validate_subset_usage_summary,
)
from feature_economy.probes import train_sae_linear_probe


class RankingTests(unittest.TestCase):
    def test_rank_features_from_linear_probe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path = tmpdir / "codes.npz"
            manifest_path = tmpdir / "manifest.jsonl"
            np.savez(
                codes_path,
                codes=np.asarray(
                    [
                        [3.0, 0.0, 0.0],
                        [2.0, 0.0, 0.1],
                        [0.0, 3.0, 0.0],
                        [0.0, 2.0, 0.1],
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
            ranking_path = rank_features_from_linear_probe(
                codes_npz=codes_path,
                probe_logits_npz=tmpdir / "probe" / "probe_logits.npz",
                task_id="tiny_cls",
                model_id="dino_v2_base",
                sae_id="dino_l11_topk32_exp4",
                output_dir=tmpdir / "ranking",
                top_k=2,
            )
            record = json.loads(ranking_path.read_text())
            validate_feature_ranking(record)
            self.assertEqual(record["ranking_method"], "probe_weight")
            self.assertEqual(len(record["rows"]), 2)
            index = build_artifact_index(
                tmpdir / "ranking",
                tmpdir / "artifact_index.json",
                require_valid=True,
            )
            self.assertEqual(index["num_valid"], 2)

    def test_compute_subset_usage_from_ranking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path = tmpdir / "codes.npz"
            ranking_path = tmpdir / "task_feature_ranking.json"
            np.savez(
                codes_path,
                codes=np.asarray(
                    [
                        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                        [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                        [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
            )
            ranking = {
                "record_type": "task_feature_ranking",
                "task_id": "tiny_cls",
                "model_id": "dino_v2_base",
                "sae_id": "dino_l11_topk32_exp4",
                "ranking_method": "probe_weight",
                "rows": [
                    {
                        "feature_id": 0,
                        "task_rank": 1,
                        "ranking_score": 2.0,
                        "probe_weight_score": 2.0,
                        "validation_contribution_score": 0.0,
                        "mean_activation": 1.0,
                        "mean_positive_activation": 1.0,
                    },
                    {
                        "feature_id": 2,
                        "task_rank": 2,
                        "ranking_score": 1.0,
                        "probe_weight_score": 1.0,
                        "validation_contribution_score": 0.0,
                        "mean_activation": 0.5,
                        "mean_positive_activation": 1.0,
                    },
                ],
            }
            ranking_path.write_text(json.dumps(ranking), encoding="utf-8")
            subset_path = compute_subset_usage_from_ranking(
                codes_npz=codes_path,
                ranking_json=ranking_path,
                output_dir=tmpdir / "subset",
                top_k=2,
                random_seed=0,
                high_usage_threshold=3,
            )
            record = json.loads(subset_path.read_text())
            validate_subset_usage_summary(record)
            self.assertEqual(record["subsets"][0]["selection"], "task_selected")
            self.assertEqual(record["subsets"][1]["selection"], "matched_random")
            self.assertEqual(record["subsets"][0]["num_features"], 2)
            self.assertEqual(record["subsets"][1]["num_features"], 2)
            selected_ids = set(record["subsets"][0]["feature_ids"])
            random_ids = set(record["subsets"][1]["feature_ids"])
            self.assertFalse(selected_ids & random_ids)

    def test_compute_subset_usage_backfills_when_bucket_is_exhausted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path = tmpdir / "codes.npz"
            ranking_path = tmpdir / "task_feature_ranking.json"
            codes = np.ones((4, 10), dtype=np.float32)
            codes[:, 4:] = 0.0
            np.savez(codes_path, codes=codes)
            ranking = {
                "record_type": "task_feature_ranking",
                "task_id": "tiny_count",
                "model_id": "dino_v2_base",
                "sae_id": "dino_l11_topk32_exp4",
                "ranking_method": "probe_weight",
                "rows": [
                    {
                        "feature_id": feature_id,
                        "task_rank": rank,
                        "ranking_score": float(10 - rank),
                        "probe_weight_score": float(10 - rank),
                        "validation_contribution_score": 0.0,
                        "mean_activation": 0.0,
                        "mean_positive_activation": 0.0,
                    }
                    for rank, feature_id in enumerate([4, 5, 6, 7, 8], start=1)
                ],
            }
            ranking_path.write_text(json.dumps(ranking), encoding="utf-8")
            subset_path = compute_subset_usage_from_ranking(
                codes_npz=codes_path,
                ranking_json=ranking_path,
                output_dir=tmpdir / "subset",
                top_k=5,
                random_seed=0,
            )
            record = json.loads(subset_path.read_text())
            matched = record["subsets"][1]["feature_ids"]
            self.assertEqual(len(matched), 5)
            self.assertGreater(record["matching"]["diagnostics"]["backfill_count"], 0)
            self.assertFalse(set(matched) & {4, 5, 6, 7, 8})

    def test_rank_features_rejects_dim_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path = tmpdir / "codes.npz"
            logits_path = tmpdir / "probe_logits.npz"
            np.savez(codes_path, codes=np.ones((2, 3), dtype=np.float32))
            np.savez(logits_path, weights=np.ones((5, 2), dtype=np.float32))
            with self.assertRaises(ValueError):
                rank_features_from_linear_probe(
                    codes_npz=codes_path,
                    probe_logits_npz=logits_path,
                    task_id="tiny_cls",
                    model_id="dino_v2_base",
                    sae_id="dino_l11_topk32_exp4",
                    output_dir=tmpdir / "ranking",
                )

    def test_rank_features_supports_validation_contribution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path, logits_path = _write_tiny_ranking_inputs(tmpdir)
            contribution_path = tmpdir / "contribution.npz"
            np.savez(
                contribution_path,
                validation_contribution_score=np.asarray([0.1, 10.0, 0.2], dtype=np.float32),
            )

            ranking_path = rank_features_from_linear_probe(
                codes_npz=codes_path,
                probe_logits_npz=logits_path,
                task_id="tiny_cls",
                model_id="dino_v2_base",
                sae_id="dino_l11_topk32_exp4",
                output_dir=tmpdir / "ranking",
                top_k=2,
                ranking_method="validation_contribution",
                contribution_npz=contribution_path,
            )
            record = json.loads(ranking_path.read_text())
            validate_feature_ranking(record)
            self.assertEqual(record["ranking_method"], "validation_contribution")
            self.assertEqual(record["rows"][0]["feature_id"], 1)
            self.assertEqual(record["rows"][0]["validation_contribution_score"], 10.0)

    def test_rank_features_supports_regression_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path = tmpdir / "codes.npz"
            logits_path = tmpdir / "probe_logits.npz"
            np.savez(
                codes_path,
                codes=np.asarray(
                    [
                        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
                        [[0.0, 0.0, 3.0], [1.0, 0.0, 0.0]],
                    ],
                    dtype=np.float32,
                ),
            )
            np.savez(
                logits_path,
                weights=np.asarray([0.2, 5.0, 1.0, 0.0], dtype=np.float32),
            )
            ranking_path = rank_features_from_linear_probe(
                codes_npz=codes_path,
                probe_logits_npz=logits_path,
                task_id="tiny_depth",
                model_id="dino_v2_base",
                sae_id="dino_l11_topk32_exp4",
                output_dir=tmpdir / "ranking",
                top_k=2,
            )
            record = json.loads(ranking_path.read_text())
            validate_feature_ranking(record)
            self.assertEqual(record["rows"][0]["feature_id"], 1)

    def test_rank_features_supports_hybrid_ranking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path, logits_path = _write_tiny_ranking_inputs(tmpdir)
            contribution_path = tmpdir / "contribution.npz"
            np.savez(
                contribution_path,
                validation_contribution_score=np.asarray([0.1, 10.0, 0.2], dtype=np.float32),
            )

            ranking_path = rank_features_from_linear_probe(
                codes_npz=codes_path,
                probe_logits_npz=logits_path,
                task_id="tiny_cls",
                model_id="dino_v2_base",
                sae_id="dino_l11_topk32_exp4",
                output_dir=tmpdir / "ranking",
                top_k=2,
                ranking_method="hybrid",
                contribution_npz=contribution_path,
                hybrid_alpha=0.0,
            )
            record = json.loads(ranking_path.read_text())
            self.assertEqual(record["ranking_method"], "hybrid")
            self.assertEqual(record["rows"][0]["feature_id"], 1)

    def test_rank_features_requires_contribution_for_alternative_methods(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path, logits_path = _write_tiny_ranking_inputs(tmpdir)
            with self.assertRaises(ValueError):
                rank_features_from_linear_probe(
                    codes_npz=codes_path,
                    probe_logits_npz=logits_path,
                    task_id="tiny_cls",
                    model_id="dino_v2_base",
                    sae_id="dino_l11_topk32_exp4",
                    output_dir=tmpdir / "ranking",
                    ranking_method="hybrid",
                )


def _write_tiny_ranking_inputs(root: Path) -> tuple[Path, Path]:
    codes_path = root / "codes.npz"
    logits_path = root / "probe_logits.npz"
    np.savez(
        codes_path,
        codes=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    np.savez(
        logits_path,
        weights=np.asarray(
            [
                [5.0, 0.0],
                [0.1, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    return codes_path, logits_path


if __name__ == "__main__":
    unittest.main()
