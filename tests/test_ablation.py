import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_economy.analysis import ablate_linear_probe_features
from feature_economy.artifacts import build_artifact_index, validate_ablation_summary


class AblationTests(unittest.TestCase):
    def test_ablate_linear_probe_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path, logits_path, ranking_path = _write_tiny_ablation_inputs(tmpdir)
            summary_path = ablate_linear_probe_features(
                codes_npz=codes_path,
                probe_logits_npz=logits_path,
                ranking_json=ranking_path,
                task_type="classification",
                output_dir=tmpdir / "ablation",
                top_k=2,
                random_seed=0,
            )
            record = json.loads(summary_path.read_text())
            validate_ablation_summary(record)
            self.assertEqual(record["ablations"][0]["subset"], "top_2")
            self.assertGreater(record["ablations"][0]["metric_delta"]["top1"], 0.0)
            self.assertEqual(record["random_controls"][0]["num_features"], 2)
            index = build_artifact_index(
                tmpdir / "ablation",
                tmpdir / "artifact_index.json",
                require_valid=True,
            )
            self.assertEqual(index["num_valid"], 2)

    def test_ablate_linear_probe_features_with_pooled_token_codes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path = tmpdir / "codes.npz"
            logits_path = tmpdir / "probe_logits.npz"
            ranking_path = tmpdir / "task_feature_ranking.json"
            codes = np.asarray(
                [
                    [[1.0, 0.0], [1.0, 0.0]],
                    [[1.0, 0.0], [1.0, 0.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                    [[0.0, 1.0], [0.0, 1.0]],
                ],
                dtype=np.float32,
            )
            weights = np.asarray(
                [[2.0, -2.0], [-2.0, 2.0], [0.0, 0.0]],
                dtype=np.float32,
            )
            pooled = codes.mean(axis=1)
            logits = np.concatenate(
                [pooled, np.ones((pooled.shape[0], 1), dtype=np.float32)],
                axis=1,
            ) @ weights
            np.savez_compressed(codes_path, codes=codes)
            np.savez_compressed(
                logits_path,
                logits=logits,
                weights=weights,
                classes=np.asarray([0, 1], dtype=np.int64),
                labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
            )
            _write_ranking(ranking_path, task_id="tiny_cls", feature_ids=[0])
            summary_path = ablate_linear_probe_features(
                codes_npz=codes_path,
                probe_logits_npz=logits_path,
                ranking_json=ranking_path,
                task_type="classification",
                output_dir=tmpdir / "ablation",
                top_k=1,
                random_seed=0,
            )
            record = json.loads(summary_path.read_text())
            validate_ablation_summary(record)
            self.assertGreater(record["ablations"][0]["metric_delta"]["top1"], 0.0)

    def test_ablate_dense_depth_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path = tmpdir / "codes.npz"
            logits_path = tmpdir / "probe_logits.npz"
            ranking_path = tmpdir / "task_feature_ranking.json"
            targets = np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
            codes = np.concatenate(
                [
                    targets[..., None],
                    np.zeros((*targets.shape, 1), dtype=np.float32),
                    np.ones((*targets.shape, 1), dtype=np.float32),
                ],
                axis=-1,
            )
            weights = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            prediction = codes[..., 0]
            np.savez_compressed(codes_path, codes=codes)
            np.savez_compressed(
                logits_path,
                prediction=prediction,
                weights=weights,
                targets=targets,
            )
            _write_ranking(ranking_path, task_id="tiny_depth", feature_ids=[0])
            summary_path = ablate_linear_probe_features(
                codes_npz=codes_path,
                probe_logits_npz=logits_path,
                ranking_json=ranking_path,
                task_type="dense_depth",
                output_dir=tmpdir / "ablation",
                top_k=1,
                random_seed=0,
            )
            record = json.loads(summary_path.read_text())
            validate_ablation_summary(record)
            self.assertGreater(record["ablations"][0]["metric_delta"]["rmse"], 0.0)

    def test_ablate_dense_segmentation_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            codes_path = tmpdir / "codes.npz"
            logits_path = tmpdir / "probe_logits.npz"
            ranking_path = tmpdir / "task_feature_ranking.json"
            targets = np.asarray([[[0, 1], [0, 1]]], dtype=np.int64)
            codes = np.stack([1 - targets, targets, np.zeros_like(targets)], axis=-1).astype(
                np.float32
            )
            weights = np.asarray(
                [
                    [2.0, 0.0],
                    [0.0, 2.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                dtype=np.float32,
            )
            logits = np.concatenate(
                [codes.reshape(-1, 3), np.ones((4, 1), dtype=np.float32)],
                axis=1,
            ) @ weights
            np.savez_compressed(codes_path, codes=codes)
            np.savez_compressed(
                logits_path,
                logits=logits.reshape(1, 2, 2, 2),
                weights=weights,
                classes=np.asarray([0, 1], dtype=np.int64),
                targets=targets,
            )
            _write_ranking(ranking_path, task_id="tiny_seg", feature_ids=[1])
            summary_path = ablate_linear_probe_features(
                codes_npz=codes_path,
                probe_logits_npz=logits_path,
                ranking_json=ranking_path,
                task_type="dense_segmentation",
                output_dir=tmpdir / "ablation",
                top_k=1,
                random_seed=0,
            )
            record = json.loads(summary_path.read_text())
            validate_ablation_summary(record)
            self.assertGreater(record["ablations"][0]["metric_delta"]["pixel_accuracy"], 0.0)


def _write_tiny_ablation_inputs(root: Path) -> tuple[Path, Path, Path]:
    codes_path = root / "codes.npz"
    logits_path = root / "probe_logits.npz"
    ranking_path = root / "task_feature_ranking.json"
    codes = np.asarray(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    weights = np.asarray(
        [
            [2.0, 0.0],
            [0.0, 0.0],
            [0.0, 2.0],
            [0.1, 0.1],
            [0.0, 0.0],
            [0.1, 0.1],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    classes = np.asarray([0, 1], dtype=np.int64)
    logits = np.concatenate([codes, np.ones((codes.shape[0], 1), dtype=np.float32)], axis=1) @ weights
    np.savez_compressed(codes_path, codes=codes)
    np.savez_compressed(logits_path, logits=logits, weights=weights, classes=classes, labels=labels)
    _write_ranking(ranking_path, task_id="tiny_cls", feature_ids=[0, 2])
    return codes_path, logits_path, ranking_path


def _write_ranking(path: Path, *, task_id: str, feature_ids: list[int]) -> None:
    rows = []
    for rank, feature_id in enumerate(feature_ids, start=1):
        rows.append(
            {
                "feature_id": feature_id,
                "task_rank": rank,
                "ranking_score": 2.0,
                "probe_weight_score": 2.0,
                "validation_contribution_score": 0.0,
                "mean_activation": 0.5,
                "mean_positive_activation": 1.0,
            }
        )
    ranking = {
        "record_type": "task_feature_ranking",
        "task_id": task_id,
        "model_id": "dino_v2_base",
        "sae_id": "dino_l11_topk32_exp4",
        "ranking_method": "probe_weight",
        "rows": rows,
    }
    path.write_text(json.dumps(ranking, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
