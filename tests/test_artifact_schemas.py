import json
import unittest
from pathlib import Path

from feature_economy.artifacts import (
    validate_ablation_summary,
    validate_artifact_bundle_check,
    validate_availability_summary,
    validate_contribution_scores_summary,
    validate_dense_target_export_summary,
    validate_feature_extraction_summary,
    validate_feature_ranking,
    validate_probe_summary,
    validate_reproduction_run_plan,
    validate_run_manifest,
    validate_sae_code_summary,
    validate_subset_usage_summary,
)
from feature_economy.artifacts.schemas import SchemaError


FIXTURE_DIR = Path(__file__).parent / "fixtures"


class ArtifactSchemaTests(unittest.TestCase):
    def test_probe_summary_fixture_is_valid(self):
        record = json.loads((FIXTURE_DIR / "tiny_probe_summary.json").read_text())
        validate_probe_summary(record)

    def test_probe_summary_rejects_missing_sae_id(self):
        record = {
            "record_type": "sae_probe_summary",
            "task_id": "nyuv2_depth",
            "model_id": "dino_v2_base",
            "seed": 42,
            "metrics": {"rmse": 1.0},
        }
        with self.assertRaises(SchemaError):
            validate_probe_summary(record)

    def test_run_manifest_contract(self):
        validate_run_manifest(
            {
                "run_id": "tiny",
                "command": "feature-economy probe-sae --experiment tiny.yaml",
                "git_commit": "unknown",
                "config_files": ["tiny.yaml"],
                "inputs": {"dataset": "tiny"},
                "outputs": {"summary": "summary.json"},
            }
        )

    def test_reproduction_run_plan_contract(self):
        validate_reproduction_run_plan(
            {
                "record_type": "reproduction_run_plan",
                "num_rows": 1,
                "rows": [
                    {
                        "stage": "sae_probe",
                        "task_id": "imagenet_1k",
                        "model_id": "dino_v2_base",
                        "sae_id": "dino_l11_topk32_exp4",
                        "command": "probe-sae",
                        "artifact_dir": "${ARTIFACT_ROOT}/probes/sae/example",
                    }
                ],
            }
        )

    def test_artifact_bundle_check_contract(self):
        validate_artifact_bundle_check(
            {
                "record_type": "artifact_bundle_check",
                "run_plan": "run_plan.json",
                "artifact_root": "/tmp/artifacts",
                "num_rows": 1,
                "complete_rows": 0,
                "missing_rows": 1,
                "complete": False,
                "rows": [
                    {
                        "stage": "sae_probe",
                        "task_id": "imagenet_1k",
                        "model_id": "dino_v2_base",
                        "sae_id": "dino_l11_topk32_exp4",
                        "artifact_dir": "/tmp/artifacts/probes/sae/example",
                        "required_files": ["sae_probe_summary.json"],
                        "missing_files": ["sae_probe_summary.json"],
                        "complete": False,
                    }
                ],
            }
        )

    def test_feature_ranking_contract(self):
        validate_feature_ranking(
            {
                "record_type": "task_feature_ranking",
                "ranking_method": "hybrid",
                "rows": [
                    {
                        "feature_id": 1,
                        "task_rank": 1,
                        "ranking_score": 0.5,
                        "probe_weight_score": 0.4,
                        "validation_contribution_score": 0.6,
                        "mean_activation": 0.01,
                        "mean_positive_activation": 0.02,
                    }
                ],
            }
        )

    def test_contribution_scores_summary_contract(self):
        validate_contribution_scores_summary(
            {
                "record_type": "contribution_scores_summary",
                "task_id": "nyuv2_depth",
                "model_id": "dino_v2_base",
                "sae_id": "dino_l11_topk32_exp4",
                "task_type": "dense_depth",
                "primary_metric": "rmse",
                "score_key": "validation_contribution_score",
                "output_npz": "contribution_scores.npz",
                "num_features": 4096,
                "scored_features": 4096,
                "baseline_metrics": {"rmse": 0.9},
                "score_quantiles": {"median": 0.0, "max": 0.1},
                "top_features": [{"feature_id": 7, "score": 0.1}],
            }
        )

    def test_feature_extraction_summary_contract(self):
        validate_feature_extraction_summary(
            {
                "record_type": "feature_extraction_summary",
                "model_id": "dino_v2_base",
                "layer": 11,
                "input_manifest": "tiny.jsonl",
                "output_npz": "features.npz",
                "num_examples": 3,
                "feature_shape": [3, 4],
                "token_format": "fixture_global",
                "transform": {
                    "mode": "shared_imagenet",
                    "resize": 256,
                    "crop": 224,
                    "normalize": "imagenet",
                },
            }
        )

    def test_dense_target_export_summary_contract(self):
        validate_dense_target_export_summary(
            {
                "record_type": "dense_target_export_summary",
                "task_type": "dense_depth",
                "input_manifest": "nyuv2/val_manifest.jsonl",
                "output_npz": "targets.npz",
                "target_key": "targets",
                "num_examples": 8,
                "target_shape": [8, 16, 16],
                "target_dtype": "float32",
            }
        )

    def test_sae_code_summary_contract(self):
        validate_sae_code_summary(
            {
                "record_type": "sae_code_summary",
                "model_id": "dino_v2_base",
                "sae_id": "dino_l11_topk32_exp4",
                "input_features": "features.npz",
                "output_npz": "codes.npz",
                "num_examples": 3,
                "code_shape": [3, 16],
                "normalize_activations": "layer_norm",
            }
        )

    def test_ablation_summary_contract(self):
        validate_ablation_summary(
            {
                "record_type": "feature_ablation_summary",
                "task_id": "nyuv2_depth",
                "model_id": "ijepa_vit_h14",
                "sae_id": "ijepa_l31_topk32_exp4",
                "baseline_metrics": {"rmse": 1.0},
                "ablations": [
                    {
                        "subset": "frequent_or_above",
                        "num_features": 9,
                        "metrics": {"rmse": 1.1},
                        "metric_delta": {"rmse": 0.1},
                    }
                ],
                "random_controls": [],
            }
        )

    def test_availability_summary_contract(self):
        validate_availability_summary(
            {
                "record_type": "availability_summary",
                "model_id": "dino_v2_base",
                "sae_id": "dino_l11_topk32_exp4",
                "dataset_id": "imagenet_1k",
                "split": "val",
                "usage_unit": "fired_count",
                "total_features": 4096,
                "fired_features": 4000,
                "dead_features": 96,
                "buckets": {
                    "dead": 96,
                    "rare": 512,
                    "medium": 1024,
                    "frequent_or_above": 2464,
                },
                "active_fired_count_quantiles": {
                    "p10": 12,
                    "p50": 100,
                    "p90": 1000,
                },
            }
        )

    def test_subset_usage_summary_contract(self):
        validate_subset_usage_summary(
            {
                "record_type": "subset_usage_summary",
                "task_id": "nyuv2_depth",
                "model_id": "ijepa_vit_h14",
                "sae_id": "ijepa_l31_topk32_exp4",
                "ranking_method": "hybrid",
                "usage_unit": "fired_count",
                "subsets": [
                    {
                        "name": "top_100",
                        "selection": "task_selected",
                        "num_features": 100,
                        "median_usage": 12.5,
                        "high_usage_fraction": 0.09,
                    },
                    {
                        "name": "random_100_seed0",
                        "selection": "matched_random",
                        "num_features": 100,
                        "median_usage": 1.0,
                        "high_usage_fraction": 0.01,
                    },
                ],
            }
        )


if __name__ == "__main__":
    unittest.main()
