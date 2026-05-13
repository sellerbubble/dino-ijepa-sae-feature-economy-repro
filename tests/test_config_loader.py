import unittest
from pathlib import Path

from feature_economy.configs import (
    ConfigError,
    load_yaml,
    validate_all_configs,
    validate_model_config,
    validate_probe_config,
    validate_task_config,
)


PUBLIC_REPRO_ROOT = Path(__file__).resolve().parents[1]


class ConfigLoaderTests(unittest.TestCase):
    def test_all_example_configs_are_valid(self):
        validated = validate_all_configs(PUBLIC_REPRO_ROOT / "configs")
        self.assertEqual(len(validated), 25)
        self.assertIn(
            PUBLIC_REPRO_ROOT / "configs" / "sweeps" / "ranking_control.yaml",
            validated,
        )
        self.assertIn(
            PUBLIC_REPRO_ROOT / "configs" / "sweeps" / "layer_sweep.yaml",
            validated,
        )

    def test_load_yaml_requires_mapping(self):
        fixture = PUBLIC_REPRO_ROOT / "tests" / "fixtures" / "tiny_probe_summary.json"
        record = load_yaml(fixture)
        self.assertEqual(record["record_type"], "sae_probe_summary")

    def test_model_config_rejects_private_paths(self):
        record = {
            "id": "bad_model",
            "display_name": "Bad Model",
            "provider": "local",
            "local_path": "/Users/liwenhao.109/private/model.pt",
            "architecture": "vit",
            "patch_size": 14,
            "default_layers": {"last": 11},
            "output": {"token_format": "patch_tokens"},
            "transform": {"mode": "shared_imagenet"},
        }
        with self.assertRaises(ConfigError):
            validate_model_config(record)

    def test_task_primary_metric_must_be_reported(self):
        record = {
            "id": "bad_task",
            "type": "classification",
            "train_manifest": "${DATA_ROOT}/train.jsonl",
            "val_manifest": "${DATA_ROOT}/val.jsonl",
            "metrics": {"primary": "top1", "report": ["top5"]},
            "readout": {"type": "linear_classifier", "epochs": 1, "batch_size": 2},
        }
        with self.assertRaises(ConfigError):
            validate_task_config(record)

    def test_probe_config_primary_metric_must_be_reported(self):
        record = {
            "id": "bad_probe",
            "task_id": "imagenet_1k",
            "probe_family": "classification",
            "supported_input_spaces": ["native", "sae_code"],
            "backend": "paper_scale_torch",
            "status": "draft_recipe",
            "training": {
                "epochs": 20,
                "batch_size": 512,
                "optimizer": "adamw",
                "learning_rate": 1e-3,
            },
            "selection": {"checkpoint_rule": "best_validation_top1"},
            "metrics": {"primary": "top1", "report": ["top5"]},
            "outputs": {
                "summary": "summary.json",
                "checkpoint": "probe.pt",
                "probe_outputs": "probe_outputs.npz",
            },
        }
        with self.assertRaises(ConfigError):
            validate_probe_config(record)


if __name__ == "__main__":
    unittest.main()
