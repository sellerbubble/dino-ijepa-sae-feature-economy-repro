import tempfile
import unittest
from pathlib import Path

from feature_economy.models import (
    ModelRegistry,
    RegistryError,
    resolve_placeholders,
    transform_policy_from_config,
)


PUBLIC_REPRO_ROOT = Path(__file__).resolve().parents[1]


class ModelRegistryTests(unittest.TestCase):
    def test_registry_loads_models_and_saes(self):
        registry = ModelRegistry(PUBLIC_REPRO_ROOT / "configs")
        self.assertEqual(set(registry.models), {"dino_v2_base", "ijepa_vit_h14"})
        self.assertEqual(
            set(registry.saes),
            {"dino_l11_topk32_exp4", "ijepa_l31_topk32_exp4"},
        )
        pair = registry.describe_model_sae_pair("dino_v2_base", "dino_l11_topk32_exp4")
        self.assertEqual(pair["layer"], 11)
        self.assertEqual(pair["normalize_activations"], "layer_norm")
        self.assertEqual(pair["transform"].mode, "shared_imagenet")
        self.assertEqual(pair["transform"].crop, 224)

    def test_checkpoint_placeholder_resolution(self):
        registry = ModelRegistry(PUBLIC_REPRO_ROOT / "configs")
        unresolved = registry.sae_checkpoint_path("dino_l11_topk32_exp4", env={})
        self.assertIn("${SAE_ROOT}", unresolved)
        resolved = registry.sae_checkpoint_path(
            "dino_l11_topk32_exp4",
            env={"SAE_ROOT": "/tmp/saes"},
            require_resolved=True,
        )
        self.assertEqual(resolved, "/tmp/saes/dino_l11_topk32_exp4/final_sae.pt")

    def test_missing_placeholder_can_be_required(self):
        with self.assertRaises(RegistryError):
            resolve_placeholders("${SAE_ROOT}/x.pt", env={}, require_resolved=True)

    def test_registry_rejects_sae_with_unknown_model(self):
        source = PUBLIC_REPRO_ROOT / "configs"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for family in ["models", "saes", "tasks", "experiments"]:
                (tmpdir / family).mkdir(parents=True)
                for path in (source / family).glob("*.yaml"):
                    (tmpdir / family / path.name).write_text(path.read_text(), encoding="utf-8")
            bad_sae = tmpdir / "saes" / "bad.yaml"
            bad_sae.write_text(
                "\n".join(
                    [
                        "id: bad_sae",
                        "model_id: missing_model",
                        "layer: 11",
                        "checkpoint: ${SAE_ROOT}/bad.pt",
                        "topk: 32",
                        "expansion: 4",
                        "normalize_activations: layer_norm",
                        "trained_on: imagenet_1k_train",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(RegistryError):
                ModelRegistry(tmpdir)

    def test_transform_policy_from_config(self):
        policy = transform_policy_from_config(
            {"mode": "shared_imagenet", "resize": 256, "crop": 224, "normalize": "imagenet"}
        )
        self.assertEqual(policy.mean, (0.485, 0.456, 0.406))
        self.assertEqual(policy.std, (0.229, 0.224, 0.225))

    def test_transform_policy_rejects_crop_larger_than_resize(self):
        with self.assertRaises(ValueError):
            transform_policy_from_config(
                {
                    "mode": "shared_imagenet",
                    "resize": 224,
                    "crop": 256,
                    "normalize": "imagenet",
                }
            )


if __name__ == "__main__":
    unittest.main()
