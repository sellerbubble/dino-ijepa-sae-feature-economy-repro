import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_economy.artifacts import build_artifact_index, validate_feature_extraction_summary
from feature_economy.models import (
    extract_fixture_features,
    extract_huggingface_features,
    extract_torchscript_features,
)
from feature_economy.models.feature_extraction import _select_hidden_state


PUBLIC_REPRO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _optional_torch_and_pil():
    try:
        import torch
        from PIL import Image
    except Exception:
        return None, None
    return torch, Image


class FeatureExtractionTests(unittest.TestCase):
    def test_extract_fixture_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = extract_fixture_features(
                config_root=PUBLIC_REPRO_ROOT / "configs",
                manifest_path=FIXTURE_DIR / "tiny_imagenet_eval_manifest.jsonl",
                task_type="classification",
                model_id="dino_v2_base",
                output_dir=tmpdir,
                expected_split="val",
                feature_dim=6,
            )
            record = json.loads(summary_path.read_text())
            validate_feature_extraction_summary(record)
            self.assertEqual(record["feature_shape"], [3, 6])
            arrays = np.load(record["output_npz"])
            self.assertEqual(arrays["features"].shape, (3, 6))

    def test_feature_extraction_summary_is_indexed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            artifact_dir = tmpdir / "features"
            extract_fixture_features(
                config_root=PUBLIC_REPRO_ROOT / "configs",
                manifest_path=FIXTURE_DIR / "tiny_imagenet_eval_manifest.jsonl",
                task_type="classification",
                model_id="dino_v2_base",
                output_dir=artifact_dir,
                expected_split="val",
            )
            index = build_artifact_index(
                artifact_dir,
                tmpdir / "artifact_index.json",
                require_valid=True,
            )
            self.assertEqual(index["num_valid"], 2)
            self.assertIn("feature_extraction_summary", {r["record_type"] for r in index["records"]})

    def test_huggingface_backend_rejects_non_hf_model_before_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(NotImplementedError):
                extract_huggingface_features(
                    config_root=PUBLIC_REPRO_ROOT / "configs",
                    manifest_path=FIXTURE_DIR / "tiny_imagenet_eval_manifest.jsonl",
                    task_type="classification",
                    model_id="ijepa_vit_h14",
                    output_dir=tmpdir,
                    expected_split="val",
                    max_examples=1,
                )

    def test_torchscript_backend_extracts_real_image_features(self):
        torch, image_class = _optional_torch_and_pil()
        if torch is None or image_class is None:
            self.skipTest("torch and Pillow are required for TorchScript extraction")

        class TinyFeatureModule(torch.nn.Module):
            def forward(self, pixel_values):
                pooled = pixel_values.mean(dim=(2, 3))
                return pooled[:, :2]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            image_path = tmpdir / "tiny.png"
            image_class.new("RGB", (260, 260), color=(128, 64, 32)).save(image_path)
            manifest_path = tmpdir / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps({"image": "tiny.png", "split": "val", "label": 0}) + "\n",
                encoding="utf-8",
            )
            checkpoint_path = tmpdir / "tiny_feature_module.pt"
            traced = torch.jit.trace(
                TinyFeatureModule(),
                torch.zeros((1, 3, 224, 224), dtype=torch.float32),
            )
            traced.save(str(checkpoint_path))

            summary_path = extract_torchscript_features(
                config_root=PUBLIC_REPRO_ROOT / "configs",
                manifest_path=manifest_path,
                task_type="classification",
                model_id="ijepa_vit_h14",
                output_dir=tmpdir / "features",
                checkpoint_path=checkpoint_path,
                expected_split="val",
                batch_size=1,
            )
            record = json.loads(summary_path.read_text())
            validate_feature_extraction_summary(record)
            self.assertEqual(record["backend"], "torchscript")
            self.assertEqual(record["model_id"], "ijepa_vit_h14")
            self.assertEqual(record["feature_shape"], [1, 2])
            arrays = np.load(record["output_npz"])
            self.assertEqual(arrays["features"].shape, (1, 2))

    def test_torchscript_backend_rejects_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                extract_torchscript_features(
                    config_root=PUBLIC_REPRO_ROOT / "configs",
                    manifest_path=FIXTURE_DIR / "tiny_imagenet_eval_manifest.jsonl",
                    task_type="classification",
                    model_id="ijepa_vit_h14",
                    output_dir=tmpdir,
                    checkpoint_path=Path(tmpdir) / "missing.pt",
                    expected_split="val",
                    max_examples=1,
                )

    def test_select_hidden_state_maps_encoder_layer_to_hf_index(self):
        hidden_states = [np.asarray([index]) for index in range(4)]
        selected = _select_hidden_state(hidden_states, layer=2)
        self.assertEqual(selected.tolist(), [3])


if __name__ == "__main__":
    unittest.main()
