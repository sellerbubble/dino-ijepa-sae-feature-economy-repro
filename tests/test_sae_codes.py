import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from feature_economy.artifacts import build_artifact_index, validate_sae_code_summary
from feature_economy.models import (
    convert_sae_checkpoint_to_lightweight,
    extract_fixture_features,
    extract_fixture_sae_codes,
    extract_linear_topk_sae_codes,
)
from feature_economy.models.sae_codes import encode_linear_topk, load_lightweight_sae_checkpoint


PUBLIC_REPRO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = Path(__file__).parent / "fixtures"


class SaeCodeTests(unittest.TestCase):
    def test_extract_fixture_sae_codes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            feature_summary = extract_fixture_features(
                config_root=PUBLIC_REPRO_ROOT / "configs",
                manifest_path=FIXTURE_DIR / "tiny_imagenet_eval_manifest.jsonl",
                task_type="classification",
                model_id="dino_v2_base",
                output_dir=tmpdir / "features",
                expected_split="val",
                feature_dim=5,
            )
            features_npz = json.loads(feature_summary.read_text())["output_npz"]
            code_summary = extract_fixture_sae_codes(
                config_root=PUBLIC_REPRO_ROOT / "configs",
                features_npz=features_npz,
                model_id="dino_v2_base",
                sae_id="dino_l11_topk32_exp4",
                output_dir=tmpdir / "codes",
                code_dim=7,
            )
            record = json.loads(code_summary.read_text())
            validate_sae_code_summary(record)
            self.assertEqual(record["code_shape"], [3, 7])
            arrays = np.load(record["output_npz"])
            self.assertEqual(arrays["codes"].shape, (3, 7))

    def test_sae_code_summary_is_indexed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            feature_summary = extract_fixture_features(
                config_root=PUBLIC_REPRO_ROOT / "configs",
                manifest_path=FIXTURE_DIR / "tiny_imagenet_eval_manifest.jsonl",
                task_type="classification",
                model_id="dino_v2_base",
                output_dir=tmpdir / "features",
                expected_split="val",
            )
            extract_fixture_sae_codes(
                config_root=PUBLIC_REPRO_ROOT / "configs",
                features_npz=json.loads(feature_summary.read_text())["output_npz"],
                model_id="dino_v2_base",
                sae_id="dino_l11_topk32_exp4",
                output_dir=tmpdir / "codes",
            )
            index = build_artifact_index(
                tmpdir,
                tmpdir / "artifact_index.json",
                require_valid=True,
            )
            self.assertIn("sae_code_summary", {r["record_type"] for r in index["records"]})

    def test_extract_linear_topk_sae_codes_from_npz_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            feature_summary = extract_fixture_features(
                config_root=PUBLIC_REPRO_ROOT / "configs",
                manifest_path=FIXTURE_DIR / "tiny_imagenet_eval_manifest.jsonl",
                task_type="classification",
                model_id="dino_v2_base",
                output_dir=tmpdir / "features",
                expected_split="val",
                feature_dim=4,
            )
            features_npz = json.loads(feature_summary.read_text())["output_npz"]
            checkpoint = tmpdir / "sae.npz"
            np.savez(
                checkpoint,
                encoder_weight=np.eye(4, 6, dtype=np.float32),
                encoder_bias=np.zeros(6, dtype=np.float32),
                decoder_bias=np.zeros(4, dtype=np.float32),
            )
            code_summary = extract_linear_topk_sae_codes(
                config_root=PUBLIC_REPRO_ROOT / "configs",
                features_npz=features_npz,
                model_id="dino_v2_base",
                sae_id="dino_l11_topk32_exp4",
                output_dir=tmpdir / "codes",
                checkpoint_path=checkpoint,
                topk=2,
            )
            record = json.loads(code_summary.read_text())
            validate_sae_code_summary(record)
            self.assertEqual(record["code_shape"], [3, 6])
            arrays = np.load(record["output_npz"])
            nonzero = np.count_nonzero(arrays["codes"], axis=-1)
            self.assertTrue(np.all(nonzero <= 2))

    def test_lightweight_checkpoint_requires_encoder_weight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "bad.npz"
            np.savez(checkpoint, encoder_bias=np.zeros(2, dtype=np.float32))
            with self.assertRaises(ValueError):
                load_lightweight_sae_checkpoint(checkpoint)

    def test_encode_linear_topk_rejects_dim_mismatch(self):
        features = np.ones((2, 3), dtype=np.float32)
        checkpoint = {
            "encoder_weight": np.ones((4, 5), dtype=np.float32),
            "encoder_bias": np.zeros(5, dtype=np.float32),
            "decoder_bias": np.zeros(4, dtype=np.float32),
        }
        with self.assertRaises(ValueError):
            encode_linear_topk(
                features,
                checkpoint=checkpoint,
                normalize_activations="none",
                topk=2,
            )

    def test_convert_plain_state_dict_to_lightweight_npz(self):
        try:
            import torch
        except Exception:  # pragma: no cover - depends on optional dependency
            self.skipTest("torch is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_checkpoint = tmpdir / "full_sae.pt"
            output_checkpoint = tmpdir / "lightweight_sae.npz"
            torch.save(
                {
                    "state_dict": {
                        "W_enc": torch.ones(4, 6),
                        "b_enc": torch.arange(6, dtype=torch.float32),
                        "b_dec": torch.arange(4, dtype=torch.float32),
                    }
                },
                input_checkpoint,
            )
            converted = convert_sae_checkpoint_to_lightweight(
                input_checkpoint=input_checkpoint,
                output_checkpoint=output_checkpoint,
            )
            checkpoint = load_lightweight_sae_checkpoint(converted)
        self.assertEqual(checkpoint["encoder_weight"].shape, (4, 6))
        self.assertEqual(checkpoint["encoder_bias"].tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(checkpoint["decoder_bias"].tolist(), [0, 1, 2, 3])

    def test_convert_rejects_gated_checkpoint_by_default(self):
        try:
            import torch
        except Exception:  # pragma: no cover - depends on optional dependency
            self.skipTest("torch is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_checkpoint = tmpdir / "gated_sae.pt"
            output_checkpoint = tmpdir / "lightweight_sae.npz"
            torch.save(
                {
                    "state_dict": {
                        "W_enc": torch.ones(4, 6),
                        "b_gate": torch.zeros(6),
                        "b_mag": torch.zeros(6),
                        "r_mag": torch.zeros(6),
                    }
                },
                input_checkpoint,
            )
            with self.assertRaises(ValueError):
                convert_sae_checkpoint_to_lightweight(
                    input_checkpoint=input_checkpoint,
                    output_checkpoint=output_checkpoint,
                )


if __name__ == "__main__":
    unittest.main()
