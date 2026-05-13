"""SAE-code extraction.

This module preserves the public artifact contract for feature -> SAE-code
conversion. It supports deterministic fixture conversion and a lightweight
inference-only linear TopK SAE checkpoint format.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from feature_economy.artifacts import (
    current_git_commit,
    validate_run_manifest,
    validate_sae_code_summary,
)

from .registry import ModelRegistry


def extract_fixture_sae_codes(
    *,
    config_root: str | Path,
    features_npz: str | Path,
    model_id: str,
    sae_id: str,
    output_dir: str | Path,
    code_dim: int = 16,
) -> Path:
    """Write deterministic fixture SAE codes from feature arrays."""

    registry = ModelRegistry(config_root)
    registry.describe_model_sae_pair(model_id, sae_id)
    sae = registry.get_sae(sae_id)
    features_npz = Path(features_npz)
    arrays = np.load(features_npz)
    features = np.asarray(arrays["features"], dtype=np.float32)
    codes = _fixture_codes(features, code_dim)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_path = output_dir / "codes.npz"
    np.savez_compressed(
        code_path,
        codes=codes.astype("float32"),
        source_features=str(features_npz),
    )
    summary = {
        "record_type": "sae_code_summary",
        "model_id": model_id,
        "sae_id": sae_id,
        "input_features": str(features_npz),
        "output_npz": str(code_path),
        "num_examples": int(codes.shape[0]),
        "code_shape": list(codes.shape),
        "normalize_activations": sae["normalize_activations"],
        "fixture": True,
    }
    validate_sae_code_summary(summary)
    summary_path = output_dir / "sae_code_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "run_id": f"fixture_codes_{model_id}_{sae_id}",
        "command": "feature-economy extract-sae-codes",
        "git_commit": current_git_commit(),
        "config_files": [str(path) for path in sorted(Path(config_root).rglob("*.yaml"))],
        "inputs": {
            "features_npz": str(features_npz),
            "model_id": model_id,
            "sae_id": sae_id,
        },
        "outputs": {
            "codes": str(code_path),
            "summary": str(summary_path),
        },
        "fixture": True,
    }
    validate_run_manifest(manifest)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def extract_linear_topk_sae_codes(
    *,
    config_root: str | Path,
    features_npz: str | Path,
    model_id: str,
    sae_id: str,
    output_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    env: dict[str, str] | None = None,
    require_resolved_checkpoint: bool = True,
    topk: int | None = None,
) -> Path:
    """Encode features with a lightweight inference-only linear TopK SAE."""

    registry = ModelRegistry(config_root)
    registry.describe_model_sae_pair(model_id, sae_id)
    sae = registry.get_sae(sae_id)
    resolved_checkpoint = (
        str(checkpoint_path)
        if checkpoint_path is not None
        else registry.sae_checkpoint_path(
            sae_id,
            env=env,
            require_resolved=require_resolved_checkpoint,
        )
    )
    checkpoint = load_lightweight_sae_checkpoint(resolved_checkpoint)
    features_npz = Path(features_npz)
    arrays = np.load(features_npz)
    features = np.asarray(arrays["features"], dtype=np.float32)
    codes = encode_linear_topk(
        features,
        checkpoint=checkpoint,
        normalize_activations=str(sae["normalize_activations"]),
        topk=topk if topk is not None else int(sae["topk"]),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    code_path = output_dir / "codes.npz"
    np.savez_compressed(
        code_path,
        codes=codes.astype("float32"),
        source_features=str(features_npz),
    )
    return _write_sae_code_artifacts(
        config_root=config_root,
        features_npz=features_npz,
        model_id=model_id,
        sae_id=sae_id,
        output_dir=output_dir,
        code_path=code_path,
        code_shape=list(codes.shape),
        normalize_activations=str(sae["normalize_activations"]),
        fixture=False,
        backend="linear_topk",
        extra_inputs={
            "checkpoint": resolved_checkpoint,
            "topk": topk if topk is not None else int(sae["topk"]),
        },
    )


def convert_sae_checkpoint_to_lightweight(
    *,
    input_checkpoint: str | Path,
    output_checkpoint: str | Path,
    allow_gated: bool = False,
) -> Path:
    """Convert a common full SAE checkpoint into the public lightweight format."""

    raw = _load_raw_checkpoint(input_checkpoint)
    state = _extract_state_dict(raw)
    lightweight = _extract_lightweight_arrays(state, allow_gated=allow_gated)
    output_checkpoint = Path(output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_checkpoint, **lightweight)
    # Re-load to validate the exact public artifact written to disk.
    load_lightweight_sae_checkpoint(output_checkpoint)
    return output_checkpoint


def _fixture_codes(features: np.ndarray, code_dim: int) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("features must have shape [num_examples, feature_dim]")
    if code_dim <= 0:
        raise ValueError("code_dim must be positive")
    weights = np.arange(1, features.shape[1] * code_dim + 1, dtype=np.float32)
    weights = weights.reshape(features.shape[1], code_dim)
    projected = features @ ((weights % 17) / 17.0)
    threshold = np.percentile(projected, 50, axis=1, keepdims=True)
    return np.where(projected >= threshold, projected, 0.0)


def _load_raw_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SAE checkpoint does not exist: {path}")
    if path.suffix == ".npz":
        arrays = np.load(path)
        return {key: np.asarray(arrays[key]) for key in arrays.files}
    if path.suffix in {".pt", ".pth"}:
        return _load_torch_checkpoint(path)
    raise ValueError(f"unsupported SAE checkpoint format: {path.suffix}")


def _extract_state_dict(raw: dict[str, Any]) -> dict[str, Any]:
    if "state_dict" in raw and isinstance(raw["state_dict"], dict):
        return raw["state_dict"]
    if "autoencoder" in raw and isinstance(raw["autoencoder"], dict):
        autoencoder = raw["autoencoder"]
        if "state_dict" in autoencoder and isinstance(autoencoder["state_dict"], dict):
            return autoencoder["state_dict"]
        return autoencoder
    if "autoencoder" in raw and hasattr(raw["autoencoder"], "state_dict"):
        return _torch_to_numpy_mapping(raw["autoencoder"].state_dict())
    if "autoencoder" in raw and hasattr(raw["autoencoder"], "__dict__"):
        return _torch_to_numpy_mapping(vars(raw["autoencoder"]))
    if hasattr(raw, "state_dict"):
        return _torch_to_numpy_mapping(raw.state_dict())
    if hasattr(raw, "__dict__"):
        return _torch_to_numpy_mapping(vars(raw))
    return raw


def _extract_lightweight_arrays(
    state: dict[str, Any],
    *,
    allow_gated: bool = False,
) -> dict[str, np.ndarray]:
    if _contains_any(state, ["b_gate", "b_mag", "r_mag"]) and not allow_gated:
        raise ValueError(
            "gated SAE checkpoints need an explicit gated public backend; "
            "linear-topk conversion would not preserve the gating mechanism"
        )
    encoder_weight = _get_first_array(
        state,
        ["encoder_weight", "W_enc", "encoder.weight", "encoder.weight_orig"],
        required=True,
    )
    encoder_bias = _get_first_array(
        state,
        ["encoder_bias", "b_enc", "encoder.bias"],
        required=False,
    )
    decoder_bias = _get_first_array(
        state,
        ["decoder_bias", "b_dec", "decoder.bias"],
        required=False,
    )
    decoder_weight = _get_first_array(
        state,
        ["decoder_weight", "W_dec", "decoder.weight", "decoder.weight_orig"],
        required=False,
    )
    if encoder_weight is None:
        raise AssertionError("unreachable: required encoder_weight missing")
    encoder_weight = np.asarray(encoder_weight, dtype=np.float32)
    if encoder_weight.ndim != 2:
        raise ValueError("extracted encoder_weight must have shape [input_dim, code_dim]")
    input_dim, code_dim = encoder_weight.shape
    if encoder_bias is None:
        encoder_bias = np.zeros(code_dim, dtype=np.float32)
    if decoder_bias is None:
        decoder_bias = np.zeros(input_dim, dtype=np.float32)
    arrays = {
        "encoder_weight": encoder_weight,
        "encoder_bias": np.asarray(encoder_bias, dtype=np.float32),
        "decoder_bias": np.asarray(decoder_bias, dtype=np.float32),
    }
    if decoder_weight is not None:
        decoder_weight = np.asarray(decoder_weight, dtype=np.float32)
        if decoder_weight.shape == (input_dim, code_dim):
            decoder_weight = decoder_weight.T
        if decoder_weight.shape != (code_dim, input_dim):
            raise ValueError("extracted decoder_weight must have shape [code_dim, input_dim]")
        arrays["decoder_weight"] = decoder_weight
    return arrays


def load_lightweight_sae_checkpoint(path: str | Path) -> dict[str, np.ndarray]:
    """Load a minimal SAE checkpoint.

    Supported keys:

    - `encoder_weight`: shape `[input_dim, code_dim]`
    - `encoder_bias`: shape `[code_dim]`, optional
    - `decoder_bias`: shape `[input_dim]`, optional

    The `.npz` format is the public, dependency-light path. `.pt`/`.pth` files
    are accepted when torch is installed and contain the same key names.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SAE checkpoint does not exist: {path}")
    if path.suffix == ".npz":
        arrays = np.load(path)
        checkpoint = {key: np.asarray(arrays[key]) for key in arrays.files}
    elif path.suffix in {".pt", ".pth"}:
        checkpoint = _load_torch_checkpoint(path)
    else:
        raise ValueError(f"unsupported SAE checkpoint format: {path.suffix}")
    if "encoder_weight" not in checkpoint:
        raise ValueError("SAE checkpoint requires encoder_weight")
    encoder_weight = np.asarray(checkpoint["encoder_weight"], dtype=np.float32)
    if encoder_weight.ndim != 2:
        raise ValueError("encoder_weight must have shape [input_dim, code_dim]")
    input_dim, code_dim = encoder_weight.shape
    encoder_bias = np.asarray(
        checkpoint.get("encoder_bias", np.zeros(code_dim, dtype=np.float32)),
        dtype=np.float32,
    )
    decoder_bias = np.asarray(
        checkpoint.get("decoder_bias", np.zeros(input_dim, dtype=np.float32)),
        dtype=np.float32,
    )
    if encoder_bias.shape != (code_dim,):
        raise ValueError("encoder_bias must have shape [code_dim]")
    if decoder_bias.shape != (input_dim,):
        raise ValueError("decoder_bias must have shape [input_dim]")
    out = {
        "encoder_weight": encoder_weight,
        "encoder_bias": encoder_bias,
        "decoder_bias": decoder_bias,
    }
    if "decoder_weight" in checkpoint:
        decoder_weight = np.asarray(checkpoint["decoder_weight"], dtype=np.float32)
        if decoder_weight.shape == (input_dim, code_dim):
            decoder_weight = decoder_weight.T
        if decoder_weight.shape != (code_dim, input_dim):
            raise ValueError("decoder_weight must have shape [code_dim, input_dim]")
        out["decoder_weight"] = decoder_weight
    return out


def encode_linear_topk(
    features: np.ndarray,
    *,
    checkpoint: dict[str, np.ndarray],
    normalize_activations: str,
    topk: int,
) -> np.ndarray:
    """Apply runtime normalization, linear encoder, ReLU, and TopK sparsity."""

    if features.ndim < 2:
        raise ValueError("features must have shape [..., feature_dim]")
    if topk <= 0:
        raise ValueError("topk must be positive")
    encoder_weight = np.asarray(checkpoint["encoder_weight"], dtype=np.float32)
    encoder_bias = np.asarray(checkpoint["encoder_bias"], dtype=np.float32)
    decoder_bias = np.asarray(checkpoint["decoder_bias"], dtype=np.float32)
    if features.shape[-1] != encoder_weight.shape[0]:
        raise ValueError(
            f"feature dim {features.shape[-1]} does not match SAE input dim {encoder_weight.shape[0]}"
        )
    normalized = _normalize_features(np.asarray(features, dtype=np.float32), normalize_activations)
    centered = normalized - decoder_bias
    codes = np.maximum(centered @ encoder_weight + encoder_bias, 0.0)
    return _apply_topk(codes, min(topk, codes.shape[-1])).astype(np.float32)


def _normalize_features(features: np.ndarray, normalize_activations: str) -> np.ndarray:
    if normalize_activations == "none":
        return features
    if normalize_activations == "layer_norm":
        mean = np.mean(features, axis=-1, keepdims=True)
        std = np.std(features, axis=-1, keepdims=True)
        return (features - mean) / np.maximum(std, 1e-6)
    raise ValueError(f"unsupported normalize_activations: {normalize_activations}")


def _apply_topk(codes: np.ndarray, topk: int) -> np.ndarray:
    if topk >= codes.shape[-1]:
        return codes
    indices = np.argpartition(codes, -topk, axis=-1)[..., -topk:]
    mask = np.zeros_like(codes, dtype=bool)
    np.put_along_axis(mask, indices, True, axis=-1)
    return np.where(mask, codes, 0.0)


def _write_sae_code_artifacts(
    *,
    config_root: str | Path,
    features_npz: Path,
    model_id: str,
    sae_id: str,
    output_dir: Path,
    code_path: Path,
    code_shape: list[int],
    normalize_activations: str,
    fixture: bool,
    backend: str,
    extra_inputs: dict[str, Any] | None = None,
) -> Path:
    summary = {
        "record_type": "sae_code_summary",
        "model_id": model_id,
        "sae_id": sae_id,
        "input_features": str(features_npz),
        "output_npz": str(code_path),
        "num_examples": int(code_shape[0]),
        "code_shape": code_shape,
        "normalize_activations": normalize_activations,
        "fixture": fixture,
        "backend": backend,
    }
    validate_sae_code_summary(summary)
    summary_path = output_dir / "sae_code_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    inputs = {
        "features_npz": str(features_npz),
        "model_id": model_id,
        "sae_id": sae_id,
        "backend": backend,
    }
    if extra_inputs:
        inputs.update(extra_inputs)
    manifest = {
        "run_id": f"{backend}_codes_{model_id}_{sae_id}",
        "command": "feature-economy extract-sae-codes",
        "git_commit": current_git_commit(),
        "config_files": [str(path) for path in sorted(Path(config_root).rglob("*.yaml"))],
        "inputs": inputs,
        "outputs": {
            "codes": str(code_path),
            "summary": str(summary_path),
        },
        "fixture": fixture,
    }
    validate_run_manifest(manifest)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def _load_torch_checkpoint(path: Path) -> dict[str, np.ndarray]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("loading .pt SAE checkpoints requires torch") from exc
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - for older torch versions
        loaded = torch.load(path, map_location="cpu")
    except ModuleNotFoundError as exc:
        if "vit_prisma" not in str(exc):
            raise
        with _vit_prisma_pickle_stubs():
            loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        if hasattr(loaded, "state_dict"):
            loaded = loaded.state_dict()
        elif hasattr(loaded, "__dict__"):
            loaded = vars(loaded)
        else:
            raise ValueError(".pt SAE checkpoint must contain a mapping")
    return _torch_to_numpy_mapping(loaded)


def _torch_to_numpy_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _torch_to_numpy_mapping(nested) for key, nested in value.items()}
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "state_dict"):
        return _torch_to_numpy_mapping(value.state_dict())
    if hasattr(value, "__dict__") and value.__class__.__module__.startswith("vit_prisma"):
        return _torch_to_numpy_mapping(vars(value))
    return value


@contextmanager
def _vit_prisma_pickle_stubs():
    """Temporarily install lightweight modules for legacy vit_prisma pickles.

    Some private SAE checkpoints pickle a `vit_prisma` autoencoder object even
    though the tensors needed for public inference are just attributes or a
    state dict. Public reproduction should not require installing the full
    training package tree merely to read those tensors.
    """

    finder = _VitPrismaStubFinder()
    previous_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "vit_prisma" or name.startswith("vit_prisma.")
    }
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:  # pragma: no cover - defensive cleanup
            pass
        for name in [
            name
            for name in list(sys.modules)
            if name == "vit_prisma" or name.startswith("vit_prisma.")
        ]:
            if name in previous_modules:
                sys.modules[name] = previous_modules[name]
            else:
                sys.modules.pop(name, None)


class _VitPrismaStubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None):
        if fullname == "vit_prisma" or fullname.startswith("vit_prisma."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []

        def __getattr__(name: str):
            return _make_pickle_stub_class(spec.name, name)

        module.__getattr__ = __getattr__  # type: ignore[attr-defined]
        return module

    def exec_module(self, module):
        return None


def _make_pickle_stub_class(module_name: str, class_name: str):
    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_pickle_state"] = state

    return type(
        class_name,
        (),
        {
            "__module__": module_name,
            "__setstate__": __setstate__,
        },
    )


def _contains_any(mapping: dict[str, Any], keys: list[str]) -> bool:
    return any(key in mapping for key in keys)


def _get_first_array(
    mapping: dict[str, Any],
    keys: list[str],
    *,
    required: bool,
) -> np.ndarray | None:
    for key in keys:
        if key in mapping:
            return np.asarray(mapping[key])
    if required:
        raise ValueError(f"checkpoint is missing required keys; tried {keys}")
    return None
