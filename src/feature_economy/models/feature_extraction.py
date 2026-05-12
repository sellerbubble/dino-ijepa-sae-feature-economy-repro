"""Feature extraction.

This module preserves one output contract for both fixture and real backbone
feature extraction: a compressed `features.npz`, a
`feature_extraction_summary.json`, and a `run_manifest.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from feature_economy.artifacts import (
    current_git_commit,
    validate_feature_extraction_summary,
    validate_run_manifest,
)
from feature_economy.data import ManifestDataset

from .registry import ModelRegistry
from .transforms import preprocess_image_file, transform_policy_from_config


def extract_fixture_features(
    *,
    config_root: str | Path,
    manifest_path: str | Path,
    task_type: str,
    model_id: str,
    output_dir: str | Path,
    layer: int | None = None,
    expected_split: str | None = None,
    feature_dim: int = 8,
) -> Path:
    """Extract deterministic fixture features and write a summary artifact."""

    registry = ModelRegistry(config_root)
    model = registry.get_model(model_id)
    selected_layer = layer if layer is not None else int(model["default_layers"]["last"])
    dataset = ManifestDataset(
        manifest_path,
        task_type=task_type,
        expected_split=expected_split,
    )
    features = np.vstack(
        [
            _deterministic_feature_vector(record.image, selected_layer, feature_dim)
            for record in dataset.records
        ]
    ).astype("float32")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.npz"
    np.savez_compressed(
        feature_path,
        features=features,
        image=[record.image for record in dataset.records],
        split=[record.split for record in dataset.records],
    )
    transform = model["transform"]
    summary = {
        "record_type": "feature_extraction_summary",
        "model_id": model_id,
        "layer": selected_layer,
        "input_manifest": str(manifest_path),
        "output_npz": str(feature_path),
        "num_examples": len(dataset),
        "feature_shape": list(features.shape),
        "token_format": "fixture_global",
        "transform": transform,
        "fixture": True,
    }
    validate_feature_extraction_summary(summary)
    summary_path = output_dir / "feature_extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "run_id": f"fixture_extract_{model_id}_l{selected_layer}",
        "command": "feature-economy extract-features",
        "git_commit": current_git_commit(),
        "config_files": [str(path) for path in sorted(Path(config_root).rglob("*.yaml"))],
        "inputs": {
            "manifest": str(manifest_path),
            "task_type": task_type,
            "model_id": model_id,
            "layer": selected_layer,
        },
        "outputs": {
            "features": str(feature_path),
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


def extract_huggingface_features(
    *,
    config_root: str | Path,
    manifest_path: str | Path,
    task_type: str,
    model_id: str,
    output_dir: str | Path,
    layer: int | None = None,
    expected_split: str | None = None,
    batch_size: int = 16,
    device: str = "cpu",
    dtype: str = "float32",
    max_examples: int | None = None,
    local_files_only: bool = False,
) -> Path:
    """Extract hidden states with a HuggingFace vision backbone.

    This is the first real backbone backend in the public repo. It currently
    supports configs whose `provider` is `huggingface` and whose `hf_name` is
    set, which covers the public DINOv2 config. Local I-JEPA loading should be
    added as a separate backend rather than hidden behind this function.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive when provided")
    registry = ModelRegistry(config_root)
    model_config = registry.get_model(model_id)
    if model_config.get("provider") != "huggingface" or not model_config.get("hf_name"):
        raise NotImplementedError(
            f"model {model_id} does not define a HuggingFace backbone; "
            "add hf_name or implement a model-specific loader"
        )
    selected_layer = layer if layer is not None else int(model_config["default_layers"]["last"])
    dataset = ManifestDataset(
        manifest_path,
        task_type=task_type,
        expected_split=expected_split,
    )
    records = dataset.records[:max_examples] if max_examples is not None else dataset.records
    if not records:
        raise ValueError("no records selected for feature extraction")

    hf_name = str(model_config["hf_name"])
    policy = transform_policy_from_config(dict(model_config["transform"]))
    torch, auto_model = _import_huggingface_runtime()
    model = auto_model.from_pretrained(hf_name, local_files_only=local_files_only)
    model.to(device)
    model.eval()

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            batch = np.stack(
                [
                    preprocess_image_file(dataset.resolve_path(record.image), policy)
                    for record in batch_records
                ],
                axis=0,
            )
            pixel_values = torch.from_numpy(batch).to(device)
            result = model(pixel_values=pixel_values, output_hidden_states=True)
            hidden = _select_hidden_state(result.hidden_states, selected_layer)
            outputs.append(hidden.detach().cpu().numpy())

    features = np.concatenate(outputs, axis=0).astype(dtype)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.npz"
    np.savez_compressed(
        feature_path,
        features=features,
        image=[record.image for record in records],
        split=[record.split for record in records],
    )
    return _write_feature_artifacts(
        config_root=config_root,
        manifest_path=manifest_path,
        task_type=task_type,
        model_id=model_id,
        layer=selected_layer,
        output_dir=output_dir,
        feature_path=feature_path,
        feature_shape=list(features.shape),
        token_format=str(model_config["output"]["token_format"]),
        transform=dict(model_config["transform"]),
        fixture=False,
        backend="huggingface",
        extra_inputs={
            "hf_name": hf_name,
            "batch_size": batch_size,
            "device": device,
            "dtype": dtype,
            "max_examples": max_examples,
            "local_files_only": local_files_only,
        },
    )


def extract_torchscript_features(
    *,
    config_root: str | Path,
    manifest_path: str | Path,
    task_type: str,
    model_id: str,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    layer: int | None = None,
    expected_split: str | None = None,
    batch_size: int = 16,
    device: str = "cpu",
    dtype: str = "float32",
    max_examples: int | None = None,
    output_key: str | None = None,
    output_index: int = 0,
) -> Path:
    """Extract features from a TorchScript module.

    The TorchScript module should accept normalized NCHW image tensors and
    return the desired layer features directly, either as a tensor, tuple/list,
    or dictionary. This backend is intended for I-JEPA or other local backbones
    whose public loader is not available through HuggingFace.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive when provided")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"TorchScript checkpoint does not exist: {checkpoint_path}")
    registry = ModelRegistry(config_root)
    model_config = registry.get_model(model_id)
    selected_layer = layer if layer is not None else int(model_config["default_layers"]["last"])
    dataset = ManifestDataset(
        manifest_path,
        task_type=task_type,
        expected_split=expected_split,
    )
    records = dataset.records[:max_examples] if max_examples is not None else dataset.records
    if not records:
        raise ValueError("no records selected for feature extraction")

    torch = _import_torch_runtime()
    policy = transform_policy_from_config(dict(model_config["transform"]))
    module = torch.jit.load(str(checkpoint_path), map_location=device)
    module.to(device)
    module.eval()

    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            batch = np.stack(
                [
                    preprocess_image_file(dataset.resolve_path(record.image), policy)
                    for record in batch_records
                ],
                axis=0,
            )
            pixel_values = torch.from_numpy(batch).to(device)
            result = module(pixel_values)
            features = _select_torchscript_output(
                result,
                output_key=output_key,
                output_index=output_index,
            )
            outputs.append(features.detach().cpu().numpy())

    features_np = np.concatenate(outputs, axis=0).astype(dtype)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features.npz"
    np.savez_compressed(
        feature_path,
        features=features_np,
        image=[record.image for record in records],
        split=[record.split for record in records],
    )
    return _write_feature_artifacts(
        config_root=config_root,
        manifest_path=manifest_path,
        task_type=task_type,
        model_id=model_id,
        layer=selected_layer,
        output_dir=output_dir,
        feature_path=feature_path,
        feature_shape=list(features_np.shape),
        token_format=str(model_config["output"]["token_format"]),
        transform=dict(model_config["transform"]),
        fixture=False,
        backend="torchscript",
        extra_inputs={
            "checkpoint": str(checkpoint_path),
            "batch_size": batch_size,
            "device": device,
            "dtype": dtype,
            "max_examples": max_examples,
            "output_key": output_key,
            "output_index": output_index,
        },
    )


def _deterministic_feature_vector(key: str, layer: int, feature_dim: int) -> np.ndarray:
    values = []
    base = sum(ord(ch) for ch in key) + layer * 997
    for index in range(feature_dim):
        values.append(((base + index * 37) % 1000) / 1000.0)
    return np.asarray(values, dtype=np.float32)


def _write_feature_artifacts(
    *,
    config_root: str | Path,
    manifest_path: str | Path,
    task_type: str,
    model_id: str,
    layer: int,
    output_dir: Path,
    feature_path: Path,
    feature_shape: list[int],
    token_format: str,
    transform: dict[str, Any],
    fixture: bool,
    backend: str,
    extra_inputs: dict[str, Any] | None = None,
) -> Path:
    summary = {
        "record_type": "feature_extraction_summary",
        "model_id": model_id,
        "layer": layer,
        "input_manifest": str(manifest_path),
        "output_npz": str(feature_path),
        "num_examples": int(feature_shape[0]),
        "feature_shape": feature_shape,
        "token_format": token_format,
        "transform": transform,
        "fixture": fixture,
        "backend": backend,
    }
    validate_feature_extraction_summary(summary)
    summary_path = output_dir / "feature_extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    inputs = {
        "manifest": str(manifest_path),
        "task_type": task_type,
        "model_id": model_id,
        "layer": layer,
        "backend": backend,
    }
    if extra_inputs:
        inputs.update(extra_inputs)
    manifest = {
        "run_id": f"{backend}_extract_{model_id}_l{layer}",
        "command": "feature-economy extract-features",
        "git_commit": current_git_commit(),
        "config_files": [str(path) for path in sorted(Path(config_root).rglob("*.yaml"))],
        "inputs": inputs,
        "outputs": {
            "features": str(feature_path),
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


def _select_hidden_state(hidden_states: Any, layer: int) -> Any:
    """Select encoder layer `layer` from HF hidden_states.

    HuggingFace ViT-style outputs include the patch embedding output at index 0,
    then one entry per encoder block. Paper layer 11 therefore maps to
    `hidden_states[12]`.
    """

    if hidden_states is None:
        raise RuntimeError("model output did not include hidden_states")
    index = layer + 1
    if index >= len(hidden_states):
        raise IndexError(
            f"requested layer {layer}, but hidden_states has only {len(hidden_states)} entries"
        )
    return hidden_states[index]


def _import_huggingface_runtime() -> tuple[Any, Any]:
    try:
        import torch
        from transformers import AutoModel
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "HuggingFace extraction requires torch and transformers. "
            "Install with `pip install -e '.[experiments]'`."
        ) from exc
    return torch, AutoModel


def _import_torch_runtime() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "TorchScript extraction requires torch. Install with `pip install -e '.[experiments]'`."
        ) from exc
    return torch


def _select_torchscript_output(
    result: Any,
    *,
    output_key: str | None,
    output_index: int,
) -> Any:
    if output_key is not None:
        if not isinstance(result, dict):
            raise TypeError("--output-key requires the TorchScript module to return a dict")
        if output_key not in result:
            raise KeyError(f"TorchScript output missing key {output_key!r}")
        return result[output_key]
    if isinstance(result, dict):
        if len(result) == 1:
            return next(iter(result.values()))
        keys = ", ".join(sorted(str(key) for key in result))
        raise ValueError(
            "TorchScript module returned a dict with multiple outputs; "
            f"pass --output-key. Available keys: {keys}"
        )
    if isinstance(result, (tuple, list)):
        return result[output_index]
    return result
