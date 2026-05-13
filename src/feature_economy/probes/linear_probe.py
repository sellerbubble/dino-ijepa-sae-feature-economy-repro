"""Lightweight linear probes over saved feature or SAE-code arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from feature_economy.artifacts import (
    current_git_commit,
    validate_probe_summary,
    validate_run_manifest,
)
from feature_economy.data import ManifestDataset

from .metrics import depth_metrics, segmentation_metrics, topk_accuracy


def train_sae_linear_probe(
    *,
    codes_npz: str | Path,
    manifest_path: str | Path,
    task_type: str,
    task_id: str,
    model_id: str,
    sae_id: str,
    output_dir: str | Path,
    expected_split: str | None = None,
    seed: int = 0,
    ridge: float = 1e-3,
    targets_npz: str | Path | None = None,
    target_key: str = "targets",
    num_classes: int | None = None,
    ignore_index: int | None = None,
) -> Path:
    """Train/evaluate a closed-form ridge classifier on saved SAE codes."""

    return _train_linear_probe(
        array_npz=codes_npz,
        array_key="codes",
        manifest_path=manifest_path,
        task_type=task_type,
        task_id=task_id,
        model_id=model_id,
        sae_id=sae_id,
        output_dir=output_dir,
        expected_split=expected_split,
        seed=seed,
        ridge=ridge,
        targets_npz=targets_npz,
        target_key=target_key,
        num_classes=num_classes,
        ignore_index=ignore_index,
        record_type="sae_probe_summary",
    )


def train_native_linear_probe(
    *,
    features_npz: str | Path,
    manifest_path: str | Path,
    task_type: str,
    task_id: str,
    model_id: str,
    output_dir: str | Path,
    expected_split: str | None = None,
    seed: int = 0,
    ridge: float = 1e-3,
    targets_npz: str | Path | None = None,
    target_key: str = "targets",
    num_classes: int | None = None,
    ignore_index: int | None = None,
) -> Path:
    """Train/evaluate a closed-form ridge classifier on saved native features."""

    return _train_linear_probe(
        array_npz=features_npz,
        array_key="features",
        manifest_path=manifest_path,
        task_type=task_type,
        task_id=task_id,
        model_id=model_id,
        sae_id=None,
        output_dir=output_dir,
        expected_split=expected_split,
        seed=seed,
        ridge=ridge,
        targets_npz=targets_npz,
        target_key=target_key,
        num_classes=num_classes,
        ignore_index=ignore_index,
        record_type="native_probe_summary",
    )


def _train_linear_probe(
    *,
    array_npz: str | Path,
    array_key: str,
    manifest_path: str | Path,
    task_type: str,
    task_id: str,
    model_id: str,
    sae_id: str | None,
    output_dir: str | Path,
    expected_split: str | None,
    seed: int,
    ridge: float,
    targets_npz: str | Path | None,
    target_key: str,
    num_classes: int | None,
    ignore_index: int | None,
    record_type: str,
) -> Path:
    """Shared closed-form ridge classifier implementation."""

    if task_type not in {
        "classification",
        "count_classification",
        "dense_depth",
        "dense_segmentation",
    }:
        raise ValueError(f"unsupported task_type: {task_type}")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    dataset = ManifestDataset(
        manifest_path,
        task_type=task_type,
        expected_split=expected_split,
    )
    arrays = np.load(array_npz)
    if array_key not in arrays:
        raise ValueError(f"{array_npz} does not contain array {array_key!r}")
    features = np.asarray(arrays[array_key], dtype=np.float64)
    if features.shape[0] != len(dataset):
        raise ValueError(
            f"{array_key} rows ({features.shape[0]}) must match manifest rows ({len(dataset)})"
        )
    if task_type in {"classification", "count_classification"}:
        labels = _labels_to_int(np.asarray([record.label for record in dataset.records]))
        metrics, probe_arrays = _fit_classification_probe(
            features=features,
            labels=labels,
            task_type=task_type,
            ridge=ridge,
        )
    else:
        if targets_npz is None:
            raise ValueError(f"{task_type} linear probe requires targets_npz")
        targets = _load_targets(targets_npz, target_key)
        metrics, probe_arrays = _fit_dense_probe(
            features=features,
            targets=targets,
            task_type=task_type,
            ridge=ridge,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logits_path = output_dir / "probe_logits.npz"
    np.savez_compressed(
        logits_path,
        **probe_arrays,
    )
    summary = {
        "record_type": record_type,
        "task_id": task_id,
        "model_id": model_id,
        "seed": seed,
        "metrics": metrics,
        "checkpoint": "closed_form_ridge",
        "fixture": False,
        "backend": "linear_probe",
    }
    if sae_id is not None:
        summary["sae_id"] = sae_id
    validate_probe_summary(summary)
    summary_path = output_dir / f"{record_type}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    run_id_parts = ["linear_probe", task_id, model_id]
    if sae_id is not None:
        run_id_parts.append(sae_id)
    inputs = {
        f"{array_key}_npz": str(array_npz),
        "manifest": str(manifest_path),
        "task_type": task_type,
        "task_id": task_id,
        "model_id": model_id,
        "ridge": ridge,
    }
    if targets_npz is not None:
        inputs["targets_npz"] = str(targets_npz)
        inputs["target_key"] = target_key
    if num_classes is not None:
        inputs["num_classes"] = num_classes
    if ignore_index is not None:
        inputs["ignore_index"] = ignore_index
    if sae_id is not None:
        inputs["sae_id"] = sae_id
    manifest = {
        "run_id": "_".join(run_id_parts),
        "command": "feature-economy probe-sae" if sae_id is not None else "feature-economy probe-native",
        "git_commit": current_git_commit(),
        "config_files": [],
        "inputs": inputs,
        "outputs": {
            "summary": str(summary_path),
            "logits": str(logits_path),
        },
        "fixture": False,
    }
    validate_run_manifest(manifest)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def _fit_classification_probe(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    ridge: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    flat = _classification_readout_features(features)
    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError("linear probe requires at least two classes")
    logits, weights = _fit_ridge_logits(flat, labels, classes=classes, ridge=ridge)
    if task_type == "classification":
        topk = (1, min(5, classes.size))
        metrics = topk_accuracy(logits, _labels_to_class_indices(labels, classes), topk=topk)
    else:
        pred_indices = np.argmax(logits, axis=1)
        predictions = classes[pred_indices]
        metrics = {"accuracy": float(np.mean(predictions == labels))}
    arrays = {
        "logits": logits.astype("float32"),
        "weights": weights.astype("float32"),
        "classes": classes,
        "labels": labels,
    }
    return metrics, arrays


def _classification_readout_features(features: np.ndarray) -> np.ndarray:
    """Return one feature vector per example for classification probes.

    Public full-profile feature extraction stores token maps such as
    `[batch, tokens, dim]` for native backbones and `[batch, tokens, code_dim]`
    for SAE codes. Flattening tokens into the readout makes tiny real-data
    checks numerically huge and breaks feature-level ranking. Mean-pooling over
    non-feature axes keeps the final dimension as the interpretable channel
    basis used by Access and Allocation analyses.
    """

    if features.ndim < 2:
        raise ValueError("classification features must have shape [num_examples, feature_dim]")
    if features.ndim == 2:
        return features
    axes = tuple(range(1, features.ndim - 1))
    return np.mean(features, axis=axes)


def _fit_dense_probe(
    *,
    features: np.ndarray,
    targets: np.ndarray,
    task_type: str,
    ridge: float,
    num_classes: int | None,
    ignore_index: int | None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    if features.shape[:-1] != targets.shape:
        raise ValueError(
            "dense features must have shape target_shape + [feature_dim]; "
            f"got features {features.shape}, targets {targets.shape}"
        )
    flat_features = features.reshape(-1, features.shape[-1])
    flat_targets = targets.reshape(-1)
    if task_type == "dense_depth":
        weights = _fit_ridge_regression(flat_features, flat_targets, ridge=ridge)
        prediction = _regression_prediction(flat_features, weights).reshape(targets.shape)
        metrics = depth_metrics(prediction, targets)
        arrays = {
            "prediction": prediction.astype("float32"),
            "weights": weights.astype("float32"),
            "targets": targets.astype("float32"),
        }
        return metrics, arrays
    if num_classes is None:
        raise ValueError("dense_segmentation linear probe requires num_classes")
    valid = np.ones(flat_targets.shape, dtype=bool)
    if ignore_index is not None:
        valid = flat_targets != ignore_index
    classes = np.arange(num_classes, dtype=np.int64)
    logits, weights = _fit_ridge_logits(
        flat_features[valid],
        flat_targets[valid].astype(np.int64),
        classes=classes,
        ridge=ridge,
    )
    full_logits = _classification_logits(flat_features, weights).reshape(*targets.shape, num_classes)
    prediction = np.argmax(full_logits, axis=-1).astype(np.int64)
    metrics = segmentation_metrics(
        prediction,
        targets,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    arrays = {
        "logits": full_logits.astype("float32"),
        "weights": weights.astype("float32"),
        "classes": classes,
        "targets": targets.astype("int64"),
    }
    return metrics, arrays


def _fit_ridge_logits(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    classes: np.ndarray,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.concatenate([features, np.ones((features.shape[0], 1))], axis=1)
    targets = (labels[:, None] == classes[None, :]).astype(np.float64)
    gram = design.T @ design
    regularizer = ridge * np.eye(gram.shape[0], dtype=np.float64)
    regularizer[-1, -1] = 0.0
    weights = np.linalg.pinv(gram + regularizer) @ design.T @ targets
    return design @ weights, weights


def _fit_ridge_regression(features: np.ndarray, targets: np.ndarray, *, ridge: float) -> np.ndarray:
    design = np.concatenate([features, np.ones((features.shape[0], 1))], axis=1)
    gram = design.T @ design
    regularizer = ridge * np.eye(gram.shape[0], dtype=np.float64)
    regularizer[-1, -1] = 0.0
    return np.linalg.pinv(gram + regularizer) @ design.T @ targets.astype(np.float64)


def _regression_prediction(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.concatenate([features, np.ones((features.shape[0], 1))], axis=1)
    return design @ weights


def _classification_logits(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.concatenate([features, np.ones((features.shape[0], 1))], axis=1)
    return design @ weights


def _load_targets(targets_npz: str | Path, target_key: str) -> np.ndarray:
    arrays = np.load(targets_npz)
    if target_key not in arrays:
        raise ValueError(f"{targets_npz} does not contain array {target_key!r}")
    return np.asarray(arrays[target_key])


def _labels_to_int(labels: np.ndarray) -> np.ndarray:
    try:
        return labels.astype(np.int64)
    except ValueError as exc:
        raise ValueError("linear probe currently requires integer labels") from exc


def _labels_to_class_indices(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    index = {int(label): idx for idx, label in enumerate(classes)}
    return np.asarray([index[int(label)] for label in labels], dtype=np.int64)
