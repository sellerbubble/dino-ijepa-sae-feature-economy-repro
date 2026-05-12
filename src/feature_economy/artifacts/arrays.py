"""Validation for public `.npz` array contracts.

These checks guard the saved-array boundary used by external backbone exports,
SAE-code extraction, linear probes, ranking, and ablation. They intentionally
validate the minimal public contract rather than every private artifact variant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from feature_economy.data import ManifestDataset

from .schemas import SchemaError


ARRAY_KINDS = {"features", "codes", "probe_logits", "targets", "contribution_scores"}
TASK_TYPES = {"classification", "count_classification", "dense_depth", "dense_segmentation"}


def validate_array_contract(
    *,
    npz_path: str | Path,
    kind: str,
    task_type: str | None = None,
    manifest_path: str | Path | None = None,
    expected_split: str | None = None,
    target_key: str = "targets",
    write_json: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a public saved-array `.npz` and return a compact report."""

    if kind not in ARRAY_KINDS:
        raise SchemaError(f"unsupported array kind: {kind}")
    if task_type is not None and task_type not in TASK_TYPES:
        raise SchemaError(f"unsupported task_type: {task_type}")
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"array file does not exist: {npz_path}")
    arrays = np.load(npz_path)
    manifest_rows = _manifest_row_count(
        manifest_path=manifest_path,
        task_type=task_type,
        expected_split=expected_split,
    )
    report = {
        "record_type": "array_contract_validation",
        "path": str(npz_path),
        "kind": kind,
        "task_type": task_type,
        "arrays": {key: list(np.asarray(arrays[key]).shape) for key in arrays.files},
        "num_manifest_rows": manifest_rows,
        "valid": True,
    }
    if kind == "features":
        _validate_features(arrays, manifest_rows=manifest_rows)
    elif kind == "codes":
        _validate_codes(arrays, manifest_rows=manifest_rows)
    elif kind == "targets":
        _validate_targets(
            arrays,
            task_type=task_type,
            target_key=target_key,
            manifest_rows=manifest_rows,
        )
    elif kind == "probe_logits":
        _validate_probe_logits(arrays, task_type=task_type, manifest_rows=manifest_rows)
    elif kind == "contribution_scores":
        _validate_contribution_scores(arrays)
    if write_json is not None:
        write_json = Path(write_json)
        write_json.parent.mkdir(parents=True, exist_ok=True)
        write_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _manifest_row_count(
    *,
    manifest_path: str | Path | None,
    task_type: str | None,
    expected_split: str | None,
) -> int | None:
    if manifest_path is None:
        return None
    if task_type is None:
        raise SchemaError("--manifest requires --task-type")
    dataset = ManifestDataset(
        manifest_path,
        task_type=task_type,
        expected_split=expected_split,
    )
    return len(dataset)


def _validate_features(arrays: Any, *, manifest_rows: int | None) -> None:
    features = _require_array(arrays, "features")
    if features.ndim < 2:
        raise SchemaError("features must have shape [num_examples, ..., feature_dim]")
    _require_positive_dims(features, "features")
    _check_manifest_rows(features, manifest_rows, "features")
    _require_floating(features, "features")


def _validate_codes(arrays: Any, *, manifest_rows: int | None) -> None:
    codes = _require_array(arrays, "codes")
    if codes.ndim < 2:
        raise SchemaError("codes must have shape [num_examples, ..., code_dim]")
    _require_positive_dims(codes, "codes")
    _check_manifest_rows(codes, manifest_rows, "codes")
    _require_floating(codes, "codes")


def _validate_targets(
    arrays: Any,
    *,
    task_type: str | None,
    target_key: str,
    manifest_rows: int | None,
) -> None:
    if task_type not in {"dense_depth", "dense_segmentation"}:
        raise SchemaError("targets contract requires --task-type dense_depth or dense_segmentation")
    targets = _require_array(arrays, target_key)
    if targets.ndim < 2:
        raise SchemaError(f"{target_key} must have shape [num_examples, ...]")
    _require_positive_dims(targets, target_key)
    _check_manifest_rows(targets, manifest_rows, target_key)
    if task_type == "dense_depth":
        _require_floating(targets, target_key)
    else:
        _require_integer(targets, target_key)


def _validate_probe_logits(
    arrays: Any,
    *,
    task_type: str | None,
    manifest_rows: int | None,
) -> None:
    if task_type is None:
        raise SchemaError("probe_logits contract requires --task-type")
    if task_type in {"classification", "count_classification"}:
        logits = _require_array(arrays, "logits")
        weights = _require_array(arrays, "weights")
        classes = _require_array(arrays, "classes")
        labels = _require_array(arrays, "labels")
        if logits.ndim != 2:
            raise SchemaError("classification logits must have shape [num_examples, num_classes]")
        if weights.ndim != 2:
            raise SchemaError("classification weights must have shape [feature_dim + 1, num_classes]")
        if classes.ndim != 1 or labels.ndim != 1:
            raise SchemaError("classes and labels must be one-dimensional")
        if logits.shape[1] != classes.shape[0] or weights.shape[1] != classes.shape[0]:
            raise SchemaError("logits/weights class dimension must match classes")
        if logits.shape[0] != labels.shape[0]:
            raise SchemaError("logits rows must match labels")
        _check_manifest_rows(logits, manifest_rows, "logits")
        return
    if task_type == "dense_depth":
        prediction = _require_array(arrays, "prediction")
        weights = _require_array(arrays, "weights")
        targets = _require_array(arrays, "targets")
        if prediction.shape != targets.shape:
            raise SchemaError("dense_depth prediction shape must match targets")
        if weights.ndim != 1:
            raise SchemaError("dense_depth weights must have shape [feature_dim + 1]")
        _check_manifest_rows(prediction, manifest_rows, "prediction")
        return
    if task_type == "dense_segmentation":
        logits = _require_array(arrays, "logits")
        weights = _require_array(arrays, "weights")
        classes = _require_array(arrays, "classes")
        targets = _require_array(arrays, "targets")
        if logits.ndim < 3:
            raise SchemaError("dense_segmentation logits must have shape target_shape + [num_classes]")
        if weights.ndim != 2:
            raise SchemaError("dense_segmentation weights must have shape [feature_dim + 1, num_classes]")
        if classes.ndim != 1:
            raise SchemaError("dense_segmentation classes must be one-dimensional")
        if logits.shape[:-1] != targets.shape:
            raise SchemaError("dense_segmentation logits spatial shape must match targets")
        if logits.shape[-1] != classes.shape[0] or weights.shape[1] != classes.shape[0]:
            raise SchemaError("logits/weights class dimension must match classes")
        _check_manifest_rows(logits, manifest_rows, "logits")
        return
    raise SchemaError(f"unsupported task_type for probe_logits: {task_type}")


def _validate_contribution_scores(arrays: Any) -> None:
    scores = _require_array(arrays, "validation_contribution_score")
    if scores.ndim != 1:
        raise SchemaError("validation_contribution_score must have shape [code_dim]")
    _require_positive_dims(scores, "validation_contribution_score")
    _require_floating(scores, "validation_contribution_score")


def _require_array(arrays: Any, key: str) -> np.ndarray:
    if key not in arrays:
        raise SchemaError(f"array file missing required key: {key}")
    return np.asarray(arrays[key])


def _require_positive_dims(array: np.ndarray, name: str) -> None:
    if any(dim <= 0 for dim in array.shape):
        raise SchemaError(f"{name} has non-positive dimensions: {array.shape}")


def _check_manifest_rows(array: np.ndarray, manifest_rows: int | None, name: str) -> None:
    if manifest_rows is not None and array.shape[0] != manifest_rows:
        raise SchemaError(
            f"{name} rows ({array.shape[0]}) must match manifest rows ({manifest_rows})"
        )


def _require_floating(array: np.ndarray, name: str) -> None:
    if not np.issubdtype(array.dtype, np.floating):
        raise SchemaError(f"{name} must use a floating dtype")


def _require_integer(array: np.ndarray, name: str) -> None:
    if not np.issubdtype(array.dtype, np.integer):
        raise SchemaError(f"{name} must use an integer dtype")
