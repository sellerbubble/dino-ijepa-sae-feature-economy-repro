"""Fixture-native probe evaluation.

This module evaluates tiny manifests that include precomputed predictions. It is
the first non-smoke runner-shaped path: metrics are computed from input records
and written as a valid `native_probe_summary`. No backbone or probe training is
performed yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from feature_economy.artifacts import (
    current_git_commit,
    validate_probe_summary,
    validate_run_manifest,
)
from feature_economy.data import ManifestDataset

from .metrics import accuracy, depth_metrics, segmentation_metrics, topk_accuracy


def evaluate_native_fixture(
    *,
    manifest_path: str | Path,
    task_type: str,
    task_id: str,
    model_id: str,
    output_dir: str | Path,
    seed: int = 0,
    expected_split: str | None = None,
    num_classes: int | None = None,
    ignore_index: int | None = None,
    command_name: str = "feature-economy eval-native-fixture",
) -> Path:
    """Evaluate precomputed fixture predictions and write a native probe summary."""

    return _evaluate_fixture_probe(
        manifest_path=manifest_path,
        task_type=task_type,
        task_id=task_id,
        model_id=model_id,
        output_dir=output_dir,
        seed=seed,
        expected_split=expected_split,
        num_classes=num_classes,
        ignore_index=ignore_index,
        command_name=command_name,
        record_type="native_probe_summary",
        sae_id=None,
    )


def evaluate_sae_fixture(
    *,
    manifest_path: str | Path,
    task_type: str,
    task_id: str,
    model_id: str,
    sae_id: str,
    output_dir: str | Path,
    seed: int = 0,
    expected_split: str | None = None,
    num_classes: int | None = None,
    ignore_index: int | None = None,
    command_name: str = "feature-economy probe-sae",
) -> Path:
    """Evaluate precomputed fixture predictions and write an SAE probe summary."""

    return _evaluate_fixture_probe(
        manifest_path=manifest_path,
        task_type=task_type,
        task_id=task_id,
        model_id=model_id,
        output_dir=output_dir,
        seed=seed,
        expected_split=expected_split,
        num_classes=num_classes,
        ignore_index=ignore_index,
        command_name=command_name,
        record_type="sae_probe_summary",
        sae_id=sae_id,
    )


def _evaluate_fixture_probe(
    *,
    manifest_path: str | Path,
    task_type: str,
    task_id: str,
    model_id: str,
    output_dir: str | Path,
    seed: int,
    expected_split: str | None,
    num_classes: int | None,
    ignore_index: int | None,
    command_name: str,
    record_type: str,
    sae_id: str | None,
) -> Path:
    """Evaluate precomputed fixture predictions and write a probe summary."""

    dataset = ManifestDataset(
        manifest_path,
        task_type=task_type,
        expected_split=expected_split,
    )
    metrics = _compute_metrics(dataset, task_type, num_classes=num_classes, ignore_index=ignore_index)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "record_type": record_type,
        "task_id": task_id,
        "model_id": model_id,
        "seed": seed,
        "metrics": metrics,
        "checkpoint": "fixture_predictions",
        "fixture": True,
    }
    if sae_id is not None:
        summary["sae_id"] = sae_id
    validate_probe_summary(summary)
    summary_path = output_dir / f"{record_type}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "run_id": f"fixture_{record_type}_{task_id}_{model_id}",
        "command": command_name,
        "git_commit": current_git_commit(),
        "config_files": [],
        "inputs": {
            "manifest": str(manifest_path),
            "task_type": task_type,
            "task_id": task_id,
            "model_id": model_id,
            "sae_id": sae_id or "",
        },
        "outputs": {"summary": str(summary_path)},
        "fixture": True,
    }
    validate_run_manifest(manifest)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def _compute_metrics(
    dataset: ManifestDataset,
    task_type: str,
    *,
    num_classes: int | None,
    ignore_index: int | None,
) -> dict[str, float]:
    if task_type == "classification":
        logits = np.asarray([_extra(record, "logits") for record in dataset.records], dtype=float)
        labels = np.asarray([record.label for record in dataset.records])
        topk = (1, min(5, logits.shape[1]))
        return topk_accuracy(logits, labels, topk=topk)
    if task_type == "count_classification":
        predictions = np.asarray([_extra(record, "prediction") for record in dataset.records])
        labels = np.asarray([record.label for record in dataset.records])
        return {"accuracy": accuracy(predictions, labels)}
    if task_type == "dense_depth":
        prediction = np.asarray([_extra(record, "prediction_values") for record in dataset.records])
        target = np.asarray([_extra(record, "target_values") for record in dataset.records])
        return depth_metrics(prediction, target)
    if task_type == "dense_segmentation":
        if num_classes is None:
            raise ValueError("num_classes is required for dense_segmentation")
        prediction = np.asarray([_extra(record, "prediction_values") for record in dataset.records])
        target = np.asarray([_extra(record, "target_values") for record in dataset.records])
        return segmentation_metrics(
            prediction,
            target,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
    raise ValueError(f"unsupported task_type: {task_type}")


def _extra(record: Any, key: str) -> Any:
    if record.extra is None or key not in record.extra:
        raise ValueError(f"manifest record missing extra field {key!r}")
    return record.extra[key]
