"""Image manifest inspection for the public reproduction pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from feature_economy.data import ManifestDataset

from .registry import ModelRegistry
from .transforms import preprocess_image_file, transform_policy_from_config


def inspect_manifest_images(
    *,
    config_root: str | Path,
    manifest_path: str | Path,
    task_type: str,
    model_id: str,
    output_json: str | Path | None = None,
    expected_split: str | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    """Load and preprocess a small manifest slice, returning shape/stat metadata."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    registry = ModelRegistry(config_root)
    model = registry.get_model(model_id)
    policy = transform_policy_from_config(dict(model["transform"]))
    dataset = ManifestDataset(
        manifest_path,
        task_type=task_type,
        expected_split=expected_split,
    )
    inspected = []
    for index in range(min(limit, len(dataset))):
        record = dataset[index]
        image_path = dataset.image_path(index)
        pixels = preprocess_image_file(image_path, policy)
        inspected.append(
            {
                "index": index,
                "image": record.image,
                "resolved_image": str(image_path),
                "shape": list(pixels.shape),
                "mean": float(np.mean(pixels)),
                "std": float(np.std(pixels)),
            }
        )
    report = {
        "artifact_type": "image_manifest_inspection",
        "schema_version": 1,
        "model_id": model_id,
        "task_type": task_type,
        "input_manifest": str(manifest_path),
        "num_manifest_records": len(dataset),
        "num_inspected": len(inspected),
        "transform": {
            "mode": policy.mode,
            "resize": policy.resize,
            "crop": policy.crop,
            "normalize": policy.normalize,
            "mean": list(policy.mean),
            "std": list(policy.std),
        },
        "records": inspected,
    }
    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
