"""Availability analysis from saved SAE-code arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from feature_economy.artifacts import (
    current_git_commit,
    validate_availability_summary,
    validate_run_manifest,
)


DEFAULT_BUCKETS = {
    "dead": (0, 0),
    "rare": (1, 10),
    "medium": (11, 100),
    "frequent_or_above": (101, None),
}


def compute_availability_from_codes(
    *,
    codes_npz: str | Path,
    model_id: str,
    sae_id: str,
    dataset_id: str,
    split: str,
    output_dir: str | Path,
    threshold: float = 0.0,
) -> Path:
    """Compute fired-count availability statistics from saved SAE codes."""

    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    codes_npz = Path(codes_npz)
    arrays = np.load(codes_npz)
    if "codes" not in arrays:
        raise ValueError(f"{codes_npz} does not contain array 'codes'")
    codes = np.asarray(arrays["codes"])
    if codes.ndim < 2:
        raise ValueError("codes must have shape [..., num_features]")
    flat = codes.reshape(-1, codes.shape[-1])
    fired_count = np.sum(flat > threshold, axis=0).astype(np.int64)
    total_features = int(fired_count.shape[0])
    dead_features = int(np.sum(fired_count == 0))
    fired_features = int(total_features - dead_features)
    active_counts = fired_count[fired_count > 0]
    record = {
        "record_type": "availability_summary",
        "model_id": model_id,
        "sae_id": sae_id,
        "dataset_id": dataset_id,
        "split": split,
        "usage_unit": "fired_count",
        "total_features": total_features,
        "fired_features": fired_features,
        "dead_features": dead_features,
        "buckets": _bucket_counts(fired_count),
        "active_fired_count_quantiles": _active_quantiles(active_counts),
        "threshold": threshold,
        "num_positions": int(flat.shape[0]),
        "input_codes": str(codes_npz),
        "fixture": False,
    }
    validate_availability_summary(record)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "availability_summary.json"
    summary_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "run_id": f"availability_{dataset_id}_{split}_{model_id}_{sae_id}",
        "command": "feature-economy compute-usage",
        "git_commit": current_git_commit(),
        "config_files": [],
        "inputs": {
            "codes_npz": str(codes_npz),
            "model_id": model_id,
            "sae_id": sae_id,
            "dataset_id": dataset_id,
            "split": split,
            "threshold": threshold,
        },
        "outputs": {"summary": str(summary_path)},
        "fixture": False,
    }
    validate_run_manifest(manifest)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def _bucket_counts(fired_count: np.ndarray) -> dict[str, int]:
    buckets: dict[str, int] = {}
    for name, (lower, upper) in DEFAULT_BUCKETS.items():
        mask = fired_count >= lower
        if upper is not None:
            mask = np.logical_and(mask, fired_count <= upper)
        buckets[name] = int(np.sum(mask))
    return buckets


def _active_quantiles(active_counts: np.ndarray) -> dict[str, float]:
    if active_counts.size == 0:
        return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    quantiles: dict[str, Any] = {
        "p10": np.percentile(active_counts, 10),
        "p25": np.percentile(active_counts, 25),
        "p50": np.percentile(active_counts, 50),
        "p75": np.percentile(active_counts, 75),
        "p90": np.percentile(active_counts, 90),
    }
    return {key: float(value) for key, value in quantiles.items()}
