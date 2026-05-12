"""Table builders for public reproduction artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from feature_economy.artifacts import (
    validate_ablation_summary,
    validate_availability_summary,
    validate_probe_summary,
    validate_subset_usage_summary,
)
from feature_economy.artifacts.schemas import SchemaError


PROBE_SCORE_COLUMNS = [
    "record_type",
    "task_id",
    "model_id",
    "sae_id",
    "seed",
    "metric",
    "value",
    "smoke",
    "source_path",
]

AVAILABILITY_COLUMNS = [
    "model_id",
    "sae_id",
    "dataset_id",
    "split",
    "usage_unit",
    "total_features",
    "fired_features",
    "dead_features",
    "active_ratio",
    "dead_ratio",
    "bucket",
    "bucket_count",
    "smoke",
    "source_path",
]

SUBSET_USAGE_COLUMNS = [
    "task_id",
    "model_id",
    "sae_id",
    "ranking_method",
    "usage_unit",
    "subset",
    "selection",
    "num_features",
    "median_usage",
    "high_usage_fraction",
    "smoke",
    "source_path",
]

ABLATION_COLUMNS = [
    "task_id",
    "model_id",
    "sae_id",
    "subset",
    "selection",
    "num_features",
    "metric",
    "baseline_value",
    "value",
    "delta",
    "smoke",
    "source_path",
]


def write_probe_score_table(input_dir: str | Path, output_csv: str | Path) -> int:
    """Write a long-form probe score CSV from saved probe summary artifacts.

    The function recursively scans `input_dir` for JSON files, keeps files that
    satisfy the public probe-summary schema, and ignores other JSON artifacts.
    This makes it suitable for mixed artifact directories while keeping table
    generation deterministic.

    Returns the number of metric rows written.
    """

    input_dir = Path(input_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*.json")):
        record = _load_json_mapping(path)
        if record is None:
            continue
        try:
            validate_probe_summary(record)
        except SchemaError:
            continue
        for metric, value in sorted(record["metrics"].items()):
            rows.append(
                {
                    "record_type": record["record_type"],
                    "task_id": record["task_id"],
                    "model_id": record["model_id"],
                    "sae_id": record.get("sae_id", ""),
                    "seed": record["seed"],
                    "metric": metric,
                    "value": value,
                    "smoke": bool(record.get("smoke", False)),
                    "source_path": str(path),
                }
            )

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROBE_SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_probe_score_table_from_index(index_json: str | Path, output_csv: str | Path) -> int:
    """Write a probe score CSV from an explicit artifact index."""

    index_json = Path(index_json)
    with index_json.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    if not isinstance(index, Mapping) or index.get("record_type") != "artifact_index":
        raise ValueError(f"not an artifact index: {index_json}")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for entry in index.get("records", []):
        if not isinstance(entry, Mapping) or not entry.get("valid"):
            continue
        if entry.get("record_type") not in {"native_probe_summary", "sae_probe_summary"}:
            continue
        path = Path(str(entry["path"]))
        record = _load_json_mapping(path)
        if record is None:
            continue
        validate_probe_summary(record)
        for metric, value in sorted(record["metrics"].items()):
            rows.append(
                {
                    "record_type": record["record_type"],
                    "task_id": record["task_id"],
                    "model_id": record["model_id"],
                    "sae_id": record.get("sae_id", ""),
                    "seed": record["seed"],
                    "metric": metric,
                    "value": value,
                    "smoke": bool(record.get("smoke", False)),
                    "source_path": str(path),
                }
            )

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROBE_SCORE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_all_tables_from_index(index_json: str | Path, output_dir: str | Path) -> dict[str, int]:
    """Write all currently supported paper-facing CSV tables from an index."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "probe_scores": write_probe_score_table_from_index(
            index_json,
            output_dir / "probe_scores.csv",
        ),
        "availability_summary": write_availability_table_from_index(
            index_json,
            output_dir / "availability_summary.csv",
        ),
        "subset_usage": write_subset_usage_table_from_index(
            index_json,
            output_dir / "subset_usage_summary.csv",
        ),
        "ablation_summary": write_ablation_table_from_index(
            index_json,
            output_dir / "ablation_summary.csv",
        ),
    }


def write_availability_table_from_index(index_json: str | Path, output_csv: str | Path) -> int:
    """Write availability bucket rows from indexed artifacts."""

    rows: list[dict[str, Any]] = []
    for record, path in _iter_valid_records(index_json, "availability_summary"):
        validate_availability_summary(record)
        total = record["total_features"]
        fired = record["fired_features"]
        dead = record["dead_features"]
        active_ratio = fired / total if total else 0.0
        dead_ratio = dead / total if total else 0.0
        for bucket, bucket_count in sorted(record["buckets"].items()):
            rows.append(
                {
                    "model_id": record["model_id"],
                    "sae_id": record["sae_id"],
                    "dataset_id": record["dataset_id"],
                    "split": record["split"],
                    "usage_unit": record["usage_unit"],
                    "total_features": total,
                    "fired_features": fired,
                    "dead_features": dead,
                    "active_ratio": active_ratio,
                    "dead_ratio": dead_ratio,
                    "bucket": bucket,
                    "bucket_count": bucket_count,
                    "smoke": bool(record.get("smoke", False)),
                    "source_path": str(path),
                }
            )
    _write_rows(output_csv, AVAILABILITY_COLUMNS, rows)
    return len(rows)


def write_subset_usage_table_from_index(index_json: str | Path, output_csv: str | Path) -> int:
    """Write task-selected and random subset usage rows from indexed artifacts."""

    rows: list[dict[str, Any]] = []
    for record, path in _iter_valid_records(index_json, "subset_usage_summary"):
        validate_subset_usage_summary(record)
        for subset in record["subsets"]:
            rows.append(
                {
                    "task_id": record["task_id"],
                    "model_id": record["model_id"],
                    "sae_id": record["sae_id"],
                    "ranking_method": record["ranking_method"],
                    "usage_unit": record["usage_unit"],
                    "subset": subset["name"],
                    "selection": subset["selection"],
                    "num_features": subset["num_features"],
                    "median_usage": subset["median_usage"],
                    "high_usage_fraction": subset["high_usage_fraction"],
                    "smoke": bool(record.get("smoke", False)),
                    "source_path": str(path),
                }
            )
    _write_rows(output_csv, SUBSET_USAGE_COLUMNS, rows)
    return len(rows)


def write_ablation_table_from_index(index_json: str | Path, output_csv: str | Path) -> int:
    """Write feature ablation and matched-random control rows from indexed artifacts."""

    rows: list[dict[str, Any]] = []
    for record, path in _iter_valid_records(index_json, "feature_ablation_summary"):
        validate_ablation_summary(record)
        baseline_metrics = record["baseline_metrics"]
        for ablation in record["ablations"]:
            rows.extend(
                _ablation_rows(
                    record=record,
                    item=ablation,
                    selection="ablated",
                    baseline_metrics=baseline_metrics,
                    path=path,
                )
            )
        for control in record["random_controls"]:
            rows.extend(
                _ablation_rows(
                    record=record,
                    item=control,
                    selection="random_control",
                    baseline_metrics=baseline_metrics,
                    path=path,
                )
            )
    _write_rows(output_csv, ABLATION_COLUMNS, rows)
    return len(rows)


def _load_json_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
    except json.JSONDecodeError:
        return None
    if not isinstance(record, Mapping):
        return None
    return record


def _load_index(index_json: str | Path) -> Mapping[str, Any]:
    index_json = Path(index_json)
    with index_json.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    if not isinstance(index, Mapping) or index.get("record_type") != "artifact_index":
        raise ValueError(f"not an artifact index: {index_json}")
    return index


def _iter_valid_records(
    index_json: str | Path,
    record_type: str,
) -> list[tuple[Mapping[str, Any], Path]]:
    index = _load_index(index_json)
    records: list[tuple[Mapping[str, Any], Path]] = []
    for entry in index.get("records", []):
        if not isinstance(entry, Mapping) or not entry.get("valid"):
            continue
        if entry.get("record_type") != record_type:
            continue
        path = Path(str(entry["path"]))
        record = _load_json_mapping(path)
        if record is not None:
            records.append((record, path))
    return records


def _write_rows(output_csv: str | Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _ablation_rows(
    *,
    record: Mapping[str, Any],
    item: Mapping[str, Any],
    selection: str,
    baseline_metrics: Mapping[str, Any],
    path: Path,
) -> list[dict[str, Any]]:
    metric_delta = item.get("metric_delta", {})
    metrics = item.get("metrics", {})
    rows = []
    for metric, delta in sorted(metric_delta.items()):
        rows.append(
            {
                "task_id": record["task_id"],
                "model_id": record["model_id"],
                "sae_id": record.get("sae_id", ""),
                "subset": item["subset"],
                "selection": selection,
                "num_features": item["num_features"],
                "metric": metric,
                "baseline_value": baseline_metrics.get(metric, ""),
                "value": metrics.get(metric, ""),
                "delta": delta,
                "smoke": bool(record.get("smoke", False)),
                "source_path": str(path),
            }
        )
    return rows
