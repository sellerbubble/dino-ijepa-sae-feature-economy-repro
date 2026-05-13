"""Artifact indexing for public reproduction outputs.

The indexer scans a directory of JSON artifacts, validates known public record
types, and writes a compact registry. This creates the bridge future paper-table
and figure builders need: claims should consume an explicit artifact index
rather than wandering through an unstructured results directory.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .schemas import (
    SchemaError,
    validate_ablation_summary,
    validate_artifact_bundle_check,
    validate_array_contract_summary,
    validate_availability_summary,
    validate_contribution_scores_summary,
    validate_dense_target_export_summary,
    validate_feature_extraction_summary,
    validate_feature_ranking,
    validate_probe_summary,
    validate_reproduction_run_plan,
    validate_run_manifest,
    validate_runtime_dependency_report,
    validate_sae_code_summary,
    validate_subset_usage_summary,
)


VALIDATORS: dict[str, Callable[[Mapping[str, Any]], None]] = {
    "native_probe_summary": validate_probe_summary,
    "sae_probe_summary": validate_probe_summary,
    "feature_extraction_summary": validate_feature_extraction_summary,
    "dense_target_export_summary": validate_dense_target_export_summary,
    "sae_code_summary": validate_sae_code_summary,
    "task_feature_ranking": validate_feature_ranking,
    "contribution_scores_summary": validate_contribution_scores_summary,
    "availability_summary": validate_availability_summary,
    "subset_usage_summary": validate_subset_usage_summary,
    "feature_ablation_summary": validate_ablation_summary,
    "array_contract_validation": validate_array_contract_summary,
    "run_manifest": validate_run_manifest,
    "runtime_dependency_report": validate_runtime_dependency_report,
    "reproduction_run_plan": validate_reproduction_run_plan,
    "artifact_bundle_check": validate_artifact_bundle_check,
}

INDEX_COLUMNS = [
    "record_type",
    "path",
    "task_id",
    "model_id",
    "sae_id",
    "seed",
    "smoke",
    "valid",
    "error",
]


def build_artifact_index(
    input_dir: str | Path,
    output_json: str | Path,
    output_csv: str | Path | None = None,
    require_valid: bool = False,
) -> dict[str, Any]:
    """Build and write an index for public JSON artifacts."""

    input_dir = Path(input_dir)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json_resolved = output_json.resolve()
    records = [
        _index_json_file(path)
        for path in sorted(input_dir.rglob("*.json"))
        if not _should_skip_json(path, output_json_resolved)
    ]
    index = {
        "record_type": "artifact_index",
        "input_dir": str(input_dir),
        "num_records": len(records),
        "num_valid": sum(1 for record in records if record["valid"]),
        "records": records,
    }
    output_json.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    if output_csv is not None:
        write_artifact_index_csv(index, output_csv)
    if require_valid and index["num_valid"] != index["num_records"]:
        invalid = [record for record in records if not record["valid"]]
        examples = "; ".join(f"{record['path']}: {record['error']}" for record in invalid[:3])
        raise ValueError(f"artifact index contains invalid records: {examples}")
    return index


def _should_skip_json(path: Path, output_json_resolved: Path) -> bool:
    """Skip generated index/package metadata when indexing a release bundle."""

    if path.resolve() == output_json_resolved:
        return True
    if path.name == "artifact_index.json":
        return True
    if path.name == "artifact_metadata_sanitization_report.json":
        return True
    if path.name == "artifact_export_manifest.json":
        return True
    if path.name.endswith("_release_manifest.json"):
        return True
    return False


def write_artifact_index_csv(index: Mapping[str, Any], output_csv: str | Path) -> None:
    """Write an artifact index CSV for quick inspection."""

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for record in index["records"]:
            writer.writerow({column: record.get(column, "") for column in INDEX_COLUMNS})


def _index_json_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        return _invalid(path, "unknown", f"json_decode_error: {exc}")
    if not isinstance(payload, Mapping):
        return _invalid(path, "unknown", "json payload is not an object")

    record_type = _infer_record_type(payload, path)
    validator = VALIDATORS.get(record_type)
    if validator is None:
        return _invalid(path, record_type, "unsupported record_type")
    try:
        validator(payload)
    except SchemaError as exc:
        return _invalid(path, record_type, str(exc))

    return {
        "record_type": record_type,
        "path": str(path),
        "task_id": payload.get("task_id", ""),
        "model_id": payload.get("model_id", ""),
        "sae_id": payload.get("sae_id", ""),
        "seed": payload.get("seed", ""),
        "smoke": bool(payload.get("smoke", False)),
        "valid": True,
        "error": "",
    }


def _infer_record_type(payload: Mapping[str, Any], path: Path) -> str:
    if "record_type" in payload:
        return str(payload["record_type"])
    if payload.get("artifact_type") == "runtime_dependency_report":
        return "runtime_dependency_report"
    if path.name == "run_manifest.json":
        return "run_manifest"
    return "unknown"


def _invalid(path: Path, record_type: str, error: str) -> dict[str, Any]:
    return {
        "record_type": record_type,
        "path": str(path),
        "task_id": "",
        "model_id": "",
        "sae_id": "",
        "seed": "",
        "smoke": "",
        "valid": False,
        "error": error,
    }
