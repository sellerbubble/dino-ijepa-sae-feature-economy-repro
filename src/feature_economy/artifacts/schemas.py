"""Lightweight artifact schema validation.

These helpers intentionally validate only the public contract that downstream
table/figure scripts rely on. They do not try to replace a full JSON Schema
system yet; keeping the first skeleton small makes the contract easier to
review before runner ports begin.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class SchemaError(ValueError):
    """Raised when an artifact does not satisfy the public contract."""


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{name} must be a mapping")
    return value


def _require_keys(record: Mapping[str, Any], keys: Sequence[str], name: str) -> None:
    missing = [key for key in keys if key not in record]
    if missing:
        raise SchemaError(f"{name} missing required keys: {', '.join(missing)}")


def _require_metrics(record: Mapping[str, Any], name: str) -> None:
    metrics = _require_mapping(record.get("metrics"), f"{name}.metrics")
    if not metrics:
        raise SchemaError(f"{name}.metrics must not be empty")
    for key, value in metrics.items():
        if not isinstance(key, str):
            raise SchemaError(f"{name}.metrics keys must be strings")
        if not isinstance(value, (int, float)):
            raise SchemaError(f"{name}.metrics[{key!r}] must be numeric")


def validate_run_manifest(record: Mapping[str, Any]) -> None:
    """Validate the minimal provenance manifest written by public commands."""

    record = _require_mapping(record, "run_manifest")
    _require_keys(
        record,
        ["run_id", "command", "git_commit", "config_files", "inputs", "outputs"],
        "run_manifest",
    )
    if not isinstance(record["config_files"], list):
        raise SchemaError("run_manifest.config_files must be a list")
    _require_mapping(record["inputs"], "run_manifest.inputs")
    _require_mapping(record["outputs"], "run_manifest.outputs")


def validate_array_contract_summary(record: Mapping[str, Any]) -> None:
    """Validate `.npz` array contract validation reports."""

    record = _require_mapping(record, "array_contract_validation")
    _require_keys(
        record,
        ["record_type", "path", "kind", "arrays", "valid"],
        "array_contract_validation",
    )
    if record["record_type"] != "array_contract_validation":
        raise SchemaError(
            "array_contract_validation.record_type must be array_contract_validation"
        )
    if record["kind"] not in {
        "features",
        "codes",
        "probe_logits",
        "targets",
        "contribution_scores",
    }:
        raise SchemaError("array_contract_validation.kind is unsupported")
    if record["valid"] is not True:
        raise SchemaError("array_contract_validation.valid must be true")
    arrays = _require_mapping(record["arrays"], "array_contract_validation.arrays")
    if not arrays:
        raise SchemaError("array_contract_validation.arrays must not be empty")
    for key, shape in arrays.items():
        if not isinstance(key, str):
            raise SchemaError("array_contract_validation array keys must be strings")
        if not isinstance(shape, list):
            raise SchemaError(f"array_contract_validation.arrays[{key!r}] must be a list")
        for dim in shape:
            if not isinstance(dim, int) or dim < 0:
                raise SchemaError(
                    f"array_contract_validation.arrays[{key!r}] dims must be non-negative"
                )


def validate_reproduction_run_plan(record: Mapping[str, Any]) -> None:
    """Validate a public reproduction run-plan record."""

    record = _require_mapping(record, "reproduction_run_plan")
    _require_keys(
        record,
        ["record_type", "num_rows", "rows"],
        "reproduction_run_plan",
    )
    if record["record_type"] != "reproduction_run_plan":
        raise SchemaError("reproduction_run_plan.record_type must be reproduction_run_plan")
    if not isinstance(record["num_rows"], int) or record["num_rows"] < 0:
        raise SchemaError("reproduction_run_plan.num_rows must be non-negative")
    rows = record["rows"]
    if not isinstance(rows, list):
        raise SchemaError("reproduction_run_plan.rows must be a list")
    if len(rows) != record["num_rows"]:
        raise SchemaError("reproduction_run_plan.num_rows must match rows length")
    for index, row in enumerate(rows):
        row = _require_mapping(row, f"reproduction_run_plan.rows[{index}]")
        _require_keys(
            row,
            ["stage", "task_id", "model_id", "command", "artifact_dir"],
            f"reproduction_run_plan.rows[{index}]",
        )
        for key in ["stage", "task_id", "model_id", "command", "artifact_dir"]:
            if not isinstance(row[key], str) or not row[key]:
                raise SchemaError(
                    f"reproduction_run_plan.rows[{index}].{key} must be a non-empty string"
                )


def validate_artifact_bundle_check(record: Mapping[str, Any]) -> None:
    """Validate an artifact bundle completeness report."""

    record = _require_mapping(record, "artifact_bundle_check")
    _require_keys(
        record,
        [
            "record_type",
            "run_plan",
            "artifact_root",
            "num_rows",
            "complete_rows",
            "missing_rows",
            "complete",
            "rows",
        ],
        "artifact_bundle_check",
    )
    if record["record_type"] != "artifact_bundle_check":
        raise SchemaError("artifact_bundle_check.record_type must be artifact_bundle_check")
    for key in ["num_rows", "complete_rows", "missing_rows"]:
        if not isinstance(record[key], int) or record[key] < 0:
            raise SchemaError(f"artifact_bundle_check.{key} must be non-negative")
    if record["complete_rows"] + record["missing_rows"] != record["num_rows"]:
        raise SchemaError("artifact_bundle_check row counts are inconsistent")
    if not isinstance(record["complete"], bool):
        raise SchemaError("artifact_bundle_check.complete must be boolean")
    rows = record["rows"]
    if not isinstance(rows, list):
        raise SchemaError("artifact_bundle_check.rows must be a list")
    if len(rows) != record["num_rows"]:
        raise SchemaError("artifact_bundle_check.num_rows must match rows length")
    for index, row in enumerate(rows):
        row = _require_mapping(row, f"artifact_bundle_check.rows[{index}]")
        _require_keys(
            row,
            ["stage", "artifact_dir", "required_files", "missing_files", "complete"],
            f"artifact_bundle_check.rows[{index}]",
        )
        if not isinstance(row["required_files"], list):
            raise SchemaError(
                f"artifact_bundle_check.rows[{index}].required_files must be a list"
            )
        if not isinstance(row["missing_files"], list):
            raise SchemaError(
                f"artifact_bundle_check.rows[{index}].missing_files must be a list"
            )
        if not isinstance(row["complete"], bool):
            raise SchemaError(f"artifact_bundle_check.rows[{index}].complete must be boolean")


def validate_probe_summary(record: Mapping[str, Any]) -> None:
    """Validate native or SAE-code probe summaries."""

    record = _require_mapping(record, "probe_summary")
    _require_keys(
        record,
        ["record_type", "task_id", "model_id", "seed", "metrics"],
        "probe_summary",
    )
    if record["record_type"] not in {"native_probe_summary", "sae_probe_summary"}:
        raise SchemaError("probe_summary.record_type must identify native or SAE probe")
    if record["record_type"] == "sae_probe_summary" and "sae_id" not in record:
        raise SchemaError("sae_probe_summary requires sae_id")
    _require_metrics(record, "probe_summary")


def validate_feature_extraction_summary(record: Mapping[str, Any]) -> None:
    """Validate hidden-state / feature extraction summaries."""

    record = _require_mapping(record, "feature_extraction_summary")
    _require_keys(
        record,
        [
            "record_type",
            "model_id",
            "layer",
            "input_manifest",
            "output_npz",
            "num_examples",
            "feature_shape",
            "token_format",
            "transform",
        ],
        "feature_extraction_summary",
    )
    if record["record_type"] != "feature_extraction_summary":
        raise SchemaError(
            "feature_extraction_summary.record_type must be feature_extraction_summary"
        )
    if not isinstance(record["layer"], int) or record["layer"] < 0:
        raise SchemaError("feature_extraction_summary.layer must be a non-negative integer")
    if not isinstance(record["num_examples"], int) or record["num_examples"] <= 0:
        raise SchemaError("feature_extraction_summary.num_examples must be positive")
    shape = record["feature_shape"]
    if not isinstance(shape, list) or not shape:
        raise SchemaError("feature_extraction_summary.feature_shape must be a non-empty list")
    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            raise SchemaError("feature_extraction_summary.feature_shape dims must be positive")
    _require_mapping(record["transform"], "feature_extraction_summary.transform")


def validate_sae_code_summary(record: Mapping[str, Any]) -> None:
    """Validate SAE code extraction summaries."""

    record = _require_mapping(record, "sae_code_summary")
    _require_keys(
        record,
        [
            "record_type",
            "model_id",
            "sae_id",
            "input_features",
            "output_npz",
            "num_examples",
            "code_shape",
            "normalize_activations",
        ],
        "sae_code_summary",
    )
    if record["record_type"] != "sae_code_summary":
        raise SchemaError("sae_code_summary.record_type must be sae_code_summary")
    if not isinstance(record["num_examples"], int) or record["num_examples"] <= 0:
        raise SchemaError("sae_code_summary.num_examples must be positive")
    shape = record["code_shape"]
    if not isinstance(shape, list) or not shape:
        raise SchemaError("sae_code_summary.code_shape must be a non-empty list")
    for dim in shape:
        if not isinstance(dim, int) or dim <= 0:
            raise SchemaError("sae_code_summary.code_shape dims must be positive")
    if record["normalize_activations"] not in {"none", "layer_norm"}:
        raise SchemaError("sae_code_summary.normalize_activations is unsupported")


def validate_feature_ranking(record: Mapping[str, Any]) -> None:
    """Validate a task feature ranking artifact."""

    record = _require_mapping(record, "feature_ranking")
    _require_keys(record, ["record_type", "ranking_method", "rows"], "feature_ranking")
    if record["record_type"] != "task_feature_ranking":
        raise SchemaError("feature_ranking.record_type must be task_feature_ranking")
    if record["ranking_method"] not in {
        "probe_weight",
        "validation_contribution",
        "hybrid",
    }:
        raise SchemaError("feature_ranking.ranking_method is unknown")
    rows = record["rows"]
    if not isinstance(rows, list) or not rows:
        raise SchemaError("feature_ranking.rows must be a non-empty list")
    required = [
        "feature_id",
        "task_rank",
        "ranking_score",
        "probe_weight_score",
        "validation_contribution_score",
        "mean_activation",
        "mean_positive_activation",
    ]
    for index, row in enumerate(rows):
        row = _require_mapping(row, f"feature_ranking.rows[{index}]")
        _require_keys(row, required, f"feature_ranking.rows[{index}]")


def validate_contribution_scores_summary(record: Mapping[str, Any]) -> None:
    """Validate per-feature validation-contribution score summaries."""

    record = _require_mapping(record, "contribution_scores_summary")
    _require_keys(
        record,
        [
            "record_type",
            "task_id",
            "model_id",
            "sae_id",
            "task_type",
            "primary_metric",
            "score_key",
            "output_npz",
            "num_features",
            "scored_features",
            "baseline_metrics",
            "score_quantiles",
            "top_features",
        ],
        "contribution_scores_summary",
    )
    if record["record_type"] != "contribution_scores_summary":
        raise SchemaError(
            "contribution_scores_summary.record_type must be contribution_scores_summary"
        )
    if record["task_type"] not in {
        "classification",
        "count_classification",
        "dense_depth",
        "dense_segmentation",
    }:
        raise SchemaError("contribution_scores_summary.task_type is unsupported")
    for key in ["primary_metric", "score_key", "output_npz"]:
        if not isinstance(record[key], str) or not record[key]:
            raise SchemaError(f"contribution_scores_summary.{key} must be a non-empty string")
    for key in ["num_features", "scored_features"]:
        if not isinstance(record[key], int) or record[key] <= 0:
            raise SchemaError(f"contribution_scores_summary.{key} must be positive")
    if record["scored_features"] > record["num_features"]:
        raise SchemaError("contribution_scores_summary.scored_features exceeds num_features")
    _require_mapping(
        record["baseline_metrics"],
        "contribution_scores_summary.baseline_metrics",
    )
    _require_mapping(
        record["score_quantiles"],
        "contribution_scores_summary.score_quantiles",
    )
    top_features = record["top_features"]
    if not isinstance(top_features, list):
        raise SchemaError("contribution_scores_summary.top_features must be a list")
    for index, row in enumerate(top_features):
        row = _require_mapping(row, f"contribution_scores_summary.top_features[{index}]")
        _require_keys(
            row,
            ["feature_id", "score"],
            f"contribution_scores_summary.top_features[{index}]",
        )
        if not isinstance(row["feature_id"], int) or row["feature_id"] < 0:
            raise SchemaError(
                f"contribution_scores_summary.top_features[{index}].feature_id "
                "must be non-negative"
            )
        if not isinstance(row["score"], (int, float)):
            raise SchemaError(
                f"contribution_scores_summary.top_features[{index}].score must be numeric"
            )


def validate_availability_summary(record: Mapping[str, Any]) -> None:
    """Validate feature availability / utilization summaries."""

    record = _require_mapping(record, "availability_summary")
    _require_keys(
        record,
        [
            "record_type",
            "model_id",
            "sae_id",
            "dataset_id",
            "split",
            "usage_unit",
            "total_features",
            "fired_features",
            "dead_features",
            "buckets",
        ],
        "availability_summary",
    )
    if record["record_type"] != "availability_summary":
        raise SchemaError("availability_summary.record_type must be availability_summary")
    if record["usage_unit"] not in {"fired_count", "usage_rate"}:
        raise SchemaError("availability_summary.usage_unit must be fired_count or usage_rate")
    for key in ["total_features", "fired_features", "dead_features"]:
        if not isinstance(record[key], int) or record[key] < 0:
            raise SchemaError(f"availability_summary.{key} must be a non-negative integer")
    if record["fired_features"] > record["total_features"]:
        raise SchemaError("availability_summary.fired_features exceeds total_features")
    if record["dead_features"] > record["total_features"]:
        raise SchemaError("availability_summary.dead_features exceeds total_features")
    buckets = _require_mapping(record["buckets"], "availability_summary.buckets")
    for bucket_name, bucket_count in buckets.items():
        if not isinstance(bucket_name, str):
            raise SchemaError("availability_summary bucket names must be strings")
        if not isinstance(bucket_count, int) or bucket_count < 0:
            raise SchemaError(f"availability_summary bucket {bucket_name!r} must be non-negative")
    quantiles = record.get("active_fired_count_quantiles")
    if quantiles is not None:
        _require_mapping(quantiles, "availability_summary.active_fired_count_quantiles")


def validate_subset_usage_summary(record: Mapping[str, Any]) -> None:
    """Validate task-selected versus random subset usage summaries."""

    record = _require_mapping(record, "subset_usage_summary")
    _require_keys(
        record,
        [
            "record_type",
            "task_id",
            "model_id",
            "sae_id",
            "ranking_method",
            "usage_unit",
            "subsets",
        ],
        "subset_usage_summary",
    )
    if record["record_type"] != "subset_usage_summary":
        raise SchemaError("subset_usage_summary.record_type must be subset_usage_summary")
    if record["ranking_method"] not in {
        "probe_weight",
        "validation_contribution",
        "hybrid",
    }:
        raise SchemaError("subset_usage_summary.ranking_method is unknown")
    if record["usage_unit"] not in {"fired_count", "usage_rate"}:
        raise SchemaError("subset_usage_summary.usage_unit must be fired_count or usage_rate")
    subsets = record["subsets"]
    if not isinstance(subsets, list) or not subsets:
        raise SchemaError("subset_usage_summary.subsets must be a non-empty list")
    for index, subset in enumerate(subsets):
        subset = _require_mapping(subset, f"subset_usage_summary.subsets[{index}]")
        _require_keys(
            subset,
            ["name", "selection", "num_features", "median_usage", "high_usage_fraction"],
            f"subset_usage_summary.subsets[{index}]",
        )
        if subset["selection"] not in {"task_selected", "matched_random", "random"}:
            raise SchemaError(
                f"subset_usage_summary.subsets[{index}].selection is unsupported"
            )
        if not isinstance(subset["num_features"], int) or subset["num_features"] <= 0:
            raise SchemaError(
                f"subset_usage_summary.subsets[{index}].num_features must be positive"
            )
        for numeric_key in ["median_usage", "high_usage_fraction"]:
            if not isinstance(subset[numeric_key], (int, float)):
                raise SchemaError(
                    f"subset_usage_summary.subsets[{index}].{numeric_key} must be numeric"
                )


def validate_ablation_summary(record: Mapping[str, Any]) -> None:
    """Validate SAE feature ablation summaries."""

    record = _require_mapping(record, "ablation_summary")
    _require_keys(
        record,
        [
            "record_type",
            "task_id",
            "model_id",
            "baseline_metrics",
            "ablations",
            "random_controls",
        ],
        "ablation_summary",
    )
    if record["record_type"] != "feature_ablation_summary":
        raise SchemaError("ablation_summary.record_type must be feature_ablation_summary")
    _require_mapping(record["baseline_metrics"], "ablation_summary.baseline_metrics")
    if not isinstance(record["ablations"], list) or not record["ablations"]:
        raise SchemaError("ablation_summary.ablations must be a non-empty list")
    if not isinstance(record["random_controls"], list):
        raise SchemaError("ablation_summary.random_controls must be a list")
    for index, ablation in enumerate(record["ablations"]):
        ablation = _require_mapping(ablation, f"ablation_summary.ablations[{index}]")
        _require_keys(
            ablation,
            ["subset", "num_features", "metrics", "metric_delta"],
            f"ablation_summary.ablations[{index}]",
        )
        _require_mapping(ablation["metrics"], f"ablation_summary.ablations[{index}].metrics")
        _require_mapping(
            ablation["metric_delta"],
            f"ablation_summary.ablations[{index}].metric_delta",
        )
