"""Config loading and validation for the public reproduction skeleton.

The validator is deliberately lightweight. Its job is to catch broken public
interfaces early: missing identifiers, unsupported modes, task metric mistakes,
and accidental private absolute paths. It does not check that datasets or
checkpoints exist, because public configs should remain portable across
machines.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a public reproduction config is malformed."""


PRIVATE_PATH_PATTERNS = (
    re.compile(r"^/Users/"),
    re.compile(r"^/mnt/workspace"),
    re.compile(r"^/mnt/workspace-"),
    re.compile(r"^/mnt/workspace1"),
)


def load_yaml(path: str | Path) -> Mapping[str, Any]:
    """Load a YAML mapping from disk."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return data


def validate_all_configs(config_root: str | Path) -> list[Path]:
    """Validate all YAML configs under a public config root.

    Returns the sorted list of validated config files. This is useful for tests
    and for the `feature-economy check-configs` command.
    """

    config_root = Path(config_root)
    if not config_root.exists():
        raise ConfigError(f"config root does not exist: {config_root}")
    validators: dict[str, Callable[[Mapping[str, Any]], None]] = {
        "models": validate_model_config,
        "saes": validate_sae_config,
        "tasks": validate_task_config,
        "experiments": validate_experiment_config,
    }
    validated: list[Path] = []
    for path in sorted(config_root.rglob("*.yaml")):
        try:
            family = path.relative_to(config_root).parts[0]
        except IndexError as exc:
            raise ConfigError(f"cannot infer config family for {path}") from exc
        validator = validators.get(family)
        if validator is None:
            raise ConfigError(f"unsupported config family {family!r} for {path}")
        validator(load_yaml(path))
        validated.append(path)
    if not validated:
        raise ConfigError(f"no YAML configs found under {config_root}")
    return validated


def validate_model_config(record: Mapping[str, Any]) -> None:
    _require_keys(
        record,
        [
            "id",
            "display_name",
            "provider",
            "architecture",
            "patch_size",
            "default_layers",
            "output",
            "transform",
        ],
        "model_config",
    )
    _require_identifier(record["id"], "model_config.id")
    _require_type(record["patch_size"], int, "model_config.patch_size")
    _require_mapping(record["default_layers"], "model_config.default_layers")
    _require_mapping(record["output"], "model_config.output")
    _require_mapping(record["transform"], "model_config.transform")
    _reject_private_paths(record, "model_config")


def validate_sae_config(record: Mapping[str, Any]) -> None:
    _require_keys(
        record,
        [
            "id",
            "model_id",
            "layer",
            "checkpoint",
            "topk",
            "expansion",
            "normalize_activations",
            "trained_on",
        ],
        "sae_config",
    )
    _require_identifier(record["id"], "sae_config.id")
    _require_identifier(record["model_id"], "sae_config.model_id")
    _require_type(record["layer"], int, "sae_config.layer")
    _require_type(record["topk"], int, "sae_config.topk")
    _require_type(record["expansion"], int, "sae_config.expansion")
    if record["normalize_activations"] not in {"none", "layer_norm"}:
        raise ConfigError("sae_config.normalize_activations must be none or layer_norm")
    _reject_private_paths(record, "sae_config")


def validate_task_config(record: Mapping[str, Any]) -> None:
    _require_keys(
        record,
        ["id", "type", "train_manifest", "val_manifest", "metrics", "readout"],
        "task_config",
    )
    _require_identifier(record["id"], "task_config.id")
    if record["type"] not in {
        "classification",
        "count_classification",
        "dense_depth",
        "dense_segmentation",
    }:
        raise ConfigError(f"unsupported task_config.type: {record['type']!r}")
    metrics = _require_mapping(record["metrics"], "task_config.metrics")
    _require_keys(metrics, ["primary", "report"], "task_config.metrics")
    if not isinstance(metrics["report"], list) or not metrics["report"]:
        raise ConfigError("task_config.metrics.report must be a non-empty list")
    if metrics["primary"] not in metrics["report"]:
        raise ConfigError("task_config.metrics.primary must appear in metrics.report")
    readout = _require_mapping(record["readout"], "task_config.readout")
    _require_keys(readout, ["type", "epochs", "batch_size"], "task_config.readout")
    _require_type(readout["epochs"], int, "task_config.readout.epochs")
    _require_type(readout["batch_size"], int, "task_config.readout.batch_size")
    _reject_private_paths(record, "task_config")


def validate_experiment_config(record: Mapping[str, Any]) -> None:
    _require_keys(record, ["id", "mode", "seed", "output_dir"], "experiment_config")
    _require_identifier(record["id"], "experiment_config.id")
    if record["mode"] not in {
        "native_probe",
        "sae_probe",
        "compute_usage",
        "sae_feature_ablation",
    }:
        raise ConfigError(f"unsupported experiment_config.mode: {record['mode']!r}")
    _require_type(record["seed"], int, "experiment_config.seed")
    if record["mode"] in {"native_probe", "sae_probe"}:
        _require_mapping(record.get("matrix"), "experiment_config.matrix")
    if record["mode"] == "sae_probe":
        ranking = _require_mapping(record.get("ranking"), "experiment_config.ranking")
        if ranking.get("method") not in {"probe_weight", "validation_contribution", "hybrid"}:
            raise ConfigError("experiment_config.ranking.method is unsupported")
    if record["mode"] == "compute_usage":
        _require_keys(record, ["dataset_split", "model_sae_pairs", "usage"], "experiment_config")
        usage = _require_mapping(record["usage"], "experiment_config.usage")
        if not isinstance(usage.get("buckets"), list) or not usage["buckets"]:
            raise ConfigError("experiment_config.usage.buckets must be a non-empty list")
    if record["mode"] == "sae_feature_ablation":
        if not isinstance(record.get("subsets"), list) or not record["subsets"]:
            raise ConfigError("experiment_config.subsets must be a non-empty list")
    _reject_private_paths(record, "experiment_config")


def _require_keys(record: Mapping[str, Any], keys: Sequence[str], name: str) -> None:
    missing = [key for key in keys if key not in record]
    if missing:
        raise ConfigError(f"{name} missing required keys: {', '.join(missing)}")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _require_type(value: Any, expected_type: type, name: str) -> None:
    if not isinstance(value, expected_type):
        raise ConfigError(f"{name} must be {expected_type.__name__}")


def _require_identifier(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name} must be a non-empty string")
    if not re.match(r"^[A-Za-z0-9_.-]+$", value):
        raise ConfigError(f"{name} contains unsupported characters: {value!r}")


def _reject_private_paths(value: Any, name: str) -> None:
    """Reject obvious private absolute paths while allowing env placeholders."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            _reject_private_paths(nested, f"{name}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_private_paths(nested, f"{name}[{index}]")
    elif isinstance(value, str):
        for pattern in PRIVATE_PATH_PATTERNS:
            if pattern.search(value):
                raise ConfigError(f"{name} contains private absolute path: {value}")
