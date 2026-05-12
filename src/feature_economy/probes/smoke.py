"""Config-driven smoke artifact writer.

This module does not run a model. It exercises the public experiment config
shape and writes valid probe-summary artifacts with deterministic placeholder
metrics. The point is to validate the future runner contract before migrating
full benchmark code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feature_economy.artifacts import (
    current_git_commit,
    validate_probe_summary,
    validate_run_manifest,
)
from feature_economy.configs import load_yaml, validate_all_configs


def write_smoke_probe_artifacts(
    config_root: str | Path,
    experiment_id: str,
    output_dir: str | Path,
) -> list[Path]:
    """Write smoke probe summaries for a native or SAE probe experiment config."""

    config_root = Path(config_root)
    output_dir = Path(output_dir)
    validate_all_configs(config_root)
    models = _load_family(config_root, "models")
    saes = _load_family(config_root, "saes")
    tasks = _load_family(config_root, "tasks")
    experiments = _load_family(config_root, "experiments")
    if experiment_id not in experiments:
        raise ValueError(f"unknown experiment id: {experiment_id}")
    experiment = experiments[experiment_id]
    mode = experiment["mode"]
    if mode not in {"native_probe", "sae_probe"}:
        raise ValueError(f"smoke probe only supports native_probe/sae_probe, got {mode}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if mode == "native_probe":
        for model_id in experiment["matrix"]["models"]:
            _require_known(models, model_id, "model")
            for task_id in experiment["matrix"]["tasks"]:
                task = _require_known(tasks, task_id, "task")
                record = _build_probe_record(
                    record_type="native_probe_summary",
                    experiment=experiment,
                    task=task,
                    model_id=model_id,
                    sae_id=None,
                )
                written.append(_write_summary(output_dir, record))
    else:
        for pair in experiment["matrix"]["model_sae_pairs"]:
            model_id = pair["model"]
            sae_id = pair["sae"]
            _require_known(models, model_id, "model")
            _require_known(saes, sae_id, "sae")
            for task_id in experiment["matrix"]["tasks"]:
                task = _require_known(tasks, task_id, "task")
                record = _build_probe_record(
                    record_type="sae_probe_summary",
                    experiment=experiment,
                    task=task,
                    model_id=model_id,
                    sae_id=sae_id,
                )
                written.append(_write_summary(output_dir, record))

    manifest = {
        "run_id": f"smoke_{experiment_id}",
        "command": (
            "feature-economy smoke-probe "
            f"--config-root {config_root} --experiment-id {experiment_id} "
            f"--output-dir {output_dir}"
        ),
        "git_commit": current_git_commit(),
        "config_files": [str(path) for path in sorted(config_root.rglob("*.yaml"))],
        "inputs": {"experiment_id": experiment_id, "mode": mode},
        "outputs": {"summaries": [str(path) for path in written]},
        "smoke": True,
    }
    validate_run_manifest(manifest)
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    written.append(manifest_path)
    return written


def _load_family(config_root: Path, family: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((config_root / family).glob("*.yaml")):
        record = dict(load_yaml(path))
        records[record["id"]] = record
    return records


def _require_known(records: dict[str, dict[str, Any]], record_id: str, name: str) -> dict[str, Any]:
    if record_id not in records:
        raise ValueError(f"unknown {name} id: {record_id}")
    return records[record_id]


def _build_probe_record(
    *,
    record_type: str,
    experiment: dict[str, Any],
    task: dict[str, Any],
    model_id: str,
    sae_id: str | None,
) -> dict[str, Any]:
    metrics = {
        metric: _placeholder_metric_value(task["id"], model_id, metric)
        for metric in task["metrics"]["report"]
    }
    record: dict[str, Any] = {
        "record_type": record_type,
        "task_id": task["id"],
        "model_id": model_id,
        "seed": experiment["seed"],
        "metrics": metrics,
        "checkpoint": "smoke_probe.pt",
        "smoke": True,
    }
    if sae_id is not None:
        record["sae_id"] = sae_id
    validate_probe_summary(record)
    return record


def _placeholder_metric_value(task_id: str, model_id: str, metric: str) -> float:
    # Deterministic non-scientific value; never use as an experimental result.
    raw = sum(ord(ch) for ch in f"{task_id}:{model_id}:{metric}")
    return round((raw % 1000) / 1000.0, 6)


def _write_summary(output_dir: Path, record: dict[str, Any]) -> Path:
    sae_part = f"_{record['sae_id']}" if "sae_id" in record else ""
    filename = f"{record['task_id']}_{record['model_id']}{sae_part}_{record['record_type']}.json"
    path = output_dir / filename
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path
