"""Expand public experiment configs into a machine-readable run plan."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .loader import ConfigError, load_yaml, validate_all_configs


PLAN_COLUMNS = [
    "stage",
    "experiment_id",
    "task_id",
    "task_type",
    "model_id",
    "sae_id",
    "command",
    "artifact_dir",
]


def build_reproduction_plan(config_root: str | Path) -> dict[str, Any]:
    """Build the paper-chain run matrix from portable public configs."""

    config_root = Path(config_root)
    validate_all_configs(config_root)
    models = _load_family(config_root, "models")
    saes = _load_family(config_root, "saes")
    tasks = _load_family(config_root, "tasks")
    experiments = _load_family(config_root, "experiments")

    rows: list[dict[str, str]] = []
    for experiment_id, experiment in sorted(experiments.items()):
        mode = experiment["mode"]
        if mode == "native_probe":
            for model_id in experiment["matrix"]["models"]:
                _require_known(models, model_id, "model")
                for task_id in experiment["matrix"]["tasks"]:
                    task = _require_known(tasks, task_id, "task")
                    rows.append(
                        _row(
                            stage="native_probe",
                            experiment_id=experiment_id,
                            task=task,
                            model_id=model_id,
                            sae_id="",
                            command="probe-native",
                            artifact_dir=(
                                "${ARTIFACT_ROOT}/probes/native/"
                                f"{model_id}/{task_id}/val"
                            ),
                        )
                    )
        elif mode == "sae_probe":
            for pair in experiment["matrix"]["model_sae_pairs"]:
                model_id = pair["model"]
                sae_id = pair["sae"]
                _require_known(models, model_id, "model")
                sae = _require_known(saes, sae_id, "sae")
                if sae["model_id"] != model_id:
                    raise ConfigError(f"SAE {sae_id} does not belong to model {model_id}")
                for task_id in experiment["matrix"]["tasks"]:
                    task = _require_known(tasks, task_id, "task")
                    rows.extend(
                        [
                            _row(
                                stage="feature_extraction",
                                experiment_id=experiment_id,
                                task=task,
                                model_id=model_id,
                                sae_id="",
                                command="extract-features",
                                artifact_dir=(
                                    "${ARTIFACT_ROOT}/features/"
                                    f"{model_id}/{task_id}/l{sae['layer']}"
                                ),
                            ),
                            _row(
                                stage="sae_code_extraction",
                                experiment_id=experiment_id,
                                task=task,
                                model_id=model_id,
                                sae_id=sae_id,
                                command="extract-sae-codes",
                                artifact_dir=(
                                    "${ARTIFACT_ROOT}/codes/"
                                    f"{sae_id}/{task_id}/val"
                                ),
                            ),
                            _row(
                                stage="sae_probe",
                                experiment_id=experiment_id,
                                task=task,
                                model_id=model_id,
                                sae_id=sae_id,
                                command="probe-sae",
                                artifact_dir=(
                                    "${ARTIFACT_ROOT}/probes/sae/"
                                    f"{sae_id}/{task_id}/val"
                                ),
                            ),
                            _row(
                                stage="feature_ranking",
                                experiment_id=experiment_id,
                                task=task,
                                model_id=model_id,
                                sae_id=sae_id,
                                command="rank-features",
                                artifact_dir=(
                                    "${ARTIFACT_ROOT}/analysis/ranking/"
                                    f"{sae_id}/{task_id}/val"
                                ),
                            ),
                            _row(
                                stage="contribution_scores",
                                experiment_id=experiment_id,
                                task=task,
                                model_id=model_id,
                                sae_id=sae_id,
                                command="compute-contributions",
                                artifact_dir=(
                                    "${ARTIFACT_ROOT}/analysis/contribution/"
                                    f"{sae_id}/{task_id}/val"
                                ),
                            ),
                            _row(
                                stage="subset_usage",
                                experiment_id=experiment_id,
                                task=task,
                                model_id=model_id,
                                sae_id=sae_id,
                                command="compute-subset-usage",
                                artifact_dir=(
                                    "${ARTIFACT_ROOT}/analysis/subset_usage/"
                                    f"{sae_id}/{task_id}/val"
                                ),
                            ),
                            _row(
                                stage="feature_ablation",
                                experiment_id=experiment_id,
                                task=task,
                                model_id=model_id,
                                sae_id=sae_id,
                                command="ablate-features",
                                artifact_dir=(
                                    "${ARTIFACT_ROOT}/analysis/ablation/"
                                    f"{sae_id}/{task_id}/val"
                                ),
                            ),
                        ]
                    )
        elif mode == "compute_usage":
            for pair in experiment["model_sae_pairs"]:
                model_id = pair["model"]
                sae_id = pair["sae"]
                _require_known(models, model_id, "model")
                _require_known(saes, sae_id, "sae")
                rows.append(
                    {
                        "stage": "availability",
                        "experiment_id": experiment_id,
                        "task_id": experiment["dataset_split"],
                        "task_type": "feature_usage",
                        "model_id": model_id,
                        "sae_id": sae_id,
                        "command": "compute-usage",
                        "artifact_dir": (
                            "${ARTIFACT_ROOT}/analysis/availability/"
                            f"{sae_id}/{experiment['dataset_split']}"
                        ),
                    }
                )
        elif mode == "sae_feature_ablation":
            # Concrete feature-ablation rows are emitted from the SAE-probe matrix above.
            continue
        else:
            raise ConfigError(f"unsupported experiment mode in planner: {mode}")

    return {
        "record_type": "reproduction_run_plan",
        "config_root": str(config_root),
        "num_rows": len(rows),
        "stages": sorted({row["stage"] for row in rows}),
        "rows": rows,
    }


def write_reproduction_plan(
    *,
    config_root: str | Path,
    output_json: str | Path,
    output_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Write a JSON/CSV run plan and return the in-memory record."""

    plan = build_reproduction_plan(config_root)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if output_csv is not None:
        _write_plan_csv(plan, output_csv)
    return plan


def _load_family(config_root: Path, family: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((config_root / family).glob("*.yaml")):
        record = dict(load_yaml(path))
        records[record["id"]] = record
    return records


def _row(
    *,
    stage: str,
    experiment_id: str,
    task: dict[str, Any],
    model_id: str,
    sae_id: str,
    command: str,
    artifact_dir: str,
) -> dict[str, str]:
    return {
        "stage": stage,
        "experiment_id": experiment_id,
        "task_id": str(task["id"]),
        "task_type": str(task["type"]),
        "model_id": model_id,
        "sae_id": sae_id,
        "command": command,
        "artifact_dir": artifact_dir,
    }


def _require_known(records: dict[str, dict[str, Any]], record_id: str, name: str) -> dict[str, Any]:
    try:
        return records[record_id]
    except KeyError as exc:
        raise ConfigError(f"unknown {name} id in experiment matrix: {record_id}") from exc


def _write_plan_csv(plan: dict[str, Any], output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_COLUMNS)
        writer.writeheader()
        for row in plan["rows"]:
            writer.writerow({column: row.get(column, "") for column in PLAN_COLUMNS})
