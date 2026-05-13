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
    "ranking_method",
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
    sweeps = _load_family(config_root, "sweeps")
    advanced = _load_family(config_root, "advanced")
    ranking_methods = sweeps.get("ranking_control", {}).get(
        "methods",
        ["probe_weight", "validation_contribution", "hybrid"],
    )

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
                    base_rows = [
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
                    ]
                    rows.extend(base_rows)
                    task_slug = _task_slug(task_id)
                    for ranking_method in ranking_methods:
                        control_root = (
                            "${ARTIFACT_ROOT}/analysis/ranking_controls/"
                            f"{sae_id}/{task_slug}/{ranking_method}"
                        )
                        rows.extend(
                            [
                                _row(
                                    stage="feature_ranking",
                                    experiment_id=experiment_id,
                                    task=task,
                                    model_id=model_id,
                                    sae_id=sae_id,
                                    command="rank-features",
                                    artifact_dir=f"{control_root}/ranking",
                                    ranking_method=ranking_method,
                                ),
                                _row(
                                    stage="subset_usage",
                                    experiment_id=experiment_id,
                                    task=task,
                                    model_id=model_id,
                                    sae_id=sae_id,
                                    command="compute-subset-usage",
                                    artifact_dir=f"{control_root}/subset_usage",
                                    ranking_method=ranking_method,
                                ),
                                _row(
                                    stage="feature_ablation",
                                    experiment_id=experiment_id,
                                    task=task,
                                    model_id=model_id,
                                    sae_id=sae_id,
                                    command="ablate-features",
                                    artifact_dir=f"{control_root}/ablation",
                                    ranking_method=ranking_method,
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
                        "ranking_method": "",
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

    rows.extend(_layer_sweep_rows(models=models, saes=saes, sweeps=sweeps))
    rows.extend(_advanced_rows(models=models, saes=saes, tasks=tasks, advanced=advanced))

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
    ranking_method: str = "",
) -> dict[str, str]:
    return {
        "stage": stage,
        "experiment_id": experiment_id,
        "task_id": str(task["id"]),
        "task_type": str(task["type"]),
        "model_id": model_id,
        "sae_id": sae_id,
        "ranking_method": ranking_method,
        "command": command,
        "artifact_dir": artifact_dir,
    }


def _layer_sweep_rows(
    *,
    models: dict[str, dict[str, Any]],
    saes: dict[str, dict[str, Any]],
    sweeps: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    sweep = sweeps.get("layer_sweep")
    if not sweep:
        return []
    dataset_split = str(sweep["dataset_split"])
    rows: list[dict[str, str]] = []
    for group in sweep["groups"]:
        if not group.get("enabled_in_plan", False):
            continue
        group_id = str(group["id"])
        for pair in group.get("model_sae_pairs", []):
            model_id = str(pair["model"])
            sae_id = str(pair["sae"])
            _require_known(models, model_id, "model")
            sae = _require_known(saes, sae_id, "sae")
            if sae["model_id"] != model_id:
                raise ConfigError(f"SAE {sae_id} does not belong to model {model_id}")
            layer = int(sae["layer"])
            experiment_id = f"layer_sweep:{group_id}"
            task = {"id": dataset_split, "type": "feature_usage"}
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
                            f"{model_id}/{dataset_split}/l{layer}"
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
                            f"{sae_id}/{dataset_split}"
                        ),
                    ),
                    _row(
                        stage="availability",
                        experiment_id=experiment_id,
                        task=task,
                        model_id=model_id,
                        sae_id=sae_id,
                        command="compute-usage",
                        artifact_dir=(
                            "${ARTIFACT_ROOT}/analysis/layer_sweeps/"
                            f"{group_id}/{sae_id}/availability"
                        ),
                    ),
                ]
            )
    return rows


def _advanced_rows(
    *,
    models: dict[str, dict[str, Any]],
    saes: dict[str, dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    advanced: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for advanced_id, config in sorted(advanced.items()):
        if config["mode"] != "native_subspace_ablation":
            raise ConfigError(f"unsupported advanced mode in planner: {config['mode']}")
        task = _require_known(tasks, str(config["task_id"]), "task")
        for run in config["runs"]:
            model_id = str(run["model"])
            sae_id = str(run["sae"])
            _require_known(models, model_id, "model")
            sae = _require_known(saes, sae_id, "sae")
            if sae["model_id"] != model_id:
                raise ConfigError(f"SAE {sae_id} does not belong to model {model_id}")
            for top_k in run["top_ks"]:
                rows.append(
                    _row(
                        stage="native_subspace_ablation",
                        experiment_id=advanced_id,
                        task=task,
                        model_id=model_id,
                        sae_id=sae_id,
                        command="ablate-native-subspace",
                        artifact_dir=(
                            "${ARTIFACT_ROOT}/analysis/native_subspace_ablation/"
                            f"{advanced_id}/{run['id']}/top_{top_k}"
                        ),
                        ranking_method=str(run["ranking_source"]),
                    )
                )
    return rows


def _task_slug(task_id: str) -> str:
    if task_id == "imagenet_1k":
        return "imagenet_val"
    if task_id == "nyuv2_depth":
        return "nyuv2_val"
    if task_id == "ade20k_segmentation":
        return "ade20k_val"
    if task_id == "clevr_count":
        return "clevr_count_val"
    return f"{task_id}_val"


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
