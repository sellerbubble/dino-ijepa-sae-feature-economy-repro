"""Smoke writer for AAA Feature Economy analysis artifacts.

This module writes deterministic placeholder artifacts for availability, access,
ranking, and allocation. It is not an experiment runner. It makes the public
artifact contract executable end-to-end before full analysis code is ported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from feature_economy.artifacts import (
    current_git_commit,
    validate_ablation_summary,
    validate_availability_summary,
    validate_feature_ranking,
    validate_run_manifest,
    validate_subset_usage_summary,
)
from feature_economy.configs import load_yaml, validate_all_configs


def write_smoke_analysis_artifacts(
    config_root: str | Path,
    output_dir: str | Path,
    include: set[str] | None = None,
) -> list[Path]:
    """Write smoke artifacts for the AAA analysis chain."""

    config_root = Path(config_root)
    output_dir = Path(output_dir)
    validate_all_configs(config_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    include = include or {"availability", "ranking", "subset_usage", "ablation"}

    saes = _load_family(config_root, "saes")
    tasks = _load_family(config_root, "tasks")
    experiments = _load_family(config_root, "experiments")
    sae_experiment = experiments["paper_v0_sae"]
    availability_experiment = experiments["paper_v0_availability"]
    written: list[Path] = []

    for pair in availability_experiment["model_sae_pairs"]:
        if "availability" not in include:
            continue
        model_id = pair["model"]
        sae_id = pair["sae"]
        sae = saes[sae_id]
        record = _availability_record(
            model_id=model_id,
            sae_id=sae_id,
            dataset_id=availability_experiment["dataset_split"],
            layer=sae["layer"],
        )
        written.append(_write_json(output_dir, f"availability_{model_id}_{sae_id}.json", record))

    for pair in sae_experiment["matrix"]["model_sae_pairs"]:
        model_id = pair["model"]
        sae_id = pair["sae"]
        for task_id in sae_experiment["matrix"]["tasks"]:
            task = tasks[task_id]
            prefix = f"{task_id}_{model_id}_{sae_id}"
            if "ranking" in include:
                ranking = _ranking_record(task_id, model_id, sae_id)
                written.append(_write_json(output_dir, f"{prefix}_ranking.json", ranking))
            if "subset_usage" in include:
                subset = _subset_usage_record(task_id, model_id, sae_id)
                written.append(_write_json(output_dir, f"{prefix}_subset_usage.json", subset))
            if "ablation" in include:
                ablation = _ablation_record(task, model_id, sae_id)
                written.append(_write_json(output_dir, f"{prefix}_ablation.json", ablation))

    manifest = {
        "run_id": "smoke_analysis",
        "command": f"feature-economy smoke-analysis --config-root {config_root} --output-dir {output_dir}",
        "git_commit": current_git_commit(),
        "config_files": [str(path) for path in sorted(config_root.rglob("*.yaml"))],
        "inputs": {"config_root": str(config_root), "include": sorted(include)},
        "outputs": {"artifacts": [str(path) for path in written]},
        "smoke": True,
    }
    validate_run_manifest(manifest)
    written.append(_write_json(output_dir, "run_manifest.json", manifest))
    return written


def _load_family(config_root: Path, family: str) -> dict[str, dict[str, Any]]:
    return {
        record["id"]: record
        for record in (dict(load_yaml(path)) for path in sorted((config_root / family).glob("*.yaml")))
    }


def _availability_record(*, model_id: str, sae_id: str, dataset_id: str, layer: int) -> dict[str, Any]:
    total_features = 4096
    dead_features = 64 + layer
    fired_features = total_features - dead_features
    record = {
        "record_type": "availability_summary",
        "model_id": model_id,
        "sae_id": sae_id,
        "dataset_id": dataset_id,
        "split": "smoke",
        "usage_unit": "fired_count",
        "total_features": total_features,
        "fired_features": fired_features,
        "dead_features": dead_features,
        "buckets": {
            "dead": dead_features,
            "rare": 512,
            "medium": 1024,
            "frequent_or_above": total_features - dead_features - 1536,
        },
        "active_fired_count_quantiles": {
            "p10": 10 + layer,
            "p50": 100 + layer,
            "p90": 1000 + layer,
        },
        "smoke": True,
    }
    validate_availability_summary(record)
    return record


def _ranking_record(task_id: str, model_id: str, sae_id: str) -> dict[str, Any]:
    rows = []
    for rank in range(1, 6):
        rows.append(
            {
                "feature_id": rank * 10,
                "task_rank": rank,
                "ranking_score": round(1.0 / rank, 6),
                "probe_weight_score": round(0.8 / rank, 6),
                "validation_contribution_score": round(0.9 / rank, 6),
                "mean_activation": round(0.01 * rank, 6),
                "mean_positive_activation": round(0.02 * rank, 6),
            }
        )
    record = {
        "record_type": "task_feature_ranking",
        "task_id": task_id,
        "model_id": model_id,
        "sae_id": sae_id,
        "ranking_method": "hybrid",
        "rows": rows,
        "smoke": True,
    }
    validate_feature_ranking(record)
    return record


def _subset_usage_record(task_id: str, model_id: str, sae_id: str) -> dict[str, Any]:
    record = {
        "record_type": "subset_usage_summary",
        "task_id": task_id,
        "model_id": model_id,
        "sae_id": sae_id,
        "ranking_method": "hybrid",
        "usage_unit": "fired_count",
        "subsets": [
            {
                "name": "top_100",
                "selection": "task_selected",
                "num_features": 100,
                "median_usage": 42.0,
                "high_usage_fraction": 0.12,
            },
            {
                "name": "matched_random_100_seed0",
                "selection": "matched_random",
                "num_features": 100,
                "median_usage": 7.0,
                "high_usage_fraction": 0.02,
            },
        ],
        "smoke": True,
    }
    validate_subset_usage_summary(record)
    return record


def _ablation_record(task: dict[str, Any], model_id: str, sae_id: str) -> dict[str, Any]:
    primary_metric = task["metrics"]["primary"]
    baseline_value = _placeholder_metric_value(task["id"], model_id, primary_metric)
    record = {
        "record_type": "feature_ablation_summary",
        "task_id": task["id"],
        "model_id": model_id,
        "sae_id": sae_id,
        "baseline_metrics": {primary_metric: baseline_value},
        "ablations": [
            {
                "subset": "frequent_or_above",
                "num_features": 8,
                "metrics": {primary_metric: round(baseline_value + 0.1, 6)},
                "metric_delta": {primary_metric: 0.1},
            }
        ],
        "random_controls": [
            {
                "subset": "matched_random",
                "num_features": 8,
                "metric_delta": {primary_metric: 0.01},
            }
        ],
        "smoke": True,
    }
    validate_ablation_summary(record)
    return record


def _placeholder_metric_value(task_id: str, model_id: str, metric: str) -> float:
    raw = sum(ord(ch) for ch in f"{task_id}:{model_id}:{metric}")
    return round((raw % 1000) / 1000.0, 6)


def _write_json(output_dir: Path, filename: str, record: dict[str, Any]) -> Path:
    path = output_dir / filename
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path
