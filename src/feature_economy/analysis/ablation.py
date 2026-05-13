"""Feature ablation summaries for saved SAE codes and linear probes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from feature_economy.artifacts import (
    current_git_commit,
    validate_ablation_summary,
    validate_feature_ranking,
    validate_run_manifest,
)
from feature_economy.probes.metrics import (
    accuracy,
    depth_metrics,
    segmentation_metrics,
    topk_accuracy,
)

from .ranking import _bucket_matched_random_ids, _selected_feature_ids


def ablate_linear_probe_features(
    *,
    codes_npz: str | Path,
    probe_logits_npz: str | Path,
    ranking_json: str | Path,
    task_type: str,
    output_dir: str | Path,
    top_k: int = 100,
    random_seed: int = 0,
) -> Path:
    """Zero selected SAE features and recompute a saved linear probe."""

    if task_type not in {
        "classification",
        "count_classification",
        "dense_depth",
        "dense_segmentation",
    }:
        raise ValueError(f"unsupported task_type: {task_type}")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    codes_npz = Path(codes_npz)
    probe_logits_npz = Path(probe_logits_npz)
    ranking_json = Path(ranking_json)
    codes = np.asarray(np.load(codes_npz)["codes"], dtype=np.float64)
    if codes.ndim < 2:
        raise ValueError("codes must have shape [num_examples, ..., num_features]")
    probe = np.load(probe_logits_npz)
    weights = np.asarray(probe["weights"], dtype=np.float64)
    ranking = json.loads(ranking_json.read_text(encoding="utf-8"))
    validate_feature_ranking(ranking)

    selected = _selected_feature_ids(ranking, top_k=top_k, num_features=codes.shape[-1])
    fired_count = np.sum(codes.reshape(-1, codes.shape[-1]) > 0, axis=0).astype(np.int64)
    matched_random = _bucket_matched_random_ids(
        fired_count=fired_count,
        selected=selected,
        seed=random_seed,
    )

    baseline_metrics = _evaluate_probe_arrays(
        codes=codes,
        probe=probe,
        weights=weights,
        task_type=task_type,
    )
    ablated_metrics = _evaluate_probe_arrays(
        codes=_zero_feature_channels(codes, selected),
        probe=probe,
        weights=weights,
        task_type=task_type,
    )
    random_metrics = _evaluate_probe_arrays(
        codes=_zero_feature_channels(codes, matched_random),
        probe=probe,
        weights=weights,
        task_type=task_type,
    )
    record = {
        "record_type": "feature_ablation_summary",
        "task_id": ranking["task_id"],
        "model_id": ranking["model_id"],
        "sae_id": ranking["sae_id"],
        "baseline_metrics": baseline_metrics,
        "ablations": [
            {
                "subset": f"top_{len(selected)}",
                "num_features": len(selected),
                "feature_ids": selected,
                "metrics": ablated_metrics,
                "metric_delta": _performance_drop(baseline_metrics, ablated_metrics),
            }
        ],
        "random_controls": [
            {
                "subset": f"matched_random_{len(matched_random)}_seed{random_seed}",
                "num_features": len(matched_random),
                "feature_ids": matched_random,
                "metrics": random_metrics,
                "metric_delta": _performance_drop(baseline_metrics, random_metrics),
            }
        ],
        "delta_convention": "positive_means_worse_after_ablation",
        "inputs": {
            "codes_npz": str(codes_npz),
            "probe_logits_npz": str(probe_logits_npz),
            "ranking_json": str(ranking_json),
        },
        "fixture": False,
    }
    validate_ablation_summary(record)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "feature_ablation_summary.json"
    summary_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "run_id": f"linear_ablation_{ranking['task_id']}_{ranking['model_id']}_{ranking['sae_id']}",
        "command": "feature-economy ablate-features",
        "git_commit": current_git_commit(),
        "config_files": [],
        "inputs": {
            "codes_npz": str(codes_npz),
            "probe_logits_npz": str(probe_logits_npz),
            "ranking_json": str(ranking_json),
            "task_type": task_type,
            "top_k": top_k,
            "random_seed": random_seed,
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


def _linear_logits(codes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    flat = _classification_readout_codes(codes, weights)
    if weights.shape[0] != flat.shape[1] + 1:
        raise ValueError(
            f"weight rows ({weights.shape[0]}) must equal classification readout dim + bias "
            f"({flat.shape[1] + 1})"
        )
    design = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    return design @ weights


def _classification_readout_codes(codes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Match classification ablation readout to the saved linear-probe weights."""

    if codes.ndim < 2:
        raise ValueError("codes must have shape [num_examples, ..., num_features]")
    flattened = codes.reshape(codes.shape[0], -1)
    if weights.shape[0] == flattened.shape[1] + 1:
        return flattened
    if codes.ndim > 2:
        pooled = np.mean(codes, axis=tuple(range(1, codes.ndim - 1)))
        if weights.shape[0] == pooled.shape[1] + 1:
            return pooled
    return flattened


def _dense_linear_output(codes: np.ndarray, weights: np.ndarray) -> np.ndarray:
    flat = codes.reshape(-1, codes.shape[-1])
    if weights.shape[0] != flat.shape[1] + 1:
        raise ValueError(
            f"weight rows ({weights.shape[0]}) must equal code dim + bias "
            f"({flat.shape[1] + 1})"
        )
    design = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    output = design @ weights
    if output.ndim == 1:
        return output.reshape(codes.shape[:-1])
    return output.reshape(*codes.shape[:-1], output.shape[-1])


def _zero_feature_channels(codes: np.ndarray, feature_ids: list[int]) -> np.ndarray:
    ablated = np.array(codes, copy=True)
    ablated[..., np.asarray(feature_ids, dtype=np.int64)] = 0.0
    return ablated


def _classification_metrics(
    *,
    logits: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    task_type: str,
) -> dict[str, float]:
    if task_type == "classification":
        class_indices = _labels_to_class_indices(labels, classes)
        return topk_accuracy(logits, class_indices, topk=(1, min(5, classes.size)))
    pred_indices = np.argmax(logits, axis=1)
    predictions = classes[pred_indices]
    return {"accuracy": accuracy(predictions, labels)}


def _evaluate_probe_arrays(
    *,
    codes: np.ndarray,
    probe: np.lib.npyio.NpzFile,
    weights: np.ndarray,
    task_type: str,
) -> dict[str, float]:
    if task_type in {"classification", "count_classification"}:
        labels = np.asarray(probe["labels"], dtype=np.int64)
        classes = np.asarray(probe["classes"], dtype=np.int64)
        if codes.shape[0] != labels.shape[0]:
            raise ValueError("codes first dimension must match probe labels")
        return _classification_metrics(
            logits=_linear_logits(codes, weights),
            labels=labels,
            classes=classes,
            task_type=task_type,
        )
    targets = np.asarray(probe["targets"])
    if codes.shape[:-1] != targets.shape:
        raise ValueError(
            f"dense codes shape {codes.shape} must match target shape {targets.shape} + feature dim"
        )
    if task_type == "dense_depth":
        prediction = _dense_linear_output(codes, weights)
        return depth_metrics(prediction, targets)
    classes = np.asarray(probe["classes"], dtype=np.int64)
    logits = _dense_linear_output(codes, weights)
    prediction = classes[np.argmax(logits, axis=-1)]
    return segmentation_metrics(
        prediction,
        targets,
        num_classes=int(classes.size),
    )


def _labels_to_class_indices(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    index = {int(label): idx for idx, label in enumerate(classes)}
    return np.asarray([index[int(label)] for label in labels], dtype=np.int64)


def _performance_drop(
    baseline_metrics: dict[str, float],
    ablated_metrics: dict[str, float],
) -> dict[str, float]:
    deltas = {}
    for metric in sorted(baseline_metrics):
        if metric in {"rmse", "abs_rel"}:
            deltas[metric] = float(ablated_metrics[metric] - baseline_metrics[metric])
        else:
            deltas[metric] = float(baseline_metrics[metric] - ablated_metrics[metric])
    return deltas
