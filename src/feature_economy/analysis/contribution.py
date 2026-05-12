"""Validation-set contribution scores for SAE features.

This module provides a lightweight public counterpart to the paper's ranking
sensitivity controls. It computes a per-feature score by zeroing one SAE
channel at a time and measuring how much the saved linear probe gets worse.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from feature_economy.artifacts import (
    current_git_commit,
    validate_contribution_scores_summary,
    validate_run_manifest,
)

from .ablation import (
    _evaluate_probe_arrays,
    _labels_to_class_indices,
    _performance_drop,
    _zero_feature_channels,
)


DEFAULT_PRIMARY_METRICS = {
    "classification": "top1",
    "count_classification": "accuracy",
    "dense_depth": "rmse",
    "dense_segmentation": "miou",
}


def compute_linear_probe_contribution_scores(
    *,
    codes_npz: str | Path,
    probe_logits_npz: str | Path,
    task_type: str,
    task_id: str,
    model_id: str,
    sae_id: str,
    output_dir: str | Path,
    primary_metric: str | None = None,
    max_features: int | None = None,
    score_key: str = "validation_contribution_score",
    scoring_method: str = "exact_metric_drop",
) -> Path:
    """Write per-feature validation contribution scores for a saved linear probe."""

    if task_type not in DEFAULT_PRIMARY_METRICS:
        raise ValueError(f"unsupported task_type: {task_type}")
    if max_features is not None and max_features <= 0:
        raise ValueError("max_features must be positive when provided")
    if not score_key:
        raise ValueError("score_key must be non-empty")
    if scoring_method not in {"exact_metric_drop", "true_class_logit_drop", "weight_activation"}:
        raise ValueError(
            "scoring_method must be one of: exact_metric_drop, true_class_logit_drop, weight_activation"
        )
    if scoring_method == "true_class_logit_drop" and task_type not in {
        "classification",
        "count_classification",
    }:
        raise ValueError("true_class_logit_drop is only supported for classification tasks")

    codes_npz = Path(codes_npz)
    probe_logits_npz = Path(probe_logits_npz)
    codes = np.asarray(np.load(codes_npz)["codes"], dtype=np.float64)
    if codes.ndim < 2:
        raise ValueError("codes must have shape [num_examples, ..., num_features]")
    probe = np.load(probe_logits_npz)
    weights = np.asarray(probe["weights"], dtype=np.float64)
    metric_name = primary_metric or DEFAULT_PRIMARY_METRICS[task_type]

    baseline_metrics = _evaluate_probe_arrays(
        codes=codes,
        probe=probe,
        weights=weights,
        task_type=task_type,
    )
    if metric_name not in baseline_metrics:
        raise ValueError(
            f"primary_metric {metric_name!r} not found in baseline metrics: "
            f"{sorted(baseline_metrics)}"
        )

    num_features = int(codes.shape[-1])
    scored_features = num_features if max_features is None else min(max_features, num_features)
    if scoring_method == "true_class_logit_drop":
        scores = _true_class_logit_drop_scores(
            codes=codes,
            probe=probe,
            weights=weights,
            scored_features=scored_features,
        )
    elif scoring_method == "weight_activation":
        scores = _weight_activation_scores(
            codes=codes,
            weights=weights,
            scored_features=scored_features,
        )
    else:
        scores = np.zeros(num_features, dtype=np.float32)
        for feature_id in range(scored_features):
            ablated_metrics = _evaluate_probe_arrays(
                codes=_zero_feature_channels(codes, [feature_id]),
                probe=probe,
                weights=weights,
                task_type=task_type,
            )
            drop = _performance_drop(baseline_metrics, ablated_metrics)
            scores[feature_id] = float(drop[metric_name])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "contribution_scores.npz"
    np.savez_compressed(scores_path, **{score_key: scores})

    summary = {
        "record_type": "contribution_scores_summary",
        "task_id": task_id,
        "model_id": model_id,
        "sae_id": sae_id,
        "task_type": task_type,
        "primary_metric": metric_name,
        "score_key": score_key,
        "scoring_method": scoring_method,
        "output_npz": str(scores_path),
        "num_features": num_features,
        "scored_features": scored_features,
        "baseline_metrics": baseline_metrics,
        "score_quantiles": _score_quantiles(scores[:scored_features]),
        "top_features": _top_feature_rows(scores, scored_features=scored_features, top_k=10),
        "inputs": {
            "codes_npz": str(codes_npz),
            "probe_logits_npz": str(probe_logits_npz),
            "max_features": max_features,
            "scoring_method": scoring_method,
        },
        "fixture": False,
    }
    validate_contribution_scores_summary(summary)
    summary_path = output_dir / "contribution_scores_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "run_id": f"contribution_{task_id}_{model_id}_{sae_id}",
        "command": "feature-economy compute-contributions",
        "git_commit": current_git_commit(),
        "config_files": [],
        "inputs": {
            "codes_npz": str(codes_npz),
            "probe_logits_npz": str(probe_logits_npz),
            "task_type": task_type,
            "task_id": task_id,
            "model_id": model_id,
            "sae_id": sae_id,
            "primary_metric": metric_name,
            "max_features": max_features,
            "score_key": score_key,
            "scoring_method": scoring_method,
        },
        "outputs": {
            "scores": str(scores_path),
            "summary": str(summary_path),
        },
        "fixture": False,
    }
    validate_run_manifest(manifest)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def _weight_activation_scores(
    *,
    codes: np.ndarray,
    weights: np.ndarray,
    scored_features: int,
) -> np.ndarray:
    """Approximate contribution from probe weight magnitude and mean activation.

    This is the dense-task counterpart to the classification-only
    `true_class_logit_drop` shortcut. Exact single-feature metric-drop is the
    most direct public score, but for dense probes it can be prohibitively slow
    because every feature ablation recomputes a full pixelwise prediction. This
    score asks a cheaper question: which channels are both active on validation
    data and heavily read by the saved linear probe?
    """

    flat = codes.reshape(-1, codes.shape[-1])
    if weights.shape[0] != flat.shape[1] + 1:
        raise ValueError(
            f"weight rows ({weights.shape[0]}) must equal code dim + bias "
            f"({flat.shape[1] + 1})"
        )
    readout_weight = np.asarray(weights[:-1], dtype=np.float64)
    if readout_weight.ndim == 2:
        readout_strength = np.linalg.norm(readout_weight, axis=1)
    else:
        readout_strength = np.abs(readout_weight)
    mean_positive = np.maximum(flat[:, :scored_features], 0.0).mean(axis=0)
    scores = np.zeros(flat.shape[1], dtype=np.float32)
    scores[:scored_features] = (
        mean_positive * readout_strength[:scored_features]
    ).astype(np.float32)
    return scores


def _true_class_logit_drop_scores(
    *,
    codes: np.ndarray,
    probe: np.lib.npyio.NpzFile,
    weights: np.ndarray,
    scored_features: int,
) -> np.ndarray:
    """Approximate per-feature contribution by true-class logit support.

    Exact per-feature metric-drop ablation is useful for tiny/debug bundles but
    scales poorly for ImageNet because it recomputes a full 50k x 1000 probe
    for every SAE channel. This public-release score asks a cheaper question:
    if a feature were zeroed, how much true-class logit support would be
    removed on average? Positive support is retained so inhibitory or noisy
    channels do not look helpful merely because they are large in magnitude.
    """

    labels = np.asarray(probe["labels"], dtype=np.int64)
    classes = np.asarray(probe["classes"], dtype=np.int64)
    class_indices = _labels_to_class_indices(labels, classes)
    flat = codes.reshape(codes.shape[0], -1)
    if weights.shape[0] != flat.shape[1] + 1:
        raise ValueError(
            f"weight rows ({weights.shape[0]}) must equal flattened code dim + bias "
            f"({flat.shape[1] + 1})"
        )
    feature_weights_for_label = weights[: flat.shape[1], class_indices].T
    support = flat[:, :scored_features] * feature_weights_for_label[:, :scored_features]
    scores = np.zeros(flat.shape[1], dtype=np.float32)
    scores[:scored_features] = np.maximum(support, 0.0).mean(axis=0).astype(np.float32)
    return scores


def _score_quantiles(scores: np.ndarray) -> dict[str, float]:
    if scores.size == 0:
        return {}
    return {
        "min": float(np.min(scores)),
        "p25": float(np.quantile(scores, 0.25)),
        "median": float(np.quantile(scores, 0.50)),
        "p75": float(np.quantile(scores, 0.75)),
        "max": float(np.max(scores)),
    }


def _top_feature_rows(
    scores: np.ndarray,
    *,
    scored_features: int,
    top_k: int,
) -> list[dict[str, float | int]]:
    if scored_features <= 0:
        return []
    order = np.argsort(scores[:scored_features])[::-1][: min(top_k, scored_features)]
    return [
        {
            "feature_id": int(feature_id),
            "score": float(scores[feature_id]),
        }
        for feature_id in order
    ]
