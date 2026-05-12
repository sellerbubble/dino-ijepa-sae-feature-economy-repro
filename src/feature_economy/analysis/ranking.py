"""Task feature ranking and subset-usage summaries from saved SAE codes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from feature_economy.artifacts import (
    current_git_commit,
    validate_feature_ranking,
    validate_run_manifest,
    validate_subset_usage_summary,
)

from .availability import DEFAULT_BUCKETS


def rank_features_from_linear_probe(
    *,
    codes_npz: str | Path,
    probe_logits_npz: str | Path,
    task_id: str,
    model_id: str,
    sae_id: str,
    output_dir: str | Path,
    top_k: int = 100,
    ranking_method: str = "probe_weight",
    contribution_npz: str | Path | None = None,
    contribution_key: str = "validation_contribution_score",
    hybrid_alpha: float = 0.5,
) -> Path:
    """Rank SAE features with probe weights, validation contribution, or a hybrid score."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if ranking_method not in {"probe_weight", "validation_contribution", "hybrid"}:
        raise ValueError(f"unsupported ranking_method: {ranking_method}")
    if not 0.0 <= hybrid_alpha <= 1.0:
        raise ValueError("hybrid_alpha must be in [0, 1]")
    codes_npz = Path(codes_npz)
    probe_logits_npz = Path(probe_logits_npz)
    codes = np.asarray(np.load(codes_npz)["codes"], dtype=np.float64)
    weights = np.asarray(np.load(probe_logits_npz)["weights"], dtype=np.float64)
    if codes.ndim < 2:
        raise ValueError("codes must have shape [..., num_features]")
    flat = codes.reshape(-1, codes.shape[-1])
    # Last row is the bias because the linear probe appends a constant column.
    feature_weights = weights[:-1]
    if feature_weights.shape[0] != flat.shape[-1]:
        raise ValueError(
            f"weight rows ({feature_weights.shape[0]}) must match code dim ({flat.shape[-1]})"
        )
    probe_weight_score = _probe_weight_score(feature_weights)
    contribution_score = _load_contribution_scores(
        contribution_npz=contribution_npz,
        contribution_key=contribution_key,
        num_features=flat.shape[-1],
        required=ranking_method in {"validation_contribution", "hybrid"},
    )
    ranking_score = _ranking_score(
        ranking_method=ranking_method,
        probe_weight_score=probe_weight_score,
        contribution_score=contribution_score,
        hybrid_alpha=hybrid_alpha,
    )
    mean_activation = np.mean(flat, axis=0)
    positive = flat > 0
    positive_sums = np.sum(np.where(positive, flat, 0.0), axis=0)
    positive_counts = np.sum(positive, axis=0)
    mean_positive = np.divide(
        positive_sums,
        positive_counts,
        out=np.zeros_like(positive_sums, dtype=np.float64),
        where=positive_counts > 0,
    )
    order = np.argsort(ranking_score)[::-1][: min(top_k, flat.shape[-1])]
    rows = []
    for rank, feature_id in enumerate(order, start=1):
        rows.append(
            {
                "feature_id": int(feature_id),
                "task_rank": rank,
                "ranking_score": float(ranking_score[feature_id]),
                "probe_weight_score": float(probe_weight_score[feature_id]),
                "validation_contribution_score": float(contribution_score[feature_id]),
                "mean_activation": float(mean_activation[feature_id]),
                "mean_positive_activation": float(mean_positive[feature_id]),
            }
        )
    record = {
        "record_type": "task_feature_ranking",
        "task_id": task_id,
        "model_id": model_id,
        "sae_id": sae_id,
        "ranking_method": ranking_method,
        "rows": rows,
        "inputs": {
            "codes_npz": str(codes_npz),
            "probe_logits_npz": str(probe_logits_npz),
            "contribution_npz": str(contribution_npz) if contribution_npz else "",
            "contribution_key": contribution_key,
            "hybrid_alpha": hybrid_alpha,
        },
        "fixture": False,
    }
    validate_feature_ranking(record)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / "task_feature_ranking.json"
    ranking_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "run_id": f"ranking_{task_id}_{model_id}_{sae_id}",
        "command": "feature-economy rank-features",
        "git_commit": current_git_commit(),
        "config_files": [],
        "inputs": {
            "codes_npz": str(codes_npz),
            "probe_logits_npz": str(probe_logits_npz),
            "task_id": task_id,
            "model_id": model_id,
            "sae_id": sae_id,
            "top_k": top_k,
            "ranking_method": ranking_method,
            "contribution_npz": str(contribution_npz) if contribution_npz else "",
            "contribution_key": contribution_key,
            "hybrid_alpha": hybrid_alpha,
        },
        "outputs": {"ranking": str(ranking_path)},
        "fixture": False,
    }
    validate_run_manifest(manifest)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return ranking_path


def _load_contribution_scores(
    *,
    contribution_npz: str | Path | None,
    contribution_key: str,
    num_features: int,
    required: bool,
) -> np.ndarray:
    if contribution_npz is None:
        if required:
            raise ValueError(
                "ranking_method validation_contribution/hybrid requires contribution_npz"
            )
        return np.zeros(num_features, dtype=np.float64)
    arrays = np.load(contribution_npz)
    if contribution_key not in arrays:
        raise ValueError(f"{contribution_npz} does not contain array {contribution_key!r}")
    scores = np.asarray(arrays[contribution_key], dtype=np.float64)
    if scores.shape != (num_features,):
        raise ValueError(
            f"contribution scores must have shape [{num_features}], got {scores.shape}"
        )
    return scores


def _probe_weight_score(feature_weights: np.ndarray) -> np.ndarray:
    if feature_weights.ndim == 1:
        return np.abs(feature_weights)
    if feature_weights.ndim == 2:
        return np.max(np.abs(feature_weights), axis=1)
    raise ValueError("feature weights must be a 1D regression or 2D classification readout")


def _ranking_score(
    *,
    ranking_method: str,
    probe_weight_score: np.ndarray,
    contribution_score: np.ndarray,
    hybrid_alpha: float,
) -> np.ndarray:
    if ranking_method == "probe_weight":
        return probe_weight_score
    if ranking_method == "validation_contribution":
        return contribution_score
    if ranking_method == "hybrid":
        return (
            hybrid_alpha * _max_abs_normalize(probe_weight_score)
            + (1.0 - hybrid_alpha) * _max_abs_normalize(contribution_score)
        )
    raise ValueError(f"unsupported ranking_method: {ranking_method}")


def _max_abs_normalize(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values))) if values.size else 0.0
    if scale <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return values / scale


def compute_subset_usage_from_ranking(
    *,
    codes_npz: str | Path,
    ranking_json: str | Path,
    output_dir: str | Path,
    top_k: int = 100,
    random_seed: int = 0,
    threshold: float = 0.0,
    high_usage_threshold: int = 101,
) -> Path:
    """Compare top task features against a bucket-matched random subset.

    The matched random control preserves the fired-count bucket composition of
    the task-selected subset while excluding the selected features themselves.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if high_usage_threshold < 0:
        raise ValueError("high_usage_threshold must be non-negative")
    codes_npz = Path(codes_npz)
    ranking_json = Path(ranking_json)
    codes = np.asarray(np.load(codes_npz)["codes"])
    if codes.ndim < 2:
        raise ValueError("codes must have shape [..., num_features]")
    flat = codes.reshape(-1, codes.shape[-1])
    fired_count = np.sum(flat > threshold, axis=0).astype(np.int64)
    ranking = json.loads(ranking_json.read_text(encoding="utf-8"))
    validate_feature_ranking(ranking)
    selected = _selected_feature_ids(ranking, top_k=top_k, num_features=fired_count.shape[0])
    matched_random, matching_diagnostics = _bucket_matched_random_sample(
        fired_count=fired_count,
        selected=selected,
        seed=random_seed,
    )
    subsets = [
        _subset_usage_item(
            name=f"top_{len(selected)}",
            selection="task_selected",
            feature_ids=selected,
            fired_count=fired_count,
            high_usage_threshold=high_usage_threshold,
        ),
        _subset_usage_item(
            name=f"matched_random_{len(matched_random)}_seed{random_seed}",
            selection="matched_random",
            feature_ids=matched_random,
            fired_count=fired_count,
            high_usage_threshold=high_usage_threshold,
        ),
    ]
    record = {
        "record_type": "subset_usage_summary",
        "task_id": ranking["task_id"],
        "model_id": ranking["model_id"],
        "sae_id": ranking["sae_id"],
        "ranking_method": ranking["ranking_method"],
        "usage_unit": "fired_count",
        "subsets": subsets,
        "matching": {
            "strategy": "fired_count_bucket_matched_with_backfill",
            "bucket_boundaries": DEFAULT_BUCKETS,
            "exclude_task_selected": True,
            "random_seed": random_seed,
            "diagnostics": matching_diagnostics,
        },
        "threshold": threshold,
        "high_usage_threshold": high_usage_threshold,
        "inputs": {
            "codes_npz": str(codes_npz),
            "ranking_json": str(ranking_json),
        },
        "fixture": False,
    }
    validate_subset_usage_summary(record)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = output_dir / "subset_usage_summary.json"
    subset_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "run_id": f"subset_usage_{ranking['task_id']}_{ranking['model_id']}_{ranking['sae_id']}",
        "command": "feature-economy compute-subset-usage",
        "git_commit": current_git_commit(),
        "config_files": [],
        "inputs": {
            "codes_npz": str(codes_npz),
            "ranking_json": str(ranking_json),
            "top_k": top_k,
            "random_seed": random_seed,
            "threshold": threshold,
            "high_usage_threshold": high_usage_threshold,
        },
        "outputs": {"subset_usage": str(subset_path)},
        "fixture": False,
    }
    validate_run_manifest(manifest)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return subset_path


def _selected_feature_ids(
    ranking: dict,
    *,
    top_k: int,
    num_features: int,
) -> list[int]:
    rows = sorted(ranking["rows"], key=lambda row: row["task_rank"])
    selected = []
    seen = set()
    for row in rows:
        feature_id = int(row["feature_id"])
        if not 0 <= feature_id < num_features:
            raise ValueError(f"ranked feature_id {feature_id} is outside code dimension")
        if feature_id in seen:
            raise ValueError(f"duplicate ranked feature_id: {feature_id}")
        selected.append(feature_id)
        seen.add(feature_id)
        if len(selected) >= top_k:
            break
    if not selected:
        raise ValueError("ranking does not contain any selectable feature rows")
    return selected


def _bucket_matched_random_ids(
    *,
    fired_count: np.ndarray,
    selected: list[int],
    seed: int,
) -> list[int]:
    matched, _ = _bucket_matched_random_sample(
        fired_count=fired_count,
        selected=selected,
        seed=seed,
    )
    return matched


def _bucket_matched_random_sample(
    *,
    fired_count: np.ndarray,
    selected: list[int],
    seed: int,
) -> tuple[list[int], dict]:
    rng = np.random.default_rng(seed)
    selected_set = set(selected)
    matched: list[int] = []
    matched_set: set[int] = set()
    bucket_diagnostics = []
    for bucket_name in DEFAULT_BUCKETS:
        selected_in_bucket = [
            feature_id
            for feature_id in selected
            if _bucket_name(int(fired_count[feature_id])) == bucket_name
        ]
        if not selected_in_bucket:
            continue
        candidates = np.asarray(
            [
                feature_id
                for feature_id, count in enumerate(fired_count)
                if feature_id not in selected_set and _bucket_name(int(count)) == bucket_name
            ],
            dtype=np.int64,
        )
        sample_size = min(candidates.size, len(selected_in_bucket))
        if sample_size:
            sampled = rng.choice(candidates, size=sample_size, replace=False)
            sampled_ids = [int(feature_id) for feature_id in sampled]
            matched.extend(sampled_ids)
            matched_set.update(sampled_ids)
        bucket_diagnostics.append(
            {
                "bucket": bucket_name,
                "requested": len(selected_in_bucket),
                "available": int(candidates.size),
                "sampled": int(sample_size),
                "shortage": int(max(len(selected_in_bucket) - sample_size, 0)),
            }
        )

    backfill_needed = len(selected) - len(matched)
    if backfill_needed > 0:
        fallback_candidates = np.asarray(
            [
                feature_id
                for feature_id in range(fired_count.shape[0])
                if feature_id not in selected_set and feature_id not in matched_set
            ],
            dtype=np.int64,
        )
        if fallback_candidates.size < backfill_needed:
            raise ValueError(
                "not enough non-selected candidates to backfill matched random subset: "
                f"need {backfill_needed}, found {fallback_candidates.size}"
            )
        sampled = rng.choice(fallback_candidates, size=backfill_needed, replace=False)
        matched.extend(int(feature_id) for feature_id in sampled)

    diagnostics = {
        "requested_total": len(selected),
        "sampled_total": len(matched),
        "backfill_count": backfill_needed,
        "bucket_diagnostics": bucket_diagnostics,
    }
    return matched, diagnostics


def _bucket_name(count: int) -> str:
    for name, (lower, upper) in DEFAULT_BUCKETS.items():
        if count >= lower and (upper is None or count <= upper):
            return name
    raise ValueError(f"count {count} does not fit any usage bucket")


def _subset_usage_item(
    *,
    name: str,
    selection: str,
    feature_ids: list[int],
    fired_count: np.ndarray,
    high_usage_threshold: int,
) -> dict:
    usage = fired_count[np.asarray(feature_ids, dtype=np.int64)]
    return {
        "name": name,
        "selection": selection,
        "num_features": len(feature_ids),
        "median_usage": float(np.median(usage)),
        "high_usage_fraction": float(np.mean(usage >= high_usage_threshold)),
        "feature_ids": feature_ids,
    }
