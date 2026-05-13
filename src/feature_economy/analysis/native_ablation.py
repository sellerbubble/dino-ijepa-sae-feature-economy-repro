"""Module F native-subspace ablation over saved public arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from feature_economy.artifacts import (
    current_git_commit,
    validate_feature_ranking,
    validate_native_subspace_ablation_summary,
    validate_run_manifest,
)
from feature_economy.models.sae_codes import encode_linear_topk, load_lightweight_sae_checkpoint

from .ablation import _evaluate_probe_arrays, _performance_drop
from .ranking import _selected_feature_ids


def ablate_native_subspace(
    *,
    features_npz: str | Path,
    sae_checkpoint: str | Path,
    probe_logits_npz: str | Path,
    ranking_json: str | Path,
    task_type: str,
    output_dir: str | Path,
    top_k: int = 20,
    random_seed: int = 0,
    random_pool: str = "exclude_topk",
    normalize_activations: str = "layer_norm",
    topk: int = 32,
) -> Path:
    """Remove SAE-decoder native subspaces and evaluate the frozen SAE probe.

    This is the public saved-array counterpart of the paper's Module F native
    intervention. It performs projection removal in the SAE runtime-normalized
    coordinate system, centered by `decoder_bias` (`b_dec`), then re-encodes the
    intervened native features through the same lightweight SAE and frozen probe.
    """

    if random_pool not in {"dictionary", "exclude_topk"}:
        raise ValueError("random_pool must be dictionary or exclude_topk")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    features_npz = Path(features_npz)
    sae_checkpoint = Path(sae_checkpoint)
    probe_logits_npz = Path(probe_logits_npz)
    ranking_json = Path(ranking_json)
    features = np.asarray(np.load(features_npz)["features"], dtype=np.float32)
    checkpoint = load_lightweight_sae_checkpoint(sae_checkpoint)
    if "decoder_weight" not in checkpoint:
        raise ValueError(
            "native subspace ablation requires decoder_weight/W_dec in the lightweight SAE checkpoint"
        )
    decoder_weight = np.asarray(checkpoint["decoder_weight"], dtype=np.float32)
    decoder_bias = np.asarray(checkpoint["decoder_bias"], dtype=np.float32)
    code_dim = decoder_weight.shape[0]

    ranking = json.loads(ranking_json.read_text(encoding="utf-8"))
    validate_feature_ranking(ranking)
    selected = _selected_feature_ids(ranking, top_k=top_k, num_features=code_dim)
    dictionary_ids = list(range(code_dim))
    if random_pool == "exclude_topk":
        pool = [feature_id for feature_id in dictionary_ids if feature_id not in set(selected)]
    else:
        pool = dictionary_ids
    if len(pool) < len(selected):
        raise ValueError("random pool is smaller than selected feature set")
    rng = np.random.default_rng(random_seed)
    matched_random = sorted(int(x) for x in rng.choice(pool, size=len(selected), replace=False))

    probe = np.load(probe_logits_npz)
    weights = np.asarray(probe["weights"], dtype=np.float64)
    baseline_codes = encode_linear_topk(
        features,
        checkpoint=checkpoint,
        normalize_activations=normalize_activations,
        topk=topk,
    )
    baseline_metrics = _evaluate_probe_arrays(
        codes=baseline_codes,
        probe=probe,
        weights=weights,
        task_type=task_type,
    )
    top_metrics = _evaluate_probe_arrays(
        codes=_encode_after_native_projection_removal(
            features,
            checkpoint=checkpoint,
            decoder_weight=decoder_weight,
            decoder_bias=decoder_bias,
            feature_ids=selected,
            normalize_activations=normalize_activations,
            topk=topk,
        ),
        probe=probe,
        weights=weights,
        task_type=task_type,
    )
    random_metrics = _evaluate_probe_arrays(
        codes=_encode_after_native_projection_removal(
            features,
            checkpoint=checkpoint,
            decoder_weight=decoder_weight,
            decoder_bias=decoder_bias,
            feature_ids=matched_random,
            normalize_activations=normalize_activations,
            topk=topk,
        ),
        probe=probe,
        weights=weights,
        task_type=task_type,
    )

    covariance = _native_covariance(features, normalize_activations=normalize_activations)
    top_geometry = _subspace_geometry(decoder_weight, selected, covariance)
    random_geometry = _subspace_geometry(decoder_weight, matched_random, covariance)
    coordinate = _projection_coordinate(normalize_activations)
    record = {
        "record_type": "native_subspace_ablation_summary",
        "task_id": ranking["task_id"],
        "model_id": ranking["model_id"],
        "sae_id": ranking["sae_id"],
        "task_type": task_type,
        "baseline_metrics": baseline_metrics,
        "projection_coordinate_system": {
            "space": coordinate,
            "normalize_activations": normalize_activations,
            "center": "decoder_bias",
        },
        "ablations": [
            {
                "subset": f"top_{len(selected)}",
                "num_features": len(selected),
                "feature_ids": selected,
                "metrics": top_metrics,
                "metric_delta": _performance_drop(baseline_metrics, top_metrics),
                "geometry": top_geometry,
            }
        ],
        "random_controls": [
            {
                "subset": f"matched_random_{len(matched_random)}_seed{random_seed}",
                "pool": random_pool,
                "num_features": len(matched_random),
                "feature_ids": matched_random,
                "metrics": random_metrics,
                "metric_delta": _performance_drop(baseline_metrics, random_metrics),
                "geometry": random_geometry,
            }
        ],
        "delta_convention": "positive_means_worse_after_ablation",
        "inputs": {
            "features_npz": str(features_npz),
            "sae_checkpoint": str(sae_checkpoint),
            "probe_logits_npz": str(probe_logits_npz),
            "ranking_json": str(ranking_json),
        },
        "fixture": False,
    }
    validate_native_subspace_ablation_summary(record)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "native_subspace_ablation_summary.json"
    summary_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "run_id": f"native_subspace_ablation_{ranking['task_id']}_{ranking['model_id']}_{ranking['sae_id']}",
        "command": "feature-economy ablate-native-subspace",
        "git_commit": current_git_commit(),
        "config_files": [],
        "inputs": {
            "features_npz": str(features_npz),
            "sae_checkpoint": str(sae_checkpoint),
            "probe_logits_npz": str(probe_logits_npz),
            "ranking_json": str(ranking_json),
            "task_type": task_type,
            "top_k": top_k,
            "random_seed": random_seed,
            "random_pool": random_pool,
            "normalize_activations": normalize_activations,
            "topk": topk,
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


def _encode_after_native_projection_removal(
    features: np.ndarray,
    *,
    checkpoint: dict[str, np.ndarray],
    decoder_weight: np.ndarray,
    decoder_bias: np.ndarray,
    feature_ids: list[int],
    normalize_activations: str,
    topk: int,
) -> np.ndarray:
    intervened = _remove_native_projection(
        features,
        decoder_weight=decoder_weight,
        decoder_bias=decoder_bias,
        feature_ids=feature_ids,
        normalize_activations=normalize_activations,
    )
    return encode_linear_topk(
        intervened,
        checkpoint=checkpoint,
        normalize_activations=normalize_activations,
        topk=topk,
    )


def _remove_native_projection(
    features: np.ndarray,
    *,
    decoder_weight: np.ndarray,
    decoder_bias: np.ndarray,
    feature_ids: list[int],
    normalize_activations: str,
) -> np.ndarray:
    basis = _projection_basis(decoder_weight, feature_ids)
    flat = features.reshape(-1, features.shape[-1]).astype(np.float32)
    normalized, norm_state = _apply_runtime_normalization(flat, normalize_activations)
    centered = normalized - decoder_bias
    removed = (centered @ basis) @ basis.T
    centered = centered - removed
    restored = _invert_runtime_normalization(centered + decoder_bias, norm_state)
    return restored.reshape(features.shape).astype(np.float32)


def _projection_basis(decoder_weight: np.ndarray, feature_ids: list[int]) -> np.ndarray:
    selected = decoder_weight[np.asarray(feature_ids, dtype=np.int64)]
    norms = np.linalg.norm(selected, axis=1, keepdims=True)
    selected = selected / np.maximum(norms, 1e-12)
    basis, _ = np.linalg.qr(selected.T, mode="reduced")
    return basis.astype(np.float32)


def _apply_runtime_normalization(
    flat: np.ndarray,
    normalize_activations: str,
) -> tuple[np.ndarray, tuple[str, np.ndarray, np.ndarray] | None]:
    if normalize_activations == "none":
        return flat, None
    if normalize_activations == "layer_norm":
        mean = np.mean(flat, axis=-1, keepdims=True)
        std = np.std(flat, axis=-1, keepdims=True)
        return (flat - mean) / np.maximum(std, 1e-6), ("layer_norm", mean, std)
    raise ValueError(f"unsupported normalize_activations: {normalize_activations}")


def _invert_runtime_normalization(
    flat: np.ndarray,
    norm_state: tuple[str, np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    if norm_state is None:
        return flat
    kind, mean, std = norm_state
    if kind != "layer_norm":
        raise ValueError(f"unsupported normalization state: {kind}")
    return flat * np.maximum(std, 1e-6) + mean


def _native_covariance(features: np.ndarray, *, normalize_activations: str) -> np.ndarray:
    flat = features.reshape(-1, features.shape[-1]).astype(np.float64)
    normalized, _ = _apply_runtime_normalization(flat.astype(np.float32), normalize_activations)
    centered = normalized.astype(np.float64) - np.mean(normalized, axis=0, keepdims=True)
    return (centered.T @ centered) / max(centered.shape[0], 1)


def _subspace_geometry(
    decoder_weight: np.ndarray,
    feature_ids: list[int],
    covariance: np.ndarray,
) -> dict[str, float | int | None]:
    basis = _projection_basis(decoder_weight, feature_ids)
    selected = decoder_weight[np.asarray(feature_ids, dtype=np.int64)]
    selected = selected / np.maximum(np.linalg.norm(selected, axis=1, keepdims=True), 1e-12)
    if len(feature_ids) >= 2:
        gram = selected @ selected.T
        tri = np.triu_indices(len(feature_ids), k=1)
        pairwise = gram[tri]
        mean_cos = float(np.mean(pairwise))
        mean_abs_cos = float(np.mean(np.abs(pairwise)))
    else:
        mean_cos = None
        mean_abs_cos = None
    total_trace = float(np.trace(covariance))
    variance_share = None
    if total_trace > 0:
        variance_share = float(np.trace(basis.T @ covariance @ basis) / total_trace)
    return {
        "subspace_rank": int(basis.shape[1]),
        "mean_pairwise_cosine": mean_cos,
        "mean_abs_pairwise_cosine": mean_abs_cos,
        "native_variance_share": variance_share,
    }


def _projection_coordinate(normalize_activations: str) -> str:
    if normalize_activations == "layer_norm":
        return "sae_runtime_normalized_hidden_minus_decoder_bias"
    return "native_hidden_minus_decoder_bias"
