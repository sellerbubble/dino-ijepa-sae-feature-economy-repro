"""Dense target export helpers for public full-profile reruns.

These helpers convert manifest-referenced depth maps or segmentation masks into
the public `targets.npz` contract used by lightweight dense probes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from feature_economy.artifacts import current_git_commit, validate_run_manifest

from .adapters import ManifestDataset


def export_dense_targets(
    *,
    manifest_path: str | Path,
    task_type: str,
    output_dir: str | Path,
    expected_split: str | None = None,
    features_npz: str | Path | None = None,
    target_shape: tuple[int, int] | None = None,
    max_examples: int | None = None,
    target_key: str = "targets",
) -> Path:
    """Export dense targets aligned to a manifest and optional feature grid."""

    if task_type not in {"dense_depth", "dense_segmentation"}:
        raise ValueError("export_dense_targets requires dense_depth or dense_segmentation")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive when provided")
    if target_shape is None and features_npz is not None:
        target_shape = _target_shape_from_features(features_npz)
    if target_shape is not None and (len(target_shape) != 2 or min(target_shape) <= 0):
        raise ValueError("target_shape must be a positive (height, width) pair")

    dataset = ManifestDataset(
        manifest_path,
        task_type=task_type,
        expected_split=expected_split,
    )
    records = dataset.records[:max_examples] if max_examples is not None else dataset.records
    if not records:
        raise ValueError("no records selected for target export")

    targets = []
    source_paths = []
    for record in records:
        if task_type == "dense_depth":
            if record.depth is None:
                raise ValueError("dense_depth record is missing depth path")
            source = dataset.resolve_path(record.depth)
            target = _load_depth_target(source)
            if target_shape is not None:
                target = _resize_float_target(target, target_shape)
            target = target.astype(np.float32)
        else:
            if record.segmentation is None:
                raise ValueError("dense_segmentation record is missing segmentation path")
            source = dataset.resolve_path(record.segmentation)
            target = _load_segmentation_target(source)
            if target_shape is not None:
                target = _resize_integer_target(target, target_shape)
            target = target.astype(np.int64)
        targets.append(target)
        source_paths.append(str(source))

    targets_np = np.stack(targets, axis=0)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "targets.npz"
    np.savez_compressed(
        target_path,
        **{target_key: targets_np},
        source_path=np.asarray(source_paths),
    )

    summary = {
        "record_type": "dense_target_export_summary",
        "task_type": task_type,
        "input_manifest": str(manifest_path),
        "output_npz": str(target_path),
        "target_key": target_key,
        "num_examples": int(targets_np.shape[0]),
        "target_shape": list(targets_np.shape),
        "target_dtype": str(targets_np.dtype),
        "expected_split": expected_split,
        "features_npz": str(features_npz) if features_npz is not None else None,
        "fixture": False,
    }
    summary_path = output_dir / "dense_target_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "run_id": f"export_dense_targets_{task_type}",
        "command": "feature-economy export-targets",
        "git_commit": current_git_commit(),
        "config_files": [],
        "inputs": {
            "manifest": str(manifest_path),
            "task_type": task_type,
            "expected_split": expected_split,
            "features_npz": str(features_npz) if features_npz is not None else None,
            "target_shape": list(target_shape) if target_shape is not None else None,
            "max_examples": max_examples,
        },
        "outputs": {
            "targets": str(target_path),
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


def _target_shape_from_features(features_npz: str | Path) -> tuple[int, int]:
    arrays = np.load(features_npz)
    if "features" not in arrays:
        raise ValueError(f"{features_npz} does not contain array 'features'")
    features = np.asarray(arrays["features"])
    if features.ndim != 4:
        raise ValueError(
            "dense target shape can only be inferred from spatial features "
            f"[num_examples, height, width, dim], got {features.shape}"
        )
    return int(features.shape[1]), int(features.shape[2])


def _load_depth_target(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.float32)
    if path.suffix == ".npz":
        arrays = np.load(path)
        key = "depth" if "depth" in arrays else arrays.files[0]
        return np.asarray(arrays[key], dtype=np.float32)
    return np.asarray(_load_image(path), dtype=np.float32)


def _load_segmentation_target(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path), dtype=np.int64)
    if path.suffix == ".npz":
        arrays = np.load(path)
        key = "segmentation" if "segmentation" in arrays else arrays.files[0]
        return np.asarray(arrays[key], dtype=np.int64)
    return np.asarray(_load_image(path), dtype=np.int64)


def _load_image(path: Path) -> Any:
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("loading image targets requires Pillow") from exc
    with Image.open(path) as image:
        return image.copy()


def _resize_float_target(target: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if target.shape == shape:
        return target
    try:
        from PIL import Image
    except Exception:
        return _resize_nearest_numpy(np.asarray(target), shape).astype(np.float32)
    image = Image.fromarray(np.asarray(target, dtype=np.float32), mode="F")
    return np.asarray(image.resize((shape[1], shape[0]), resample=_pil_resampling(Image, "BILINEAR")))


def _resize_integer_target(target: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if target.shape == shape:
        return target
    try:
        from PIL import Image
    except Exception:
        return _resize_nearest_numpy(np.asarray(target), shape).astype(np.int64)
    image = Image.fromarray(np.asarray(target, dtype=np.int32), mode="I")
    resized = image.resize((shape[1], shape[0]), resample=_pil_resampling(Image, "NEAREST"))
    return np.asarray(resized, dtype=np.int64)


def _pil_resampling(image_module: Any, name: str) -> int:
    resampling = getattr(image_module, "Resampling", image_module)
    return int(getattr(resampling, name))


def _resize_nearest_numpy(target: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Dependency-light nearest-neighbor resize used when Pillow is unavailable."""

    if target.ndim != 2:
        raise ValueError(f"dense target resize expects 2D arrays, got shape {target.shape}")
    row_idx = np.linspace(0, target.shape[0] - 1, shape[0]).round().astype(np.int64)
    col_idx = np.linspace(0, target.shape[1] - 1, shape[1]).round().astype(np.int64)
    return target[row_idx[:, None], col_idx[None, :]]
