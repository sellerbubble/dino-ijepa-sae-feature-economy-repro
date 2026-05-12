"""Transform policy metadata and lightweight image preprocessing.

This module keeps the public transform contract small and auditable. The
preprocessing helper uses PIL and NumPy only, so the first real data gate can run
without importing torch or instantiating DINO/I-JEPA.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TransformPolicy:
    """Serializable transform policy for public runner configs."""

    mode: str
    resize: int
    crop: int
    normalize: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


class TransformError(RuntimeError):
    """Raised when image preprocessing cannot be applied."""


def transform_policy_from_config(transform_config: dict[str, Any]) -> TransformPolicy:
    """Build a transform policy from a model config's `transform` block."""

    mode = transform_config.get("mode")
    resize = transform_config.get("resize")
    crop = transform_config.get("crop")
    normalize = transform_config.get("normalize")
    if mode != "shared_imagenet":
        raise ValueError(f"unsupported transform mode: {mode!r}")
    if normalize != "imagenet":
        raise ValueError(f"unsupported normalization policy: {normalize!r}")
    if not isinstance(resize, int) or resize <= 0:
        raise ValueError("transform.resize must be a positive integer")
    if not isinstance(crop, int) or crop <= 0:
        raise ValueError("transform.crop must be a positive integer")
    if crop > resize:
        raise ValueError("transform.crop cannot exceed transform.resize")
    return TransformPolicy(
        mode=mode,
        resize=resize,
        crop=crop,
        normalize=normalize,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )


def preprocess_image_file(image_path: str | Path, policy: TransformPolicy) -> np.ndarray:
    """Load an image and return normalized CHW float32 pixels."""

    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise TransformError("Pillow is required for image preprocessing") from exc

    image_path = Path(image_path)
    try:
        with Image.open(image_path) as image:
            return preprocess_pil_image(image, policy)
    except FileNotFoundError as exc:
        raise TransformError(f"image does not exist: {image_path}") from exc


def preprocess_pil_image(image: Any, policy: TransformPolicy) -> np.ndarray:
    """Apply resize, center crop, RGB conversion, and ImageNet normalization."""

    if policy.mode != "shared_imagenet":
        raise TransformError(f"unsupported transform policy: {policy.mode!r}")
    image = image.convert("RGB")
    image = _resize_shorter_side(image, policy.resize)
    image = _center_crop(image, policy.crop)
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(policy.mean, dtype=np.float32)
    std = np.asarray(policy.std, dtype=np.float32)
    array = (array - mean) / std
    return np.transpose(array, (2, 0, 1)).astype(np.float32)


def _resize_shorter_side(image: Any, shorter_side: int) -> Any:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise TransformError("image has invalid size")
    scale = shorter_side / min(width, height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    return image.resize((new_width, new_height), resample=_pil_bicubic_resample())


def _center_crop(image: Any, crop_size: int) -> Any:
    width, height = image.size
    if width < crop_size or height < crop_size:
        raise TransformError(
            f"resized image {width}x{height} is smaller than crop {crop_size}"
        )
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    return image.crop((left, top, left + crop_size, top + crop_size))


def _pil_bicubic_resample() -> int:
    from PIL import Image

    return getattr(getattr(Image, "Resampling", Image), "BICUBIC")
