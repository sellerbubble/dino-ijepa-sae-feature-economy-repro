"""Task metrics used by public probe runners."""

from __future__ import annotations

import numpy as np


def topk_accuracy(logits: np.ndarray, labels: np.ndarray, topk: tuple[int, ...] = (1,)) -> dict[str, float]:
    """Compute top-k classification accuracy."""

    logits = np.asarray(logits)
    labels = np.asarray(labels)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [num_examples, num_classes]")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError("labels must have shape [num_examples]")
    if not topk:
        raise ValueError("topk must not be empty")
    max_k = max(topk)
    if max_k > logits.shape[1]:
        raise ValueError("topk cannot exceed number of classes")
    order = np.argsort(logits, axis=1)[:, ::-1][:, :max_k]
    return {
        f"top{k}": float(np.mean(np.any(order[:, :k] == labels[:, None], axis=1)))
        for k in topk
    }


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Compute exact-match accuracy."""

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must have the same shape")
    return float(np.mean(predictions == labels))


def depth_metrics(prediction: np.ndarray, target: np.ndarray, valid_mask: np.ndarray | None = None) -> dict[str, float]:
    """Compute RMSE, AbsRel, and delta1 for dense depth."""

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    if valid_mask is None:
        valid_mask = target > 0
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != target.shape:
            raise ValueError("valid_mask must match target shape")
    pred = prediction[valid_mask]
    tgt = target[valid_mask]
    if pred.size == 0:
        raise ValueError("no valid depth pixels")
    pred = np.maximum(pred, 1e-8)
    tgt = np.maximum(tgt, 1e-8)
    rmse = np.sqrt(np.mean((pred - tgt) ** 2))
    abs_rel = np.mean(np.abs(pred - tgt) / tgt)
    ratio = np.maximum(pred / tgt, tgt / pred)
    delta1 = np.mean(ratio < 1.25)
    return {"rmse": float(rmse), "abs_rel": float(abs_rel), "delta1": float(delta1)}


def segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    num_classes: int,
    ignore_index: int | None = None,
) -> dict[str, float]:
    """Compute pixel accuracy and mean IoU for semantic segmentation."""

    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    valid = np.ones_like(target, dtype=bool)
    if ignore_index is not None:
        valid = target != ignore_index
    pred = prediction[valid]
    tgt = target[valid]
    if pred.size == 0:
        raise ValueError("no valid segmentation pixels")
    pixel_accuracy = float(np.mean(pred == tgt))
    ious: list[float] = []
    for class_id in range(num_classes):
        pred_mask = pred == class_id
        tgt_mask = tgt == class_id
        union = np.logical_or(pred_mask, tgt_mask).sum()
        if union == 0:
            continue
        intersection = np.logical_and(pred_mask, tgt_mask).sum()
        ious.append(float(intersection / union))
    miou = float(np.mean(ious)) if ious else 0.0
    return {"miou": miou, "pixel_accuracy": pixel_accuracy}
