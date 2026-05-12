"""JSONL manifest validation for public dataset adapters."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


class ManifestError(ValueError):
    """Raised when a dataset manifest violates the public contract."""


COMMON_REQUIRED_FIELDS = ("image", "split")

TASK_REQUIRED_FIELDS = {
    "classification": ("label",),
    "count_classification": ("label",),
    "dense_depth": ("depth",),
    "dense_segmentation": ("segmentation",),
}


def iter_jsonl_manifest(path: str | Path) -> Iterable[Mapping[str, Any]]:
    """Yield JSON objects from a JSONL manifest."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(record, Mapping):
                raise ManifestError(f"{path}:{line_number} must be a JSON object")
            yield record


def validate_manifest(
    path: str | Path,
    *,
    task_type: str,
    expected_split: str | None = None,
    require_nonempty: bool = True,
) -> int:
    """Validate a task manifest and return the number of records."""

    if task_type not in TASK_REQUIRED_FIELDS:
        raise ManifestError(f"unsupported task_type: {task_type}")
    required_fields: Sequence[str] = COMMON_REQUIRED_FIELDS + TASK_REQUIRED_FIELDS[task_type]
    count = 0
    for index, record in enumerate(iter_jsonl_manifest(path), start=1):
        count += 1
        missing = [field for field in required_fields if field not in record]
        if missing:
            raise ManifestError(f"record {index} missing required fields: {', '.join(missing)}")
        if expected_split is not None and record["split"] != expected_split:
            raise ManifestError(
                f"record {index} split {record['split']!r} does not match {expected_split!r}"
            )
        _require_string(record["image"], f"record {index}.image")
        _require_string(record["split"], f"record {index}.split")
        if task_type in {"classification", "count_classification"}:
            if not isinstance(record["label"], (int, str)):
                raise ManifestError(f"record {index}.label must be int or string")
        if task_type == "dense_depth":
            _require_string(record["depth"], f"record {index}.depth")
        if task_type == "dense_segmentation":
            _require_string(record["segmentation"], f"record {index}.segmentation")
    if require_nonempty and count == 0:
        raise ManifestError(f"manifest is empty: {path}")
    return count


def _require_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{name} must be a non-empty string")
