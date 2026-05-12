"""Manifest-backed dataset adapters.

These adapters intentionally stop at metadata records. They do not load images,
depth maps, or segmentation masks yet. That boundary keeps public smoke tests
lightweight while giving future probe runners a stable data interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .manifests import iter_jsonl_manifest, validate_manifest


@dataclass(frozen=True)
class ManifestRecord:
    """One validated manifest row."""

    image: str
    split: str
    label: int | str | None = None
    depth: str | None = None
    segmentation: str | None = None
    extra: dict[str, Any] | None = None


class ManifestDataset:
    """A small metadata-only dataset backed by a JSONL manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        task_type: str,
        expected_split: str | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.task_type = task_type
        self.expected_split = expected_split
        validate_manifest(
            self.manifest_path,
            task_type=task_type,
            expected_split=expected_split,
        )
        self.records = [_to_record(row) for row in iter_jsonl_manifest(self.manifest_path)]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ManifestRecord:
        return self.records[index]

    def resolve_path(self, value: str) -> Path:
        """Resolve manifest-relative paths against the manifest directory."""

        path = Path(value)
        if path.is_absolute():
            return path
        return self.manifest_path.parent / path

    def image_path(self, index: int) -> Path:
        """Return the resolved image path for a record."""

        return self.resolve_path(self.records[index].image)


def _to_record(row: dict[str, Any]) -> ManifestRecord:
    known = {"image", "split", "label", "depth", "segmentation"}
    extra = {key: value for key, value in row.items() if key not in known}
    return ManifestRecord(
        image=row["image"],
        split=row["split"],
        label=row.get("label"),
        depth=row.get("depth"),
        segmentation=row.get("segmentation"),
        extra=extra or None,
    )
