#!/usr/bin/env python3
# Role: split a large artifact archive into GitHub-release-sized parts.
# Status: public release utility
# Used by: maintainers after package_artifact_bundle.sh emits a large archive
# Inputs: archive path, optional part size, optional overwrite flag
# Outputs: archive.part-XX files and archive.parts.sha256
# Safe to move/delete?: keep; this prevents manual release-asset splitting drift.
# Notes: The script verifies that concatenated part hashes match the source archive.

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split a packaged artifact archive into checksummed release parts."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--part-size",
        default="1900M",
        help="Maximum part size. Supports raw bytes or K/M/G suffixes. Default: 1900M.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing part files.")
    args = parser.parse_args()

    archive = args.archive.resolve()
    if not archive.is_file():
        raise SystemExit(f"archive does not exist: {archive}")
    part_size = parse_size(args.part_size)
    parts = split_archive(archive=archive, part_size=part_size, force=args.force)
    parts_sha = write_parts_sha256(archive, parts)
    verify_reassembled_hash(archive, parts)
    print(f"Split archive into {len(parts)} parts:")
    for part in parts:
        print(f"  {part.name} ({part.stat().st_size} bytes)")
    print(f"Wrote part checksums: {parts_sha}")
    return 0


def parse_size(value: str) -> int:
    value = value.strip()
    if not value:
        raise SystemExit("part size cannot be empty")
    suffix = value[-1].upper()
    multiplier = 1
    number = value
    if suffix in {"K", "M", "G"}:
        number = value[:-1]
        multiplier = {"K": 1024, "M": 1024**2, "G": 1024**3}[suffix]
    try:
        size = int(float(number) * multiplier)
    except ValueError as exc:
        raise SystemExit(f"invalid part size: {value}") from exc
    if size <= 0:
        raise SystemExit("part size must be positive")
    return size


def split_archive(*, archive: Path, part_size: int, force: bool) -> list[Path]:
    existing_parts = sorted(archive.parent.glob(f"{archive.name}.part-*"))
    checksum_path = archive.with_name(f"{archive.name}.parts.sha256")
    if (existing_parts or checksum_path.exists()) and not force:
        raise SystemExit("part files already exist; rerun with --force to replace them")
    for path in existing_parts:
        path.unlink()
    if checksum_path.exists():
        checksum_path.unlink()

    parts: list[Path] = []
    with archive.open("rb") as source:
        index = 0
        while True:
            chunk = source.read(part_size)
            if not chunk:
                break
            part = archive.with_name(f"{archive.name}.part-{index:02d}")
            part.write_bytes(chunk)
            parts.append(part)
            index += 1
    if not parts:
        raise SystemExit(f"archive is empty: {archive}")
    return parts


def write_parts_sha256(archive: Path, parts: list[Path]) -> Path:
    checksum_path = archive.with_name(f"{archive.name}.parts.sha256")
    rows = []
    for part in parts:
        rows.append(f"{sha256_file(part)}  {part.name}\n")
    checksum_path.write_text("".join(rows), encoding="utf-8")
    return checksum_path


def verify_reassembled_hash(archive: Path, parts: list[Path]) -> None:
    source_hash = sha256_file(archive)
    digest = hashlib.sha256()
    for part in parts:
        with part.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    if digest.hexdigest() != source_hash:
        raise SystemExit("reassembled part hash does not match source archive")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
