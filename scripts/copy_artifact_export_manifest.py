#!/usr/bin/env python3
# Role: copy ready artifact rows from a filled export manifest into a public bundle.
# Status: public release utility
# Used by: maintainers after audit_artifact_export_manifest.py reports ready rows
# Inputs: artifact export manifest CSV, artifact root
# Outputs: required files copied into public artifact directories; optional JSON copy report
# Safe to move/delete?: keep; this is the controlled file-copy gate for public bundles.
# Notes: Refuses overwrites by default and copies only required files, not whole source dirs.

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any


COPY_STATUSES = {"READY", "COPIED"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy ready artifact rows from an export manifest into an artifact root."
    )
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing destination files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report copy actions without writing files.",
    )
    parser.add_argument(
        "--include-needs-conversion",
        action="store_true",
        help="Also copy rows marked NEEDS_CONVERSION. Use only for staged conversion inputs.",
    )
    args = parser.parse_args()

    report = copy_ready_artifacts(
        manifest_csv=args.manifest_csv,
        artifact_root=args.artifact_root,
        force=args.force,
        dry_run=args.dry_run,
        include_needs_conversion=args.include_needs_conversion,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        "Copied artifact manifest rows: "
        f"{report['copied_files']} files across {report['copied_rows']} rows; "
        f"{report['skipped_rows']} skipped rows; {report['problem_rows']} problem rows."
    )
    if args.output_json is not None:
        print(f"Wrote copy report to {args.output_json}")
    return 1 if report["problem_rows"] else 0


def copy_ready_artifacts(
    *,
    manifest_csv: Path,
    artifact_root: Path,
    force: bool = False,
    dry_run: bool = False,
    include_needs_conversion: bool = False,
) -> dict[str, Any]:
    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    allowed_statuses = set(COPY_STATUSES)
    if include_needs_conversion:
        allowed_statuses.add("NEEDS_CONVERSION")

    copied_rows = []
    skipped_rows = []
    problem_rows = []
    copied_files = 0
    for row in rows:
        status = (row.get("export_status") or "TODO").strip()
        if status not in allowed_statuses:
            skipped_rows.append(_skip_record(row, reason=f"status {status!r} is not copyable"))
            continue
        record = _copy_row(row, artifact_root=artifact_root, force=force, dry_run=dry_run)
        if record["problems"]:
            problem_rows.append(record)
        else:
            copied_rows.append(record)
            copied_files += len(record["copied_files"])

    return {
        "record_type": "artifact_export_copy_report",
        "manifest_csv": str(manifest_csv),
        "artifact_root": str(artifact_root),
        "dry_run": dry_run,
        "force": force,
        "include_needs_conversion": include_needs_conversion,
        "num_rows": len(rows),
        "copied_rows": len(copied_rows),
        "skipped_rows": len(skipped_rows),
        "problem_rows": len(problem_rows),
        "copied_files": copied_files,
        "rows": copied_rows + problem_rows + skipped_rows,
    }


def _copy_row(
    row: dict[str, str],
    *,
    artifact_root: Path,
    force: bool,
    dry_run: bool,
) -> dict[str, Any]:
    source_dir = Path(row.get("source_artifact_dir", "")).expanduser()
    public_dir = _resolve_public_dir(row.get("public_artifact_dir", ""), artifact_root)
    required_files = _split_required_files(row.get("required_files", ""))
    copied_files: list[str] = []
    problems: list[str] = []

    if not source_dir.is_dir():
        problems.append("source_artifact_dir does not exist or is not a directory")
    for filename in required_files:
        source_file = source_dir / filename
        dest_file = public_dir / filename
        if not source_file.is_file():
            problems.append(f"missing required source file: {filename}")
            continue
        if dest_file.exists() and not force:
            problems.append(f"destination exists and --force was not set: {dest_file}")
            continue
        if not dry_run:
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, dest_file)
        copied_files.append(str(dest_file))

    return {
        "row_id": row.get("row_id", ""),
        "export_status": row.get("export_status", ""),
        "stage": row.get("stage", ""),
        "task_id": row.get("task_id", ""),
        "model_id": row.get("model_id", ""),
        "sae_id": row.get("sae_id", ""),
        "source_artifact_dir": str(source_dir),
        "resolved_public_artifact_dir": str(public_dir),
        "required_files": required_files,
        "copied_files": copied_files,
        "problems": problems,
    }


def _skip_record(row: dict[str, str], *, reason: str) -> dict[str, Any]:
    return {
        "row_id": row.get("row_id", ""),
        "export_status": row.get("export_status", ""),
        "stage": row.get("stage", ""),
        "task_id": row.get("task_id", ""),
        "model_id": row.get("model_id", ""),
        "sae_id": row.get("sae_id", ""),
        "copied_files": [],
        "problems": [],
        "skip_reason": reason,
    }


def _split_required_files(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def _resolve_public_dir(template: str, artifact_root: Path) -> Path:
    return Path(template.replace("${ARTIFACT_ROOT}", str(artifact_root)))


if __name__ == "__main__":
    raise SystemExit(main())
