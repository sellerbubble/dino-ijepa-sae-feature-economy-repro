#!/usr/bin/env python3
# Role: audit a filled artifact export manifest before copying files.
# Status: public release utility
# Used by: maintainers when preparing real saved-array release bundles
# Inputs: artifact_export_manifest_template.csv with source_artifact_dir entries
# Outputs: JSON audit report and concise stdout summary
# Safe to move/delete?: keep; this prevents unsafe or incomplete public bundle exports.
# Notes: This is read-only. It never copies, deletes, or rewrites artifact files.

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


VALID_STATUSES = {"TODO", "READY", "COPIED", "OMIT", "NEEDS_CONVERSION"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit an artifact export manifest.")
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Optional public artifact root used to resolve public_artifact_dir templates.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit nonzero unless every non-OMIT row has source files ready.",
    )
    args = parser.parse_args()

    report = audit_manifest(args.manifest_csv, artifact_root=args.artifact_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(
        "Audited artifact export manifest: "
        f"{report['ready_rows']}/{report['actionable_rows']} actionable rows ready, "
        f"{report['omitted_rows']} omitted, {report['problem_rows']} problem rows."
    )
    print(f"Wrote audit report to {args.output_json}")

    if args.require_ready and not report["ready"]:
        return 1
    return 0


def audit_manifest(
    manifest_csv: Path,
    *,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    audited_rows = []
    for row in rows:
        audited_rows.append(_audit_row(row, artifact_root=artifact_root))

    actionable = [row for row in audited_rows if row["export_status"] != "OMIT"]
    ready_rows = sum(1 for row in actionable if row["ready"])
    problem_rows = sum(1 for row in audited_rows if row["problems"])
    report = {
        "record_type": "artifact_export_manifest_audit",
        "manifest_csv": str(manifest_csv),
        "artifact_root": str(artifact_root) if artifact_root is not None else "",
        "num_rows": len(audited_rows),
        "actionable_rows": len(actionable),
        "ready_rows": ready_rows,
        "omitted_rows": sum(1 for row in audited_rows if row["export_status"] == "OMIT"),
        "problem_rows": problem_rows,
        "ready": ready_rows == len(actionable) and problem_rows == 0,
        "rows": audited_rows,
    }
    return report


def _audit_row(row: dict[str, str], *, artifact_root: Path | None) -> dict[str, Any]:
    status = row.get("export_status", "").strip() or "TODO"
    source = row.get("source_artifact_dir", "").strip()
    required_files = _split_required_files(row.get("required_files", ""))
    problems: list[str] = []

    if status not in VALID_STATUSES:
        problems.append(f"invalid export_status {status!r}")
    if status == "OMIT":
        ready = not problems
    elif not source:
        problems.append("source_artifact_dir is empty")
        ready = False
    else:
        source_path = Path(source).expanduser()
        if not source_path.is_dir():
            problems.append("source_artifact_dir does not exist or is not a directory")
        missing = [filename for filename in required_files if not (source_path / filename).is_file()]
        problems.extend(f"missing required source file: {filename}" for filename in missing)
        ready = not problems and status in {"READY", "COPIED", "NEEDS_CONVERSION"}

    public_dir = row.get("public_artifact_dir", "")
    resolved_public_dir = _resolve_public_dir(public_dir, artifact_root)
    return {
        "row_id": row.get("row_id", ""),
        "export_status": status,
        "stage": row.get("stage", ""),
        "task_id": row.get("task_id", ""),
        "model_id": row.get("model_id", ""),
        "sae_id": row.get("sae_id", ""),
        "source_artifact_dir": source,
        "public_artifact_dir": public_dir,
        "resolved_public_artifact_dir": resolved_public_dir,
        "required_files": required_files,
        "ready": ready,
        "problems": problems,
    }


def _split_required_files(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def _resolve_public_dir(template: str, artifact_root: Path | None) -> str:
    if artifact_root is None:
        return template
    return template.replace("${ARTIFACT_ROOT}", str(artifact_root))


if __name__ == "__main__":
    raise SystemExit(main())
