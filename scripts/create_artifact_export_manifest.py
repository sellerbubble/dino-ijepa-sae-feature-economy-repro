#!/usr/bin/env python3
# Role: create a source-to-public artifact export manifest template from a run plan.
# Status: public release utility
# Used by: maintainers when preparing the first real saved-array bundle
# Inputs: reproduction_run_plan.json
# Outputs: CSV and optional JSON export manifest template
# Safe to move/delete?: keep; this is the bridge from private artifacts to public bundle rows.
# Notes: This script records placeholders only; it does not copy private artifacts.

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES_BY_STAGE = {
    "feature_extraction": [
        "features.npz",
        "feature_extraction_summary.json",
        "run_manifest.json",
    ],
    "sae_code_extraction": [
        "codes.npz",
        "sae_code_summary.json",
        "run_manifest.json",
    ],
    "native_probe": [
        "native_probe_summary.json",
        "probe_logits.npz",
        "run_manifest.json",
    ],
    "sae_probe": [
        "sae_probe_summary.json",
        "probe_logits.npz",
        "run_manifest.json",
    ],
    "availability": [
        "availability_summary.json",
        "run_manifest.json",
    ],
    "feature_ranking": [
        "task_feature_ranking.json",
        "run_manifest.json",
    ],
    "contribution_scores": [
        "contribution_scores.npz",
        "contribution_scores_summary.json",
        "run_manifest.json",
    ],
    "subset_usage": [
        "subset_usage_summary.json",
        "run_manifest.json",
    ],
    "feature_ablation": [
        "feature_ablation_summary.json",
        "run_manifest.json",
    ],
}

CSV_COLUMNS = [
    "row_id",
    "export_status",
    "stage",
    "experiment_id",
    "task_id",
    "task_type",
    "model_id",
    "sae_id",
    "ranking_method",
    "command",
    "public_artifact_dir",
    "required_files",
    "source_artifact_dir",
    "source_artifact_note",
    "conversion_needed",
    "validation_command",
    "notes",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an artifact export manifest template from a run plan."
    )
    parser.add_argument("--run-plan-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    manifest = build_export_manifest(args.run_plan_json)
    write_csv(manifest, args.output_csv)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(manifest['rows'])} export manifest rows to {args.output_csv}")
    if args.output_json is not None:
        print(f"Wrote export manifest JSON to {args.output_json}")
    return 0


def build_export_manifest(run_plan_json: Path) -> dict[str, Any]:
    plan = json.loads(run_plan_json.read_text(encoding="utf-8"))
    if plan.get("record_type") != "reproduction_run_plan":
        raise ValueError("--run-plan-json must contain a reproduction_run_plan record")
    rows = []
    for index, row in enumerate(plan.get("rows", []), start=1):
        stage = row.get("stage", "")
        required_files = REQUIRED_FILES_BY_STAGE.get(stage, [])
        rows.append(
            {
                "row_id": f"row_{index:03d}",
                "export_status": "TODO",
                "stage": stage,
                "experiment_id": row.get("experiment_id", ""),
                "task_id": row.get("task_id", ""),
                "task_type": row.get("task_type", ""),
                "model_id": row.get("model_id", ""),
                "sae_id": row.get("sae_id", ""),
                "ranking_method": row.get("ranking_method", ""),
                "command": row.get("command", ""),
                "public_artifact_dir": row.get("artifact_dir", ""),
                "required_files": required_files,
                "source_artifact_dir": "",
                "source_artifact_note": "",
                "conversion_needed": "unknown",
                "validation_command": _validation_command(stage),
                "notes": "",
            }
        )
    return {
        "record_type": "artifact_export_manifest_template",
        "source_run_plan": str(run_plan_json),
        "num_rows": len(rows),
        "rows": rows,
    }


def write_csv(manifest: dict[str, Any], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in manifest["rows"]:
            csv_row = dict(row)
            csv_row["required_files"] = ";".join(row["required_files"])
            writer.writerow(csv_row)


def _validation_command(stage: str) -> str:
    if stage in {"feature_extraction", "sae_code_extraction", "native_probe", "sae_probe"}:
        return "validate-arrays where applicable; index-artifacts --require-valid"
    if stage == "contribution_scores":
        return "validate-arrays --kind contribution_scores; index-artifacts --require-valid"
    return "index-artifacts --require-valid"


if __name__ == "__main__":
    raise SystemExit(main())
