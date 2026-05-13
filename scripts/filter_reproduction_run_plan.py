#!/usr/bin/env python3
# Role: filter a reproduction run plan into a publishable subset plan.
# Status: public release utility
# Used by: maintainers when staging partial saved-array artifact releases
# Inputs: reproduction_run_plan.json and optional row filters
# Outputs: filtered reproduction_run_plan JSON and optional CSV
# Safe to move/delete?: keep; partial releases should use this instead of hand-editing JSON.
# Notes: Availability rows use task_id like imagenet_1k_val and may need explicit stage filters.

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "stage",
    "experiment_id",
    "task_id",
    "task_type",
    "model_id",
    "sae_id",
    "ranking_method",
    "command",
    "artifact_dir",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter a reproduction run plan.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--sae-id", action="append", default=[])
    parser.add_argument("--stage", action="append", default=[])
    parser.add_argument("--experiment-id", action="append", default=[])
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write an empty plan instead of failing when no rows match.",
    )
    args = parser.parse_args()

    plan = filter_run_plan(
        args.input_json,
        model_ids=set(args.model_id),
        task_ids=set(args.task_id),
        sae_ids=set(args.sae_id),
        stages=set(args.stage),
        experiment_ids=set(args.experiment_id),
    )
    if not plan["rows"] and not args.allow_empty:
        raise SystemExit("No run-plan rows matched filters. Use --allow-empty if intentional.")
    write_plan(plan, output_json=args.output_json, output_csv=args.output_csv)
    print(f"Wrote {plan['num_rows']} filtered run-plan rows to {args.output_json}")
    if args.output_csv is not None:
        print(f"Wrote filtered run-plan CSV to {args.output_csv}")
    return 0


def filter_run_plan(
    input_json: Path,
    *,
    model_ids: set[str],
    task_ids: set[str],
    sae_ids: set[str],
    stages: set[str],
    experiment_ids: set[str],
) -> dict[str, Any]:
    plan = json.loads(input_json.read_text(encoding="utf-8"))
    if plan.get("record_type") != "reproduction_run_plan":
        raise ValueError("--input-json must contain a reproduction_run_plan record")
    rows = [
        row
        for row in plan.get("rows", [])
        if _matches(row, "model_id", model_ids)
        and _matches(row, "task_id", task_ids)
        and _matches(row, "sae_id", sae_ids, allow_empty_field=True)
        and _matches(row, "stage", stages)
        and _matches(row, "experiment_id", experiment_ids)
    ]
    return {
        "record_type": "reproduction_run_plan",
        "source_run_plan": str(input_json),
        "config_root": plan.get("config_root", ""),
        "num_rows": len(rows),
        "stages": sorted({row["stage"] for row in rows}),
        "rows": rows,
    }


def write_plan(
    plan: dict[str, Any],
    *,
    output_json: Path,
    output_csv: Path | None = None,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for row in plan["rows"]:
                writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def _matches(
    row: dict[str, Any],
    field: str,
    allowed: set[str],
    *,
    allow_empty_field: bool = False,
) -> bool:
    if not allowed:
        return True
    value = str(row.get(field, ""))
    if allow_empty_field and value == "":
        return "" in allowed
    return value in allowed


if __name__ == "__main__":
    raise SystemExit(main())
