"""Check completeness of saved artifact bundles against a run plan."""

from __future__ import annotations

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


def check_artifact_bundle(
    *,
    run_plan_json: str | Path,
    artifact_root: str | Path,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Check whether planned artifact directories contain required files."""

    run_plan_json = Path(run_plan_json)
    artifact_root = Path(artifact_root)
    plan = json.loads(run_plan_json.read_text(encoding="utf-8"))
    if plan.get("record_type") != "reproduction_run_plan":
        raise ValueError("run_plan_json must contain a reproduction_run_plan record")
    rows = []
    for row in plan.get("rows", []):
        stage = row.get("stage", "")
        required_files = REQUIRED_FILES_BY_STAGE.get(stage)
        if required_files is None:
            required_files = []
        artifact_dir = _resolve_artifact_dir(row.get("artifact_dir", ""), artifact_root)
        missing = [
            filename for filename in required_files if not (artifact_dir / filename).exists()
        ]
        rows.append(
            {
                "stage": stage,
                "task_id": row.get("task_id", ""),
                "model_id": row.get("model_id", ""),
                "sae_id": row.get("sae_id", ""),
                "artifact_dir": str(artifact_dir),
                "required_files": required_files,
                "missing_files": missing,
                "complete": not missing,
            }
        )
    complete_rows = sum(1 for row in rows if row["complete"])
    report = {
        "record_type": "artifact_bundle_check",
        "run_plan": str(run_plan_json),
        "artifact_root": str(artifact_root),
        "num_rows": len(rows),
        "complete_rows": complete_rows,
        "missing_rows": len(rows) - complete_rows,
        "complete": complete_rows == len(rows),
        "rows": rows,
    }
    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _resolve_artifact_dir(template: str, artifact_root: Path) -> Path:
    if not template:
        return artifact_root
    resolved = template.replace("${ARTIFACT_ROOT}", str(artifact_root))
    return Path(resolved)
