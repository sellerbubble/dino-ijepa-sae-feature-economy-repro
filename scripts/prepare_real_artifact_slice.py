#!/usr/bin/env python3
# Role: create a maintainer workspace for staging a real saved-array artifact slice.
# Status: public release utility
# Used by: maintainers before inspecting private artifacts or filling export manifests
# Inputs: public configs plus a supported slice profile
# Outputs: full/subset run plans and a pre-statused artifact export manifest
# Safe to move/delete?: keep; this prevents hand-edited run plans during real releases.
# Notes: This script does not read private paths and does not copy artifacts.

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from feature_economy.configs.plan import write_reproduction_plan  # noqa: E402


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

MANIFEST_COLUMNS = [
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
    "native_subspace_ablation": [
        "native_subspace_ablation_summary.json",
        "run_manifest.json",
    ],
}

DEFAULT_TRIAL_STAGE_STATUS = {
    "feature_extraction": "TODO",
    "sae_code_extraction": "TODO",
    "native_probe": "NEEDS_CONVERSION",
    "sae_probe": "NEEDS_CONVERSION",
    "availability": "NEEDS_CONVERSION",
    "feature_ranking": "NEEDS_CONVERSION",
    "contribution_scores": "TODO",
    "subset_usage": "NEEDS_CONVERSION",
    "feature_ablation": "NEEDS_CONVERSION",
    "native_subspace_ablation": "TODO",
}

DEFAULT_TRIAL_STAGE_NOTES = {
    "feature_extraction": "Check whether saved native features already exist remotely; otherwise regenerate public features.npz.",
    "sae_code_extraction": "Check whether saved SAE codes exist remotely; otherwise regenerate from public features.npz.",
    "native_probe": "Private probe summary likely exists, but public probe_logits.npz/run_manifest.json may need conversion.",
    "sae_probe": "Private SAE probe summary likely exists, but public probe_logits.npz/run_manifest.json may need conversion.",
    "availability": "Convert fired-count/usage stats into public availability_summary.json.",
    "feature_ranking": "Convert existing task_feature_ranking.json and add public run_manifest.json, or recompute from public codes/probe outputs.",
    "contribution_scores": "Recompute from public SAE probe outputs unless a compatible score array is found.",
    "subset_usage": "Convert combined summary or recompute from public codes/ranking.",
    "feature_ablation": "Convert combined summary or recompute from public codes/ranking/probe logits.",
    "native_subspace_ablation": "Run public Module F command from saved native features, lightweight SAE checkpoint, ranking, and SAE probe logits.",
}

BASE_PAPER_EXPERIMENT_IDS = {
    "paper_v0_native",
    "paper_v0_sae",
    "paper_v0_availability",
}


def _trial_profile(
    *,
    description: str,
    model_ids: set[str],
    task_ids: set[str],
    sae_ids: set[str],
    stage_notes: dict[str, str] | None = None,
    experiment_ids: set[str] | None = None,
    stages: set[str] | None = None,
) -> dict[str, Any]:
    notes = dict(DEFAULT_TRIAL_STAGE_NOTES)
    if stage_notes:
        notes.update(stage_notes)
    return {
        "description": description,
        "model_ids": model_ids,
        "task_ids": task_ids,
        "sae_ids": sae_ids,
        "stages": set(stages or set()),
        "experiment_ids": set(experiment_ids or BASE_PAPER_EXPERIMENT_IDS),
        "stage_status": dict(DEFAULT_TRIAL_STAGE_STATUS),
        "stage_notes": notes,
    }


PROFILE_FILTERS = {
    "dino_imagenet_v1_trial": _trial_profile(
        description="First real vertical slice: DINOv2-B/14 + ImageNet-1K + DINO L11 SAE.",
        model_ids={"dino_v2_base"},
        task_ids={"imagenet_1k", "imagenet_1k_val"},
        sae_ids={"", "dino_l11_topk32_exp4"},
    ),
    "ijepa_imagenet_v1_trial": _trial_profile(
        description="First real vertical slice: I-JEPA ViT-H/14 + ImageNet-1K + I-JEPA L31 SAE.",
        model_ids={"ijepa_vit_h14"},
        task_ids={"imagenet_1k", "imagenet_1k_val"},
        sae_ids={"", "ijepa_l31_topk32_exp4"},
    ),
    "imagenet_v1_trial": _trial_profile(
        description="Combined ImageNet-1K public slice for DINOv2-B/14 and I-JEPA ViT-H/14.",
        model_ids={"dino_v2_base", "ijepa_vit_h14"},
        task_ids={"imagenet_1k", "imagenet_1k_val"},
        sae_ids={"", "dino_l11_topk32_exp4", "ijepa_l31_topk32_exp4"},
    ),
    "nyuv2_v1_trial": _trial_profile(
        description="NYUv2 dense-depth public slice for DINOv2-B/14 and I-JEPA ViT-H/14.",
        model_ids={"dino_v2_base", "ijepa_vit_h14"},
        task_ids={"nyuv2_depth"},
        sae_ids={"", "dino_l11_topk32_exp4", "ijepa_l31_topk32_exp4"},
    ),
    "dino_nyuv2_v1_trial": _trial_profile(
        description="Single-model NYUv2 dense-depth public slice for DINOv2-B/14.",
        model_ids={"dino_v2_base"},
        task_ids={"nyuv2_depth"},
        sae_ids={"", "dino_l11_topk32_exp4"},
    ),
    "ijepa_nyuv2_v1_trial": _trial_profile(
        description="Single-model NYUv2 dense-depth public slice for I-JEPA ViT-H/14.",
        model_ids={"ijepa_vit_h14"},
        task_ids={"nyuv2_depth"},
        sae_ids={"", "ijepa_l31_topk32_exp4"},
    ),
    "ade20k_v1_trial": _trial_profile(
        description="ADE20K segmentation public slice for DINOv2-B/14 and I-JEPA ViT-H/14.",
        model_ids={"dino_v2_base", "ijepa_vit_h14"},
        task_ids={"ade20k_segmentation"},
        sae_ids={"", "dino_l11_topk32_exp4", "ijepa_l31_topk32_exp4"},
    ),
    "dino_ade20k_v1_trial": _trial_profile(
        description="Single-model ADE20K segmentation public slice for DINOv2-B/14.",
        model_ids={"dino_v2_base"},
        task_ids={"ade20k_segmentation"},
        sae_ids={"", "dino_l11_topk32_exp4"},
    ),
    "ijepa_ade20k_v1_trial": _trial_profile(
        description="Single-model ADE20K segmentation public slice for I-JEPA ViT-H/14.",
        model_ids={"ijepa_vit_h14"},
        task_ids={"ade20k_segmentation"},
        sae_ids={"", "ijepa_l31_topk32_exp4"},
    ),
    "clevr_count_v1_trial": _trial_profile(
        description="CLEVR/Count image-only counting public slice for DINOv2-B/14 and I-JEPA ViT-H/14.",
        model_ids={"dino_v2_base", "ijepa_vit_h14"},
        task_ids={"clevr_count"},
        sae_ids={"", "dino_l11_topk32_exp4", "ijepa_l31_topk32_exp4"},
    ),
    "dino_clevr_count_v1_trial": _trial_profile(
        description="Single-model CLEVR/Count public slice for DINOv2-B/14.",
        model_ids={"dino_v2_base"},
        task_ids={"clevr_count"},
        sae_ids={"", "dino_l11_topk32_exp4"},
    ),
    "ijepa_clevr_count_v1_trial": _trial_profile(
        description="Single-model CLEVR/Count public slice for I-JEPA ViT-H/14.",
        model_ids={"ijepa_vit_h14"},
        task_ids={"clevr_count"},
        sae_ids={"", "ijepa_l31_topk32_exp4"},
    ),
    "full_v1_template": {
        "description": "Full public v1 saved-array release template, including ranking-control and default layer-sweep rows.",
        "model_ids": set(),
        "task_ids": set(),
        "sae_ids": set(),
        "stages": set(),
        "experiment_ids": set(),
        "stage_status": {},
        "stage_notes": {},
    },
    "module_f_nyuv2_v1_trial": _trial_profile(
        description="Module F NYUv2 native-subspace ablation slice for final and second-last layers.",
        model_ids={"dino_v2_base", "ijepa_vit_h14"},
        task_ids={"nyuv2_depth"},
        sae_ids={
            "dino_l11_topk32_exp4",
            "ijepa_l31_topk32_exp4",
            "dino_l10_topk32_exp4",
            "ijepa_l30_topk32_exp4",
        },
        stages={"native_subspace_ablation"},
        experiment_ids={"module_f_native_ablation_nyuv2"},
    ),
    "layer_sweep_v1_trial": _trial_profile(
        description="Default layer-sweep diagnostic slice: second-last and representative ImageNet-1K availability rows.",
        model_ids={"dino_v2_base", "ijepa_vit_h14"},
        task_ids={"imagenet_1k_val"},
        sae_ids={
            "",
            "dino_l3_topk32_exp4",
            "dino_l7_topk32_exp4",
            "dino_l9_topk32_exp4",
            "dino_l10_topk32_exp4",
            "dino_l11_topk32_exp4",
            "ijepa_l8_topk32_exp4",
            "ijepa_l20_topk32_exp4",
            "ijepa_l30_topk32_exp4",
            "ijepa_l31_topk32_exp4",
        },
        experiment_ids={"layer_sweep:second_last", "layer_sweep:representative"},
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare run plans and export manifest for a real artifact release slice."
    )
    parser.add_argument("--config-root", type=Path, default=Path("configs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_FILTERS),
        default="dino_imagenet_v1_trial",
        help="Release-slice profile to materialize.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = PROFILE_FILTERS[args.profile]

    full_plan_json = output_dir / "full_reproduction_run_plan.json"
    full_plan_csv = output_dir / "full_reproduction_run_plan.csv"
    full_plan = write_reproduction_plan(
        config_root=args.config_root,
        output_json=full_plan_json,
        output_csv=full_plan_csv,
    )

    subset_plan = _filter_plan(full_plan, profile)
    subset_plan_json = output_dir / "reproduction_run_plan.json"
    subset_plan_csv = output_dir / "reproduction_run_plan.csv"
    _write_plan(subset_plan, subset_plan_json, subset_plan_csv)

    manifest = _build_manifest(subset_plan, profile, subset_plan_json)
    manifest_csv = output_dir / "artifact_export_manifest.csv"
    manifest_json = output_dir / "artifact_export_manifest.json"
    _write_manifest(manifest, manifest_csv, manifest_json)

    readme = _workspace_readme(args.profile, profile, subset_plan, output_dir)
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Prepared real artifact slice workspace: {output_dir}")
    print(f"Profile: {args.profile}")
    print(f"Rows: {subset_plan['num_rows']} / {full_plan['num_rows']} full-plan rows")
    print(f"Run plan: {subset_plan_json}")
    print(f"Export manifest: {manifest_csv}")
    return 0


def _filter_plan(plan: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in plan["rows"]
        if _matches(row, "model_id", profile["model_ids"])
        and _matches(row, "task_id", profile["task_ids"])
        and _matches(row, "sae_id", profile["sae_ids"])
        and _matches(row, "stage", profile["stages"])
        and _matches(row, "experiment_id", profile["experiment_ids"])
    ]
    if not rows:
        raise SystemExit("Profile produced an empty run plan; check configs and filters.")
    return {
        "record_type": "reproduction_run_plan",
        "source_run_plan": str(plan.get("source_run_plan", "")),
        "config_root": plan.get("config_root", ""),
        "profile": profile["description"],
        "num_rows": len(rows),
        "stages": sorted({row["stage"] for row in rows}),
        "rows": rows,
    }


def _matches(row: dict[str, Any], field: str, allowed: set[str]) -> bool:
    return not allowed or str(row.get(field, "")) in allowed


def _write_plan(plan: dict[str, Any], output_json: Path, output_csv: Path) -> None:
    output_json.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in plan["rows"]:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def _build_manifest(
    plan: dict[str, Any],
    profile: dict[str, Any],
    source_run_plan: Path,
) -> dict[str, Any]:
    rows = []
    for index, row in enumerate(plan["rows"], start=1):
        stage = row["stage"]
        status = profile["stage_status"].get(stage, "TODO")
        rows.append(
            {
                "row_id": f"row_{index:03d}",
                "export_status": status,
                "stage": stage,
                "experiment_id": row.get("experiment_id", ""),
                "task_id": row.get("task_id", ""),
                "task_type": row.get("task_type", ""),
                "model_id": row.get("model_id", ""),
                "sae_id": row.get("sae_id", ""),
                "ranking_method": row.get("ranking_method", ""),
                "command": row.get("command", ""),
                "public_artifact_dir": row.get("artifact_dir", ""),
                "required_files": REQUIRED_FILES_BY_STAGE.get(stage, []),
                "source_artifact_dir": "",
                "source_artifact_note": "",
                "conversion_needed": "yes" if status == "NEEDS_CONVERSION" else "unknown",
                "validation_command": _validation_command(stage),
                "notes": profile["stage_notes"].get(stage, ""),
            }
        )
    return {
        "record_type": "artifact_export_manifest_template",
        "source_run_plan": str(source_run_plan),
        "num_rows": len(rows),
        "rows": rows,
    }


def _write_manifest(manifest: dict[str, Any], output_csv: Path, output_json: Path) -> None:
    output_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
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


def _workspace_readme(
    profile_name: str,
    profile: dict[str, Any],
    plan: dict[str, Any],
    output_dir: Path,
) -> str:
    stage_counts = {stage: 0 for stage in plan["stages"]}
    for row in plan["rows"]:
        stage_counts[row["stage"]] += 1
    stage_table = "\n".join(f"| `{stage}` | {count} |" for stage, count in stage_counts.items())
    return f"""# Real Artifact Slice Workspace

Profile: `{profile_name}`

{profile["description"]}

This directory is a maintainer staging workspace. It contains no private data and
does not mark any row as publishable by itself.

## Contents

- `full_reproduction_run_plan.json`: complete public v1 run matrix.
- `reproduction_run_plan.json`: filtered run plan for this slice.
- `artifact_export_manifest.csv`: source-to-public handoff sheet to fill after
  inspecting private artifacts.
- `artifact_export_manifest.json`: JSON copy of the same manifest.

## Stage Counts

| Stage | Rows |
| --- | ---: |
{stage_table}

## Next Commands

After filling `source_artifact_dir` for ready rows:

```bash
python scripts/audit_artifact_export_manifest.py \\
  --manifest-csv {output_dir}/artifact_export_manifest.csv \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --output-json {output_dir}/artifact_export_manifest_audit.json
```

When all rows intended for this slice are ready:

```bash
bash scripts/stage_artifact_bundle_from_manifest.sh \\
  --manifest-csv {output_dir}/artifact_export_manifest.csv \\
  --run-plan-json {output_dir}/reproduction_run_plan.json \\
  --artifact-root "$ARTIFACT_ROOT" \\
  --release-output-dir {output_dir}/release_assets \\
  --bundle-name feature_economy_{profile_name} \\
  --force
```
"""


if __name__ == "__main__":
    raise SystemExit(main())
