#!/usr/bin/env python3
# Role: generate a README.md for a public saved-array artifact bundle.
# Status: public release utility
# Used by: maintainers before packaging public artifact release assets
# Inputs: reproduction run plan and optional release manifest metadata
# Outputs: artifact bundle README.md
# Safe to move/delete?: keep; bundle README is part of the public release contract.
# Notes: This writes documentation only; it does not validate scientific metrics.

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an artifact bundle README.")
    parser.add_argument("--run-plan-json", type=Path, required=True)
    parser.add_argument("--output-readme", type=Path, required=True)
    parser.add_argument("--bundle-name", required=True)
    parser.add_argument("--paper-version", default="DINO/I-JEPA SAE Feature Economy paper draft")
    parser.add_argument("--public-repo-commit", default="unknown")
    parser.add_argument("--release-manifest-json", type=Path)
    parser.add_argument(
        "--figures-included",
        action="store_true",
        help="State that generated overview figures are included.",
    )
    args = parser.parse_args()

    readme = build_readme(
        run_plan_json=args.run_plan_json,
        bundle_name=args.bundle_name,
        paper_version=args.paper_version,
        public_repo_commit=args.public_repo_commit,
        release_manifest_json=args.release_manifest_json,
        figures_included=args.figures_included,
    )
    args.output_readme.parent.mkdir(parents=True, exist_ok=True)
    args.output_readme.write_text(readme, encoding="utf-8")
    print(f"Wrote artifact bundle README to {args.output_readme}")
    return 0


def build_readme(
    *,
    run_plan_json: Path,
    bundle_name: str,
    paper_version: str,
    public_repo_commit: str,
    release_manifest_json: Path | None,
    figures_included: bool,
) -> str:
    plan = json.loads(run_plan_json.read_text(encoding="utf-8"))
    if plan.get("record_type") != "reproduction_run_plan":
        raise ValueError("--run-plan-json must contain a reproduction_run_plan record")
    rows = plan.get("rows", [])
    stage_counts = Counter(row.get("stage", "") for row in rows)
    tasks = sorted({row.get("task_id", "") for row in rows if row.get("task_id")})
    models = sorted({row.get("model_id", "") for row in rows if row.get("model_id")})
    saes = sorted({row.get("sae_id", "") for row in rows if row.get("sae_id")})
    release_manifest = _load_release_manifest(release_manifest_json)

    lines = [
        f"# {bundle_name}",
        "",
        f"Release date: {date.today().isoformat()}",
        "",
        f"Paper/version: {paper_version}",
        "",
        f"Public reproduction repo commit: `{public_repo_commit}`",
        "",
        "## Scope",
        "",
        "This artifact bundle contains saved-array and JSON/CSV artifacts for the",
        "DINO/I-JEPA SAE Feature Economy public reproduction path. It is designed",
        "for artifact-first reproduction of the analysis chain, not redistribution",
        "of raw datasets, model weights, or private cluster launchers.",
        "",
        "## Contents",
        "",
        f"- Planned rows: `{len(rows)}`",
        f"- Models: `{', '.join(models) if models else 'none'}`",
        f"- SAEs: `{', '.join(saes) if saes else 'none'}`",
        f"- Tasks/splits: `{', '.join(tasks) if tasks else 'none'}`",
        f"- Generated overview figures included: `{str(figures_included).lower()}`",
        "",
        "### Stage Counts",
        "",
        "| Stage | Rows |",
        "| --- | ---: |",
    ]
    for stage, count in sorted(stage_counts.items()):
        lines.append(f"| `{stage}` | {count} |")
    lines.extend(
        [
            "",
            "## Directory Layout",
            "",
            "```text",
            f"{bundle_name}/",
            "  run_plan/",
            "  manifests/",
            "  features/",
            "  codes/",
            "  probes/",
            "  analysis/",
            "  index/",
            "  tables/",
            "  figures/        # optional",
            "  README.md",
            "```",
            "",
            "## Verification",
            "",
            "From the standalone public reproduction repo root:",
            "",
            "```bash",
            f"export ARTIFACT_ROOT=/path/to/{bundle_name}",
            "export PYTHONPATH=$PWD/src",
            "",
            "python -m feature_economy.cli.main check-bundle \\",
            "  --run-plan-json \"$ARTIFACT_ROOT/run_plan/reproduction_run_plan.json\" \\",
            "  --artifact-root \"$ARTIFACT_ROOT\" \\",
            "  --output-json \"$ARTIFACT_ROOT/run_plan/artifact_bundle_check_final.json\" \\",
            "  --require-complete",
            "",
            "python -m feature_economy.cli.main index-artifacts \\",
            "  --input-dir \"$ARTIFACT_ROOT\" \\",
            "  --output-json \"$ARTIFACT_ROOT/index/artifact_index.json\" \\",
            "  --output-csv \"$ARTIFACT_ROOT/index/artifact_index.csv\" \\",
            "  --require-valid",
            "",
            "python -m feature_economy.cli.main make-tables \\",
            "  --artifact-index \"$ARTIFACT_ROOT/index/artifact_index.json\" \\",
            "  --output-dir \"$ARTIFACT_ROOT/tables\"",
            "```",
            "",
            "If figures are included or desired:",
            "",
            "```bash",
            "python -m feature_economy.cli.main make-figures \\",
            "  --table-dir \"$ARTIFACT_ROOT/tables\" \\",
            "  --output-dir \"$ARTIFACT_ROOT/figures\" \\",
            "  --formats png,pdf",
            "```",
            "",
            "## Checksum And Unpacking",
            "",
        ]
    )
    if release_manifest:
        archive = release_manifest.get("archive", "<archive>.tar.gz")
        checksum = release_manifest.get("sha256", "<sha256>")
        lines.extend(
            [
                f"- Archive: `{archive}`",
                f"- SHA256: `{checksum}`",
                "",
                "```bash",
                f"sha256sum -c {archive}.sha256",
                f"tar -xzf {archive}",
                "```",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "After packaging, verify the release asset with:",
                "",
                "```bash",
                f"sha256sum -c {bundle_name}.tar.gz.sha256",
                f"tar -xzf {bundle_name}.tar.gz",
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Dataset, Checkpoint, And License Notes",
            "",
            "This bundle does not include restricted raw datasets, model weights, or SAE",
            "checkpoints unless a separate release note explicitly says otherwise. Users",
            "must obtain datasets and checkpoints according to their original licenses.",
            "",
            "Saved features, SAE codes, probe outputs, rankings, and ablation summaries",
            "are provided as reproduction artifacts for the paper analysis chain.",
            "",
            "## Known Caveats",
            "",
            "- Public v1 is an artifact-first saved-array reproduction.",
            "- It is not an exact private-cluster rerun of every paper-scale trainer.",
            "- If this is a subset bundle, it validates only the rows present in its",
            "  `run_plan/reproduction_run_plan.json`.",
            "- Final manuscript figure styling may differ from generated overview figures.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_release_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
