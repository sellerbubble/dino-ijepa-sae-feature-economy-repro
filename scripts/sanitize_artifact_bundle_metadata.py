#!/usr/bin/env python3
# Role: sanitize private absolute paths from public artifact bundle metadata.
# Status: public release utility
# Used by: maintainers before packaging saved-array release assets
# Inputs: artifact bundle root and optional OLD=NEW replacement rules
# Outputs: rewritten JSON/CSV/Markdown metadata files plus an optional report
# Safe to move/delete?: keep; this is a release-safety gate for public bundles.
# Notes: This rewrites metadata only. It never edits .npz arrays or model/data files.

from __future__ import annotations

import argparse
import json
from pathlib import Path


TEXT_SUFFIXES = {".json", ".csv", ".md", ".txt"}
PRIVATE_PATH_MARKERS = ("".join(["/Users", "/"]), "".join(["/mnt", "/workspace"]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replace private absolute path prefixes in artifact metadata files, "
            "or scan metadata for private paths when no replacement rules are given."
        )
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--replace",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help=(
            "Replacement rule. Can be passed multiple times. If omitted, the "
            "script runs in scan-only mode."
        ),
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--require-no-private-paths",
        action="store_true",
        help="Fail if sanitized metadata still contains private absolute path markers.",
    )
    args = parser.parse_args()

    replacements = [_parse_replacement(rule) for rule in args.replace]
    report = sanitize_metadata(
        artifact_root=args.artifact_root,
        replacements=replacements,
        require_no_private_paths=args.require_no_private_paths,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "Sanitized artifact metadata: "
        f"{report['changed_files']} changed files, "
        f"{report['remaining_private_path_hits']} remaining private-path hits."
    )
    return 0


def sanitize_metadata(
    *,
    artifact_root: Path,
    replacements: list[tuple[str, str]],
    require_no_private_paths: bool,
) -> dict[str, object]:
    artifact_root = artifact_root.resolve()
    if not artifact_root.is_dir():
        raise SystemExit(f"artifact root does not exist: {artifact_root}")
    scanned_files = 0
    changed: list[str] = []
    remaining_hits: list[dict[str, object]] = []
    for path in sorted(_metadata_files(artifact_root)):
        scanned_files += 1
        text = path.read_text(encoding="utf-8")
        rewritten = text
        for old, new in replacements:
            rewritten = rewritten.replace(old, new)
        if rewritten != text:
            path.write_text(rewritten, encoding="utf-8")
            changed.append(str(path.relative_to(artifact_root)))
        hits = _private_path_lines(rewritten)
        if hits:
            remaining_hits.append(
                {
                    "path": str(path.relative_to(artifact_root)),
                    "hits": hits,
                }
            )

    if require_no_private_paths and remaining_hits:
        examples = "\n".join(
            f"{row['path']}:{row['hits'][0]['line_number']}: {row['hits'][0]['line']}"
            for row in remaining_hits[:10]
        )
        raise SystemExit(f"private paths remain after sanitization:\n{examples}")

    return {
        "record_type": "artifact_metadata_sanitization_report",
        "artifact_root": "${ARTIFACT_ROOT}",
        "artifact_root_name": artifact_root.name,
        "scanned_files": scanned_files,
        "changed_files": len(changed),
        "changed_paths": changed,
        "remaining_private_path_hits": sum(len(row["hits"]) for row in remaining_hits),
        "remaining_hit_files": remaining_hits,
        "replacement_count": len(replacements),
        "replacements": [{"old": "<redacted>", "new": new} for old, new in replacements],
    }


def _metadata_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and (path.suffix in TEXT_SUFFIXES or path.name == "README.md"):
            yield path


def _private_path_lines(text: str) -> list[dict[str, object]]:
    rows = []
    for index, line in enumerate(text.splitlines(), start=1):
        if any(marker in line for marker in PRIVATE_PATH_MARKERS):
            rows.append({"line_number": index, "line": line.strip()})
    return rows


def _parse_replacement(rule: str) -> tuple[str, str]:
    if "=" not in rule:
        raise SystemExit(f"replacement must have OLD=NEW form: {rule}")
    old, new = rule.split("=", 1)
    if not old:
        raise SystemExit(f"replacement OLD side cannot be empty: {rule}")
    return old, new


if __name__ == "__main__":
    raise SystemExit(main())
