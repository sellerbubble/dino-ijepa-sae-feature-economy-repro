#!/usr/bin/env python3
# Role: validate that public_repro is ready to export as a standalone public repo.
# Status: public release checker
# Used by: public CI and manual release checks
# Inputs: public_repro repository root
# Outputs: stdout checklist; exits nonzero on release-blocking issues
# Safe to move/delete?: keep; this guards the public release boundary.
# Notes: This checks repository hygiene, not scientific metric correctness.

from __future__ import annotations

import argparse
import os
import re
import stat
import sys
from pathlib import Path


REQUIRED_FILES = [
    "README.md",
    "pyproject.toml",
    ".gitignore",
    ".github/workflows/ci.yml",
    "scripts/reproduce_smoke.sh",
    "scripts/audit_artifact_export_manifest.py",
    "scripts/build_tiny_artifact_bundle.sh",
    "scripts/create_artifact_bundle_readme.py",
    "scripts/copy_artifact_export_manifest.py",
    "scripts/create_artifact_export_manifest.py",
    "scripts/export_torchscript_toy_feature_module.py",
    "scripts/export_public_repo.sh",
    "scripts/filter_reproduction_run_plan.py",
    "scripts/package_artifact_bundle.sh",
    "scripts/prepare_real_artifact_slice.py",
    "scripts/sanitize_artifact_bundle_metadata.py",
    "scripts/stage_artifact_bundle_from_manifest.sh",
    "docs/public_v1_scope.md",
    "docs/release_quickstart.md",
    "docs/canonical_chain_runbook.md",
    "docs/release_artifact_bundle_layout.md",
    "docs/first_real_saved_array_bundle_plan.md",
    "docs/reproduction_status.md",
    "docs/public_release_audit_20260513.md",
    "configs/models/dino_v2_base.yaml",
    "configs/models/ijepa_vit_h14.yaml",
    "configs/saes/dino_l11_topk32_exp4.yaml",
    "configs/saes/ijepa_l31_topk32_exp4.yaml",
    "configs/tasks/imagenet.yaml",
    "configs/tasks/nyuv2.yaml",
    "configs/tasks/ade20k.yaml",
    "configs/tasks/clevr_count.yaml",
]

REQUIRED_EXECUTABLES = [
    "scripts/reproduce_smoke.sh",
    "scripts/audit_artifact_export_manifest.py",
    "scripts/build_tiny_artifact_bundle.sh",
    "scripts/create_artifact_bundle_readme.py",
    "scripts/copy_artifact_export_manifest.py",
    "scripts/create_artifact_export_manifest.py",
    "scripts/export_torchscript_toy_feature_module.py",
    "scripts/export_public_repo.sh",
    "scripts/filter_reproduction_run_plan.py",
    "scripts/package_artifact_bundle.sh",
    "scripts/prepare_real_artifact_slice.py",
    "scripts/sanitize_artifact_bundle_metadata.py",
    "scripts/stage_artifact_bundle_from_manifest.sh",
]

PRIVATE_PATH_PATTERNS = [
    re.compile(r"/Users/"),
    re.compile(r"/mnt/workspace"),
    re.compile(r"/mnt/workspace-"),
    re.compile(r"/mnt/workspace1"),
    re.compile(r"liwenhao"),
    re.compile(r"claude_code"),
]

PRIVATE_PATH_ALLOWLIST = {
    "scripts/check_public_release.py",
    "src/feature_economy/configs/loader.py",
    "docs/release_artifact_bundle_layout.md",
    "tests/test_config_loader.py",
}

TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check public_repro release readiness.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="public_repro root directory.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    failures: list[str] = []

    failures.extend(_check_required_files(root))
    failures.extend(_check_executables(root))
    failures.extend(_check_markdown_fences(root))
    failures.extend(_check_private_paths(root))
    failures.extend(_check_readme_links(root))

    if failures:
        print("Public release check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Public release check passed.")
    return 0


def _check_required_files(root: Path) -> list[str]:
    failures = []
    for relative in REQUIRED_FILES:
        if not (root / relative).is_file():
            failures.append(f"missing required file: {relative}")
    return failures


def _check_executables(root: Path) -> list[str]:
    failures = []
    for relative in REQUIRED_EXECUTABLES:
        path = root / relative
        if not path.exists():
            continue
        mode = path.stat().st_mode
        if not mode & stat.S_IXUSR:
            failures.append(f"script is not user-executable: {relative}")
    return failures


def _check_markdown_fences(root: Path) -> list[str]:
    failures = []
    for path in sorted(root.rglob("*.md")):
        if _is_ignored(path):
            continue
        text = path.read_text(encoding="utf-8")
        if text.count("```") % 2:
            failures.append(f"unbalanced markdown fences: {_relative(root, path)}")
    return failures


def _check_private_paths(root: Path) -> list[str]:
    failures = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or _is_ignored(path) or path.suffix not in TEXT_SUFFIXES:
            continue
        relative = _relative(root, path)
        if relative in PRIVATE_PATH_ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in PRIVATE_PATH_PATTERNS:
            if pattern.search(text):
                failures.append(f"private path marker {pattern.pattern!r} in {relative}")
                break
    return failures


def _check_readme_links(root: Path) -> list[str]:
    readme = root / "README.md"
    if not readme.exists():
        return []
    text = readme.read_text(encoding="utf-8")
    expected_links = [
        "docs/public_v1_scope.md",
        "docs/release_quickstart.md",
        "docs/canonical_chain_runbook.md",
        "docs/release_artifact_bundle_layout.md",
        "docs/first_real_saved_array_bundle_plan.md",
        "docs/export_torchscript_backbones.md",
    ]
    return [f"README missing link to {link}" for link in expected_links if link not in text]


def _is_ignored(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts & {"__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache"})


def _relative(root: Path, path: Path) -> str:
    return os.fspath(path.relative_to(root))


if __name__ == "__main__":
    raise SystemExit(main())
