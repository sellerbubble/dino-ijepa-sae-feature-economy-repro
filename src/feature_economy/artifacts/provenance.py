"""Small provenance helpers for public artifacts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def current_git_commit() -> str:
    """Return the current git commit when available.

    Public artifacts should record the code version that produced them. The
    environment override supports CI or packaged runs where `.git` is absent.
    """

    override = os.environ.get("FEATURE_ECONOMY_GIT_COMMIT")
    if override:
        return override
    for cwd in _candidate_git_dirs():
        commit = _git_rev_parse(cwd)
        if commit is not None:
            return commit
    return "unknown"


def _candidate_git_dirs() -> list[Path]:
    here = Path(__file__).resolve()
    candidates = [Path.cwd(), *here.parents]
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _git_rev_parse(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None
