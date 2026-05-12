"""Runtime dependency checks for the public reproduction package."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DependencyStatus:
    """One optional dependency check result."""

    module: str
    available: bool
    version: str | None = None
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "available": self.available,
            "version": self.version,
            "error": self.error,
        }


RUNTIME_PROFILES = {
    "smoke": ("numpy", "yaml"),
    "base": ("numpy", "yaml"),
    "figures": ("numpy", "yaml", "matplotlib"),
    "experiments": ("numpy", "yaml", "torch", "torchvision", "transformers", "PIL", "tqdm"),
}


def check_runtime(profile: str = "smoke") -> list[DependencyStatus]:
    """Check importability of dependencies for a runtime profile."""

    if profile not in RUNTIME_PROFILES:
        raise ValueError(f"unknown runtime profile: {profile}")
    return [_check_module(module) for module in RUNTIME_PROFILES[profile]]


def require_runtime(profile: str = "smoke") -> list[DependencyStatus]:
    """Check a profile and raise if any dependency is missing."""

    statuses = check_runtime(profile)
    missing = [status for status in statuses if not status.available]
    if missing:
        names = ", ".join(status.module for status in missing)
        raise RuntimeError(f"missing dependencies for profile {profile}: {names}")
    return statuses


def write_runtime_report(
    statuses: list[DependencyStatus],
    output_path: str | Path,
    *,
    profile: str,
) -> Path:
    """Write a small JSON report suitable for CI logs and issue reports."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_type": "runtime_dependency_report",
        "schema_version": 1,
        "profile": profile,
        "all_available": all(status.available for status in statuses),
        "dependencies": [status.to_json() for status in statuses],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def _check_module(module: str) -> DependencyStatus:
    try:
        imported = importlib.import_module(module)
    except Exception as exc:  # pragma: no cover - error type depends on environment
        return DependencyStatus(module=module, available=False, error=f"{type(exc).__name__}: {exc}")
    version = getattr(imported, "__version__", None)
    return DependencyStatus(module=module, available=True, version=version)
