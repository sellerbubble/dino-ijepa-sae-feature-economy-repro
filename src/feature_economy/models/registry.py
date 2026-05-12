"""Config-backed model and SAE registry.

This registry intentionally does not instantiate DINO, I-JEPA, or SAE modules.
It validates and resolves the metadata future loaders need: model identifiers,
layers, transform policy, SAE checkpoint placeholders, and model-SAE linkage.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from feature_economy.configs import (
    load_yaml,
    validate_all_configs,
    validate_model_config,
    validate_sae_config,
)

from .transforms import transform_policy_from_config


class RegistryError(ValueError):
    """Raised when model/SAE registry metadata is inconsistent."""


PLACEHOLDER_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ModelRegistry:
    """Load model and SAE config metadata from `configs/`."""

    def __init__(self, config_root: str | Path) -> None:
        self.config_root = Path(config_root)
        validate_all_configs(self.config_root)
        self.models = _load_records(self.config_root / "models", validate_model_config)
        self.saes = _load_records(self.config_root / "saes", validate_sae_config)
        self._validate_links()

    def get_model(self, model_id: str) -> dict[str, Any]:
        try:
            return self.models[model_id]
        except KeyError as exc:
            raise RegistryError(f"unknown model id: {model_id}") from exc

    def get_sae(self, sae_id: str) -> dict[str, Any]:
        try:
            return self.saes[sae_id]
        except KeyError as exc:
            raise RegistryError(f"unknown SAE id: {sae_id}") from exc

    def sae_checkpoint_path(
        self,
        sae_id: str,
        *,
        env: dict[str, str] | None = None,
        require_resolved: bool = False,
    ) -> str:
        sae = self.get_sae(sae_id)
        return resolve_placeholders(
            str(sae["checkpoint"]),
            env=env,
            require_resolved=require_resolved,
        )

    def describe_model_sae_pair(self, model_id: str, sae_id: str) -> dict[str, Any]:
        model = self.get_model(model_id)
        sae = self.get_sae(sae_id)
        if sae["model_id"] != model_id:
            raise RegistryError(
                f"SAE {sae_id} belongs to {sae['model_id']}, not requested model {model_id}"
            )
        return {
            "model_id": model_id,
            "model_display_name": model["display_name"],
            "sae_id": sae_id,
            "layer": sae["layer"],
            "normalize_activations": sae["normalize_activations"],
            "transform": transform_policy_from_config(dict(model["transform"])),
            "token_format": model["output"]["token_format"],
        }

    def _validate_links(self) -> None:
        for sae_id, sae in self.saes.items():
            model_id = sae["model_id"]
            if model_id not in self.models:
                raise RegistryError(f"SAE {sae_id} references unknown model {model_id}")
            default_layers = self.models[model_id]["default_layers"]
            if sae["layer"] not in set(default_layers.values()):
                # Non-default layers may still be valid in future, but public v0
                # configs should be explicit before runner ports depend on them.
                raise RegistryError(
                    f"SAE {sae_id} layer {sae['layer']} is not listed in model {model_id} default_layers"
                )


def resolve_placeholders(
    value: str,
    *,
    env: dict[str, str] | None = None,
    require_resolved: bool = False,
) -> str:
    """Resolve `${VAR}` placeholders in config strings."""

    env = env if env is not None else os.environ

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in env:
            if require_resolved:
                raise RegistryError(f"missing environment variable for placeholder: {name}")
            return match.group(0)
        return env[name]

    return PLACEHOLDER_PATTERN.sub(replace, value)


def _load_records(config_dir: Path, validator: Any) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(config_dir.glob("*.yaml")):
        record = dict(load_yaml(path))
        validator(record)
        record_id = record["id"]
        if record_id in records:
            raise RegistryError(f"duplicate config id {record_id} in {config_dir}")
        records[record_id] = record
    return records
