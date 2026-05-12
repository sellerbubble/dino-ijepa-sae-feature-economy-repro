#!/usr/bin/env python
# Role: export a tiny TorchScript feature module that satisfies the public backbone contract.
# Status: public example utility
# Used by: docs/export_torchscript_backbones.md and smoke validation
# Inputs: output path and optional feature dimension
# Outputs: TorchScript `.pt` feature module
# Safe to move/delete?: keep or replace with a richer example when a real public I-JEPA loader is added.
# Notes: This is a toy contract demo, not a DINO/I-JEPA implementation.

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a tiny TorchScript feature module for contract smoke tests.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-dim", type=int, default=3)
    parser.add_argument("--crop", type=int, default=224)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.feature_dim <= 0:
        raise ValueError("--feature-dim must be positive")
    if args.crop <= 0:
        raise ValueError("--crop must be positive")
    torch = _import_torch()

    class TinyFeatureModule(torch.nn.Module):
        def __init__(self, feature_dim: int) -> None:
            super().__init__()
            weights = torch.arange(1, 3 * feature_dim + 1, dtype=torch.float32)
            self.register_buffer("projection", (weights.reshape(3, feature_dim) % 7) / 7.0)

        def forward(self, pixel_values):
            pooled = pixel_values.mean(dim=(2, 3))
            return pooled @ self.projection

    module = TinyFeatureModule(args.feature_dim).eval()
    example = torch.zeros((1, 3, args.crop, args.crop), dtype=torch.float32)
    traced = torch.jit.trace(module, example)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.output))
    print(f"Wrote TorchScript feature module to {args.output}")
    return 0


def _import_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "Exporting TorchScript feature modules requires torch. "
            "Install with `pip install -e '.[experiments]'`."
        ) from exc
    return torch


if __name__ == "__main__":
    raise SystemExit(main())
