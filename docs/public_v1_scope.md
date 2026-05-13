# Public V1 Scope

Date: 2026-05-13

This document fixes the intended boundary for the first public release of the
DINO/I-JEPA SAE Feature Economy reproduction repo.

## Release Position

Public v1 is a **full paper-style rerun repository** for the paper's main
evidence chain. It is designed to let external readers prepare datasets, model
checkpoints, SAE checkpoints, and GPU resources, then rerun the pipeline from
feature extraction through probes, feature-economy analyses, and paper-style
tables.

It is not intended to redistribute private data, checkpoints, or every private
cluster launcher. Smoke tests remain code-health checks. Large saved-array
bundles may be published as optional audit artifacts, but they are not the
default reproduction path.

## What Public V1 Must Support

Public v1 should support the following chain:

```text
configs and manifests
  -> image/manifest validation
  -> native feature extraction
  -> SAE code extraction
  -> native and SAE-code probes using paper-style settings when available
  -> Availability, Access, and Allocation analyses
  -> artifact indexing
  -> CSV tables and overview figures
  -> run-plan and bundle-completeness checks
```

The supported task families are:

- ImageNet-1K classification.
- NYUv2 dense depth.
- ADE20K semantic segmentation.
- CLEVR/Count image-only counting.

The supported model entry points are:

- DINO-style HuggingFace feature extraction through `extract-features --backend huggingface`.
- I-JEPA/local model feature extraction through exported TorchScript feature modules.
- Externally provided `features.npz` and `codes.npz` arrays that satisfy the
  public contracts, for users who prefer to cache or reuse intermediate arrays.

## What Public V1 Explicitly Does Not Claim

Public v1 does not claim:

- bit-identical reproduction of private paper-scale feature caches;
- an official raw I-JEPA checkpoint loader;
- equivalence to every private probe-training launcher until public paper-style
  runners are ported and validated;
- final manuscript figure reproduction;
- redistribution of datasets, model weights, or SAE checkpoints;
- support for every private artifact variant produced during research;
- mandatory download of paper-scale feature arrays or SAE codes.

These omissions are release-boundary decisions, not hidden TODOs. They keep the
first public repo executable, auditable, and portable.

## Required Public Gates

A public v1 checkout should provide these gates:

- `docs/full_reproduction.md`: default full paper-style rerun path.
- `docs/remote_safety.md`: non-destructive remote sync and artifact rules.
- `scripts/reproduce_smoke.sh`: one-command checkpoint-free smoke validation.
- `scripts/build_tiny_artifact_bundle.sh`: one complete tiny vertical slice
  whose final `check-bundle --require-complete` passes.
- `feature-economy check-runtime`: dependency inspection.
- `feature-economy check-configs`: portable config validation.
- `feature-economy plan-runs`: configured run-matrix export.
- `feature-economy check-bundle`: artifact completeness check against a run plan.
- `feature-economy check-manifest`: task manifest validation.
- `feature-economy validate-arrays`: saved-array contract validation.
- `feature-economy index-artifacts --require-valid`: artifact schema validation.
- `feature-economy make-tables` and `feature-economy make-figures`: public result exports.
- `docs/expected_results.md`: reference scores and qualitative acceptance bands.
- `docs/release_artifact_bundle_layout.md`: expected layout for optional
  saved-array audit assets.
- `scripts/check_public_release.py`: required-file, executable-bit, Markdown,
  README-link, and private-path hygiene checks.
- `.github/workflows/ci.yml`: standalone public-repo CI that runs unit tests,
  release hygiene, the smoke chain, and the tiny complete artifact bundle.

If a contribution changes the public reproduction chain, it should either update
these gates or explain why the new component is intentionally outside public v1.

## Paper-Style Runner Rule

Public paper-style runners should be added when the corresponding private
experiment path is understood well enough to expose it without private paths or
one-off assumptions. A runner is ready for public v1 only if all of the
following are true:

- the required training hyperparameters are frozen and documented;
- expected input arrays, target arrays, and output artifacts remain compatible
  with the current contracts;
- the runner has a tiny fixture, dry-run, or config-validation mode;
- the runner can run without private cluster paths;
- the added dependency burden is acceptable for external users;
- the README and roadmap clearly distinguish full rerun, smoke, and audit modes.

Until those conditions are met for a module, public v1 should expose the closest
validated artifact contract and clearly mark the module as not yet paper-style
complete.

## Extension Rule

New models, layers, SAEs, tasks, and analysis modules should enter through:

1. config updates;
2. artifact/schema contracts;
3. tiny tests or fixture smoke;
4. `plan-runs` coverage when the new component belongs to the main chain;
5. documentation of any new array or checkpoint contract.

Avoid copying private one-off scripts into public v1. If a private workflow is
needed, extract the smallest stable interface and keep the private launch logic
outside this release boundary.
