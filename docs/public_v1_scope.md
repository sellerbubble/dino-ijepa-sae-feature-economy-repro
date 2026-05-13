# Public V1 Scope

Date: 2026-05-13

This document fixes the intended boundary for the first public release of the
DINO/I-JEPA SAE Feature Economy reproduction repo.

## Release Position

Public v1 is a **lightweight executable reproduction repo** for the paper's main
evidence chain. It is designed to let external readers prepare datasets and
checkpoints, generate their own features and SAE codes, run the Feature Economy
analyses, and compare the resulting patterns against documented expected
results.

It is not intended to be a full private-cluster recreation of every paper-scale
training loop. Large saved-array bundles may be published as optional audit
artifacts, but they are not the default reproduction path.

## What Public V1 Must Support

Public v1 should support the following chain:

```text
configs and manifests
  -> image/manifest validation
  -> native feature extraction
  -> lightweight SAE code extraction
  -> native and SAE-code linear probes
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
- equivalence to private probe-training loops;
- final manuscript figure reproduction;
- redistribution of datasets, model weights, or SAE checkpoints;
- support for every private artifact variant produced during research;
- mandatory download of paper-scale feature arrays or SAE codes.

These omissions are release-boundary decisions, not hidden TODOs. They keep the
first public repo executable, auditable, and portable.

## Required Public Gates

A public v1 checkout should provide these gates:

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

## Exact Trainer Escalation Rule

Do not add a paper-scale torch probe trainer to public v1 merely to reduce a
wording caveat. Add it only if all of the following are true:

- the required training hyperparameters are frozen and documented;
- expected input arrays, target arrays, and output artifacts remain compatible
  with the current contracts;
- the trainer has a tiny fixture/smoke mode;
- the trainer can run without private cluster paths;
- the added dependency burden is acceptable for external users;
- the README and roadmap clearly distinguish lightweight and paper-scale modes.

Until those conditions are met, public v1 should treat paper-scale probe
training as a future extension and keep the reproducible path centered on
regenerating features/codes plus lightweight probes.

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
