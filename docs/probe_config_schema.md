# Probe Config Schema

Date: 2026-05-13

This document defines the public config surface for paper-scale probe trainers.
It is the bridge between the private paper runners and the public
agent-friendly reproduction system.

## Why Probe Configs Exist

The public repo keeps two probe modes:

- `lightweight_closed_form`: fast diagnostics for smoke tests and CI.
- `paper_scale_torch`: the reference path for full paper-style reruns.

Probe configs describe the paper-scale recipe without hardcoding machine paths
or forcing every user to use the same batch size on every GPU. The configs are
also the handoff contract for future agents: add a config before adding another
runner.

## Required Fields

Every `configs/probes/*.yaml` record must contain:

| Field | Meaning |
| --- | --- |
| `id` | Stable probe recipe id. |
| `task_id` | Task config consumed by this probe. |
| `probe_family` | One of `classification`, `count_classification`, `dense_depth`, `dense_segmentation`. |
| `supported_input_spaces` | Usually `[native, sae_code]`; separates backbone probes from SAE-code probes. |
| `backend` | `paper_scale_torch`, `lightweight_closed_form`, or `fixture`. |
| `status` | `draft_recipe`, `ported`, `validated`, or `deprecated`. |
| `training` | Epochs, batch size, optimizer, learning rate, and related defaults. |
| `selection` | Checkpoint-selection rule, e.g. best validation top-1 or lowest RMSE. |
| `metrics` | Primary and reported metrics. |
| `outputs` | Required filenames written by the trainer. |

The validator rejects obvious private absolute paths. Relative private
workbench source names may appear only as provenance notes while the recipe is
being ported.

## Recommended Optional Fields

Use these fields when they apply:

- `paper_reference`: private source runners and audit notes used to derive the
  recipe.
- `readout`: classifier or decoder architecture.
- `transform`: preprocessing policy.
- `target`: dense target preprocessing rules.
- `labels`: classification/count label construction rules.
- `ranking`: whether SAE-code outputs are suitable for task-feature ranking.
- `notes`: caveats that must survive into paper-facing reports.

## Output Contract

Paper-scale trainers should write the same core files as lightweight public
probes, with richer metadata:

| File | Purpose |
| --- | --- |
| `summary.json` | Probe identity, backend, split, seed, metrics, checkpoint rule, and checkpoint path. |
| `probe.pt` | Best paper-scale readout checkpoint. |
| `probe_outputs.npz` | Predictions/logits or dense outputs for validation and downstream analysis. |
| `task_feature_ranking.json` | Required for SAE-code probes that feed Access/Allocation. |
| `run_manifest.json` | Command, config ids, input artifacts, output artifacts, git commit, and environment notes. |

The public port may also emit compatibility files such as `probe_logits.npz`
when needed by existing lightweight ranking/ablation utilities. If both names
exist, the summary must identify which one is authoritative.

## Status Lifecycle

Use `status` consistently:

| Status | Meaning |
| --- | --- |
| `draft_recipe` | Derived from private audit, not yet implemented as public trainer. |
| `ported` | Public code can run the recipe on fixtures or small real slices. |
| `validated` | Public code has passed real-slice or full-data validation and expected-results checks. |
| `deprecated` | Kept for provenance but not used by new full reruns. |

Do not mark a config `validated` just because `check-configs` passes. Config
validation only proves the recipe is well-formed.

## Current Draft Recipes

The first draft paper-scale recipes are:

- `configs/probes/imagenet_paper_scale.yaml`
- `configs/probes/nyuv2_depth_paper_scale.yaml`
- `configs/probes/ade20k_segmentation_paper_scale.yaml`
- `configs/probes/clevr_count_paper_scale.yaml`

They are intentionally marked `draft_recipe`. The next implementation step is
to port the torch training backends and then update these statuses as evidence
accumulates.

## Implementation Status

As of 2026-05-13, the public codebase includes an initial
`paper-scale-torch` backend for classification-style, dense-depth, and
dense-segmentation saved arrays:

- native classification/counting probes over `features.npz`;
- SAE-code classification/counting probes over `codes.npz`;
- native dense-depth probes over feature maps plus `targets.npz`;
- SAE-code dense-depth probes over code maps plus `targets.npz`;
- native dense-segmentation probes over feature maps plus `targets.npz`;
- SAE-code dense-segmentation probes over code maps plus `targets.npz`;
- AdamW training for multiple epochs;
- best validation checkpoint selection;
- `probe.pt`, `probe_outputs.npz`, compatibility `probe_logits.npz`, and
  `run_manifest.json` outputs.

All four task families now have a public saved-array `paper-scale-torch`
trainer. The full-profile launcher can switch defaults after the command
templates and runbooks are updated to provide the required train/validation
arrays and dense targets.
