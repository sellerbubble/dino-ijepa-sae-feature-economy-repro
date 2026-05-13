# Full Paper-Style Reproduction

This is the default public reproduction path for the DINO/I-JEPA SAE Feature
Economy project. It is intended for researchers who can provide datasets, model
checkpoints or exported feature modules, SAE checkpoints, and GPU resources.

The goal is to rerun the main paper chain from data and checkpoints through
feature extraction, SAE-code extraction, task probes, feature-economy analyses,
and paper-style tables. The expected result is close scientific agreement with
the paper, not bit-identical reproduction of private intermediate arrays.

## Reproduction Modes

| Mode | Purpose | Requires GPU? | Scientific result? |
| --- | --- | --- | --- |
| Full paper-style rerun | Default path for external researchers. Regenerate features, SAE codes, probes, and analyses. | Yes | Yes |
| Smoke / tiny bundle | Code-health and command-surface validation. | No | No |
| Saved-array audit bundle | Optional inspection of archived paper-used intermediate arrays. | No after download | Audit only |

## Required Inputs

Prepare these resources before running the full chain:

| Input | Expected form | Notes |
| --- | --- | --- |
| Dataset manifests | JSONL files documented in `docs/datasets.md` | One row per image/sample, with stable split metadata. |
| Dense targets | `.npz` files for NYUv2/ADE20K targets when using saved-array probes | Keep target order aligned with the manifest. |
| DINO checkpoint | HuggingFace-compatible checkpoint or equivalent feature extractor | Configured by `configs/models/dino_v2_base.yaml`. |
| I-JEPA checkpoint | Exported TorchScript feature module or externally generated `features.npz` | See `docs/export_torchscript_backbones.md`. |
| SAE checkpoints | Public lightweight `.npz` checkpoints or convertible full checkpoints | See `docs/checkpoints.md`. |
| GPU environment | CUDA-capable environment with PyTorch and optional HuggingFace dependencies | Check with `feature-economy check-runtime --profile experiments`. |

The public repo does not redistribute datasets, model weights, or SAE weights.

## Canonical Chain

The full paper-style chain is:

```text
validate configs/manifests/runtime
  -> extract native backbone features
  -> train/evaluate native probes with task-appropriate readout pooling
  -> convert/load SAE checkpoints
  -> extract SAE codes
  -> train/evaluate SAE-code probes with task-appropriate readout pooling
  -> compute Availability
  -> rank task-recruited features and compute Access
  -> run Allocation ablations
  -> index artifacts
  -> export paper-style tables and figures
```

The preferred executable entrypoint is `scripts/run_full_profile.sh`. The
detailed manual command template remains available in
`docs/canonical_chain_runbook.md`.

## Recommended First Full Profile

Start with one model/task/SAE profile before running the full matrix:

```text
model: dino_v2_base
task: imagenet_1k validation
layer/SAE: dino_l11_topk32_exp4
stages: feature extraction -> SAE codes -> SAE probe -> Availability -> Access -> Allocation
```

This profile exercises the same artifact contracts as the full matrix while
keeping debugging manageable.

Preview the complete command chain without running GPU work:

```bash
export DATA_ROOT=/path/to/manifests
export SAE_ROOT=/path/to/sae_checkpoints
export ARTIFACT_ROOT=/path/to/output_artifacts
export PYTHONPATH=$PWD/src

DRY_RUN=1 bash scripts/run_full_profile.sh dino_imagenet_l11
```

Run the profile on real data and checkpoints:

```bash
export DATA_ROOT=/path/to/manifests
export SAE_ROOT=/path/to/sae_checkpoints
export ARTIFACT_ROOT=/path/to/output_artifacts
export PYTHONPATH=$PWD/src

bash scripts/run_full_profile.sh dino_imagenet_l11
```

On offline clusters, point the HuggingFace backend at a pre-downloaded DINO
checkpoint directory and force local loading:

```bash
export DINO_HF_NAME_OR_PATH=/path/to/facebook/dinov2-base
export LOCAL_FILES_ONLY=1

bash scripts/run_full_profile.sh dino_imagenet_l11
```

This first public launcher is intentionally conservative: it starts with one
vertical slice and preserves the same artifact layout expected by the broader
paper-style chain. Additional model, layer, task, and SAE profiles should extend
this launcher or add sibling profiles rather than creating unrelated one-off
scripts.

For classification-style tasks, the public linear probe mean-pools token maps
over non-feature axes before fitting the closed-form readout. This keeps the
readout dimension equal to the native hidden dimension or SAE feature dimension,
which is required for feature ranking and ablation to remain channel-level
analyses rather than token-by-channel flattened analyses.

## Output Layout

Use one artifact root per run:

```text
$ARTIFACT_ROOT/
  run_plan/
  manifests/
  features/
  checkpoints/
  codes/
  probes/
  analysis/
  index/
  tables/
  figures/
  logs/
```

Do not write full rerun outputs into the repository checkout. Use an external
artifact directory with enough disk space.

## Acceptance Criteria

A full rerun is successful when:

- `check-runtime --profile experiments --require` passes on the target machine.
- `check-configs` and `check-manifest` pass for the configured task.
- Feature and code arrays pass `validate-arrays`.
- Native and SAE-code probe summaries are generated for the configured task.
- Availability, Access, and Allocation artifacts are generated and indexed.
- `index-artifacts --require-valid` passes for the artifact root.
- `make-tables` exports the expected CSV tables.
- Results match the qualitative patterns and approximate ranges in
  `docs/expected_results.md`.

## What To Report

Every full rerun should record:

- git commit of the public repo;
- dataset name, split, and manifest path;
- model checkpoint or feature-module path;
- SAE checkpoint path and layer;
- transform config;
- probe recipe and random seed;
- GPU type, CUDA/PyTorch versions, and command line;
- artifact root and log directory;
- deviations from the reference configs.

## Relationship To The Audit Bundle

The saved-array release bundle is optional. Use it when you want to inspect or
compare against archived paper-used intermediate arrays. Do not treat it as the
main reproduction path for new experiments.
