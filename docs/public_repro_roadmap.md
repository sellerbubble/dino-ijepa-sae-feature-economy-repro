# Public Reproduction Roadmap

Date: 2026-05-13

This roadmap defines the staged path from the current fixture-based public
skeleton to an executable reproduction repo for the DINO/I-JEPA SAE Feature
Economy paper chain.

The current release boundary is defined in `docs/public_v1_scope.md`: public v1
is a full paper-style rerun repo with smoke tests for code health and optional
saved-array audit support. Private one-off cluster launchers remain out of
scope unless they can be converted into portable public runners.

The next upgrade is tracked in `docs/paper_faithful_upgrade_plan.md`. That plan
keeps lightweight probes as smoke/CI diagnostics while moving the default full
rerun path toward paper-scale trainers, robustness sweeps, layer sweeps, and
advanced native-space controls.

## End State

External readers should be able to reproduce the main paper chain:

```text
dataset manifest
  -> native backbone feature extraction
  -> SAE-code extraction
  -> native/SAE-code probe artifacts
  -> availability/access/allocation analyses
  -> paper tables and figures
```

The same structure should also support adding new models, layers, tasks, SAEs,
and analysis modules without copying one-off scripts.

## Phase 0: Public Skeleton And Contracts

Status: implemented for the current public smoke scope.

Goal:

- Keep the repo executable without private datasets or checkpoints.
- Define stable config, manifest, artifact, and table contracts.
- Provide a one-command smoke chain that validates the public command surface.

Current evidence:

- `scripts/reproduce_smoke.sh`
- `feature-economy check-runtime --profile smoke`
- `feature-economy check-configs`
- `feature-economy check-models`
- `feature-economy check-manifest`
- `feature-economy validate-arrays`
- `feature-economy index-artifacts --require-valid`
- `feature-economy make-tables`
- `feature-economy make-figures` when the optional `figures` extra is installed.

Completion criteria:

- The smoke script runs on a clean local checkout with only base dependencies.
- Artifact schemas reject malformed public outputs.
- Documentation clearly separates smoke outputs from scientific results.

## Phase 1: Real Data Entry Gate

Status: started.

Goal:

- Validate that real dataset manifests resolve to loadable images.
- Apply the shared DINO/I-JEPA transform policy before any backbone code is
  introduced.
- Keep this gate independent of torch and checkpoint loading.

Current evidence:

- `feature-economy inspect-images`
- `ManifestDataset.image_path()`
- PIL/NumPy implementation of resize, center crop, RGB conversion, and
  ImageNet normalization.

Completion criteria:

- Each paper task has documented train/validation manifest expectations.
- `inspect-images` succeeds on a tiny real-manifest sample for ImageNet,
  NYUv2, ADE20K, and CLEVR/Count.
- Relative-path and absolute-path manifest behavior is documented.

Remaining risk:

- Dense tasks currently use explicit `targets.npz` arrays. Direct raw
  depth-map/mask loading can be added later if public dataset adapters should
  own target preprocessing.

## Phase 2: Backbone Feature Extraction

Status: started.

Goal:

- Replace deterministic fixture features with real DINO/I-JEPA hidden states.
- Preserve the existing `feature_extraction_summary.json` and `features.npz`
  output contract.

Required design choices:

- Model loader boundary: HuggingFace DINOv2, local or HuggingFace I-JEPA.
- Token contract: CLS/register/patch handling for DINO and patch-token handling
  for I-JEPA.
- Layer selection: last and second-last first; extra layers via config.
- Device/batch controls: CLI flags, not hardcoded cluster paths.

Completion criteria:

- A tiny real-image extraction run writes valid feature artifacts for DINO and
  I-JEPA.
- The output artifact index validates with `--require-valid`.
- The command can run on CPU/GPU with explicit `--device` and `--batch-size`.

Current evidence:

- `feature-economy extract-features --backend huggingface` supports
  HuggingFace-hosted DINO-style backbones with explicit layer, batch, device,
  dtype, and local-cache controls.
- `feature-economy extract-features --backend torchscript` supports exported
  local feature modules, including I-JEPA feature extractors that accept
  normalized `NCHW` image tensors and return the target-layer features.
- `docs/export_torchscript_backbones.md` documents the feature-module contract
  and a toy export path.
- `scripts/reproduce_smoke.sh` exercises toy TorchScript export, extraction,
  array validation, and artifact indexing when torch/Pillow are available.

Remaining risk:

- The repo still does not include a universal raw I-JEPA checkpoint loader for
  every upstream checkpoint format. Use `IJEPA_HF_NAME_OR_PATH` for
  transformers-compatible local checkpoints or export a TorchScript feature
  module when needed.
- The HuggingFace backend has passed real-slice validation for the default
  DINO/I-JEPA public profiles, but users should still run `DRY_RUN=1` and
  `check-runtime --profile experiments --require` on their own machines before
  launching full-data jobs.

## Phase 3: SAE Code Extraction

Status: implemented for saved-array lightweight probes.

Goal:

- Replace deterministic fixture codes with real ImageNet-trained SAE
  activations.
- Preserve `sae_code_summary.json` and `codes.npz`.

Required design choices:

- Lightweight inference-only SAE loader.
- Runtime normalization support, especially `layer_norm`.
- Explicit decoder/encoder shape validation against model feature dimension.

Completion criteria:

- Real feature arrays can be encoded through DINO and I-JEPA SAEs.
- SAE code artifacts validate and can feed downstream probe/analysis commands.
- Failure messages identify missing checkpoint, dimension mismatch, or
  unsupported normalization.

Current evidence:

- `feature-economy extract-sae-codes --backend linear-topk` supports a
  lightweight `.npz` SAE checkpoint with `encoder_weight`, optional
  `encoder_bias`, and optional `decoder_bias`.
- The backend applies configured runtime normalization, decoder-bias centering,
  ReLU, and TopK sparsification.
- `feature-economy convert-sae-checkpoint` converts standard `W_enc`/`b_enc`/
  `b_dec` checkpoints into the public lightweight format.

Remaining risk:

- Gated SAE checkpoints are rejected by default and need a dedicated public
  backend if they become part of the reproduction target.

## Phase 4: Native And SAE-Code Probes

Status: started.

Goal:

- Train/evaluate native and SAE-code probes for ImageNet-1K, NYUv2, ADE20K, and
  CLEVR/Count using the same metric functions as fixture tests.

Completion criteria:

- Each task has a canonical config and command.
- Probe summaries distinguish native backbone probe, SAE-code probe, and
  no-ablation baseline where relevant.
- Seeds and split assumptions are recorded in run manifests.

Current evidence:

- `feature-economy probe-native --backend linear-probe` trains/evaluates a
  closed-form ridge probe on saved native features for
  classification/counting/depth/segmentation tasks.
- `feature-economy probe-sae --backend linear-probe` trains/evaluates a
  closed-form ridge probe on saved SAE codes for
  classification/counting/depth/segmentation tasks.

Remaining risk:

- Paper-scale probe hyperparameters need a torch training backend rather than
  only the lightweight closed-form classifier.

## Phase 5: AAA Analyses

Status: started.

Goal:

- Reproduce availability, access, and allocation artifacts from saved SAE codes
  and probe outputs.

Completion criteria:

- Availability: fired-count/rate distributions and bucket assignments.
- Access: task-selected top-k and matched random subset usage.
- Allocation: feature/bucket ablation summaries with matched random controls.
- Analysis outputs are table-ready through the artifact index.

Current evidence:

- `feature-economy compute-usage` reads saved `codes.npz` and writes a valid
  `availability_summary.json` with fired, dead, bucket, and active-quantile
  statistics.
- `feature-economy rank-features` reads saved SAE codes and linear-probe
  weights to write a valid probe-weight `task_feature_ranking.json`.
- `feature-economy compute-subset-usage` reads saved SAE codes and a ranking
  artifact, then writes a valid `subset_usage_summary.json` comparing top-k
  task-selected features against a fired-count bucket-matched random control.
- `feature-economy ablate-features --backend linear-probe` zeroes selected SAE
  channels, recomputes a saved linear probe, and writes a valid
  `feature_ablation_summary.json` with a matched random control for
  classification/counting/depth/segmentation saved arrays.
- `feature-economy probe-native/probe-sae --backend linear-probe` now supports
  explicit dense `targets.npz` inputs for NYUv2-style depth and ADE20K-style
  segmentation saved feature/code arrays.

Remaining risk:

- Dense depth/segmentation allocation now has a lightweight saved-array backend,
  but paper-scale reproduction still needs task-specific settings and expected
  artifact preparation.
- Dense-task usage-rate conventions still need explicit real-artifact support
  if paper reproduction requires per-pixel rates rather than fired counts.

## Phase 6: Paper Tables And Figures

Status: table export implemented for indexed artifacts; lightweight overview
figure export implemented from CSV tables.

Goal:

- Generate the paper-facing tables and figures from public artifacts only.

Completion criteria:

- One command builds all main paper tables from an artifact index.
- One command builds overview figures from public CSV tables.
- Figure scripts consume public tables, not private notebook state.
- Every number in a table or figure can be traced to a command, config, and
  artifact path.

Remaining risk:

- Current figures are reproducibility overviews, not final manuscript-designed
  figures.

## Operating Rules

- Do not move private workbench scripts into `public_repro` directly.
- Port one vertical slice at a time.
- Keep fixture mode for every new public runner.
- Add a test before relying on a new artifact contract.
- Keep private absolute paths out of configs and docs.
