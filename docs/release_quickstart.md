# Release Quickstart

This is the shortest path for checking that the public reproduction repo works
and for starting a full paper-style rerun from user-provided data, model
checkpoints, SAE checkpoints, and GPU resources. For the complete default path,
see `docs/full_reproduction.md`; for command details, see
`docs/canonical_chain_runbook.md`.

## 1. Five-Minute Smoke Check

From `public_repro/`:

```bash
python -m pip install -e '.[figures]'
OUTPUT_ROOT=/tmp/feature_economy_public_smoke bash scripts/reproduce_smoke.sh
```

This runs:

- unit tests;
- runtime, config, model, and manifest checks;
- fixture native/SAE probes;
- tiny dense saved-array probes;
- toy TorchScript feature-module export and extraction;
- feature/code/probe-logit array validation;
- Availability, Access, and Allocation smoke artifacts;
- artifact indexing;
- CSV table export;
- overview figure export when `matplotlib` is installed.

The smoke output is not a scientific result. It proves that the public command
surface and artifact contracts are executable.

## 2. Tiny Complete Artifact Bundle

To see one complete vertical slice with a final bundle-completeness check:

```bash
OUTPUT_ROOT=/tmp/feature_economy_tiny_bundle bash scripts/build_tiny_artifact_bundle.sh
```

This builds one DINO/ImageNet SAE slice from tiny fixture arrays, then verifies:

- initial `check-bundle` reports missing planned rows;
- feature/code/probe/AAA artifacts are generated;
- final `check-bundle --require-complete` reports all planned rows complete;
- `index-artifacts --require-valid` accepts the complete bundle;
- `make-tables` exports public CSV tables.

The tiny bundle is still not a scientific result. It is the smallest executable
example of the public v1 artifact contract.

## 3. Full Paper-Style Vertical Slice

The default public path regenerates intermediate arrays locally instead of
requiring a paper-scale feature-cache download. To reproduce the scientific
chain, provide real manifests, model checkpoints or exported feature modules,
SAE checkpoints, GPU resources, and manifest-referenced dense targets when
needed.

For expected score ranges and qualitative checks, see `docs/expected_results.md`.

The recommended entrypoint for the first full vertical slice is:

```bash
export DATA_ROOT=/path/to/manifests
export SAE_ROOT=/path/to/sae_checkpoints
export ARTIFACT_ROOT=/path/to/output_artifacts
export PYTHONPATH=$PWD/src

DRY_RUN=1 bash scripts/run_full_profile.sh dino_imagenet_l11
bash scripts/run_full_profile.sh dino_imagenet_l11
```

Use the dry run first to inspect every command and output path. The manual
commands below are the expanded version of the same chain for users who need to
customize individual stages.

If the GPU machine cannot reach HuggingFace, pre-download the DINO checkpoint
elsewhere and pass its local directory:

```bash
export DINO_HF_NAME_OR_PATH=/path/to/facebook/dinov2-base
export LOCAL_FILES_ONLY=1
bash scripts/run_full_profile.sh dino_imagenet_l11
```

The matched I-JEPA/ImageNet profile uses the same artifact contract. Provide
either a transformers-compatible checkpoint:

```bash
export IJEPA_HF_NAME_OR_PATH=/path/to/ijepa-vith14
export LOCAL_FILES_ONLY=1
bash scripts/run_full_profile.sh ijepa_imagenet_l31
```

or an exported TorchScript feature module:

```bash
export IJEPA_TORCHSCRIPT_CHECKPOINT=/path/to/ijepa_l31_feature_module.pt
bash scripts/run_full_profile.sh ijepa_imagenet_l31
```

For the classification vertical slice, contribution scoring defaults to the
scalable `true_class_logit_drop` method. To run a small exact audit instead, set
`CONTRIBUTION_SCORING_METHOD=exact_metric_drop`.

The same launcher now includes NYUv2 dense-depth vertical slices. A NYUv2
manifest should live at `$DATA_ROOT/nyuv2/val_manifest.jsonl` and include
`image`, `depth`, and `split` fields. Dense profiles export aligned
`targets.npz` automatically:

```bash
export DINO_HF_NAME_OR_PATH=/path/to/facebook/dinov2-base
export LOCAL_FILES_ONLY=1
bash scripts/run_full_profile.sh dino_nyuv2_l11
```

Set paths:

```bash
export DATA_ROOT=/path/to/manifests
export SAE_ROOT=/path/to/sae_checkpoints
export ARTIFACT_ROOT=/path/to/output_artifacts
export PYTHONPATH=$PWD/src
```

Validate the entry gates:

```bash
python -m feature_economy.cli.main check-runtime --profile experiments --require
python -m feature_economy.cli.main check-configs --config-root configs
python -m feature_economy.cli.main plan-runs \
  --config-root configs \
  --output-json "$ARTIFACT_ROOT/run_plan/reproduction_run_plan.json" \
  --output-csv "$ARTIFACT_ROOT/run_plan/reproduction_run_plan.csv"
python -m feature_economy.cli.main check-bundle \
  --run-plan-json "$ARTIFACT_ROOT/run_plan/reproduction_run_plan.json" \
  --artifact-root "$ARTIFACT_ROOT" \
  --output-json "$ARTIFACT_ROOT/run_plan/artifact_bundle_check_initial.json"
python -m feature_economy.cli.main check-models --config-root configs
python -m feature_economy.cli.main check-manifest \
  --manifest "$DATA_ROOT/imagenet/val_manifest.jsonl" \
  --task-type classification \
  --expected-split val
```

Extract or provide native features:

```bash
python -m feature_economy.cli.main extract-features \
  --config-root configs \
  --backend huggingface \
  --manifest "$DATA_ROOT/imagenet/val_manifest.jsonl" \
  --task-type classification \
  --model-id dino_v2_base \
  --expected-split val \
  --batch-size 16 \
  --device cuda \
  --hf-name-or-path "${DINO_HF_NAME_OR_PATH:-facebook/dinov2-base}" \
  --output-dir "$ARTIFACT_ROOT/features/dino_v2_base/imagenet_val_l11"
```

For I-JEPA or another local model, export a TorchScript feature module first;
see `docs/export_torchscript_backbones.md`.

Validate features:

```bash
python -m feature_economy.cli.main validate-arrays \
  --npz "$ARTIFACT_ROOT/features/dino_v2_base/imagenet_val_l11/features.npz" \
  --kind features \
  --task-type classification \
  --manifest "$DATA_ROOT/imagenet/val_manifest.jsonl" \
  --expected-split val
```

Convert an SAE checkpoint and extract SAE codes:

```bash
python -m feature_economy.cli.main convert-sae-checkpoint \
  --input-checkpoint "$SAE_ROOT/dino_l11_topk32_exp4/final_sae.pt" \
  --output-checkpoint "$ARTIFACT_ROOT/checkpoints/dino_l11_topk32_exp4_lightweight.npz"

python -m feature_economy.cli.main extract-sae-codes \
  --config-root configs \
  --backend linear-topk \
  --features-npz "$ARTIFACT_ROOT/features/dino_v2_base/imagenet_val_l11/features.npz" \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --checkpoint "$ARTIFACT_ROOT/checkpoints/dino_l11_topk32_exp4_lightweight.npz" \
  --output-dir "$ARTIFACT_ROOT/codes/dino_l11_topk32_exp4/imagenet_val"
```

Run an SAE-code probe, then the AAA analysis chain:

```bash
python -m feature_economy.cli.main probe-sae \
  --backend linear-probe \
  --codes-npz "$ARTIFACT_ROOT/codes/dino_l11_topk32_exp4/imagenet_val/codes.npz" \
  --manifest "$DATA_ROOT/imagenet/val_manifest.jsonl" \
  --task-type classification \
  --task-id imagenet_1k \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --expected-split val \
  --output-dir "$ARTIFACT_ROOT/probes/sae/dino_l11_topk32_exp4/imagenet_val"
```

For classification and counting, the public linear-probe backend mean-pools
token maps before fitting the readout. The saved probe weights therefore stay
aligned with the final native feature dimension or SAE code dimension, which is
what the downstream ranking and ablation commands expect.

Continue with Availability, Access, ranking-sensitivity, and Allocation:

```bash
python -m feature_economy.cli.main compute-usage \
  --config-root configs \
  --codes-npz "$ARTIFACT_ROOT/codes/dino_l11_topk32_exp4/imagenet_val/codes.npz" \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --dataset-id imagenet_1k \
  --split val \
  --output-dir "$ARTIFACT_ROOT/analysis/availability/dino_l11_topk32_exp4/imagenet_val"

python -m feature_economy.cli.main rank-features \
  --config-root configs \
  --codes-npz "$ARTIFACT_ROOT/codes/dino_l11_topk32_exp4/imagenet_val/codes.npz" \
  --probe-logits-npz "$ARTIFACT_ROOT/probes/sae/dino_l11_topk32_exp4/imagenet_val/probe_logits.npz" \
  --task-id imagenet_1k \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --top-k 100 \
  --output-dir "$ARTIFACT_ROOT/analysis/ranking/dino_l11_topk32_exp4/imagenet_val"

# Optional Module E-style ranking sensitivity control: provide a saved
# contribution-score vector from single-feature validation ablations, then
# rerun the same ranking step with an alternative ranking rule.
python -m feature_economy.cli.main compute-contributions \
  --config-root configs \
  --codes-npz "$ARTIFACT_ROOT/codes/dino_l11_topk32_exp4/imagenet_val/codes.npz" \
  --probe-logits-npz "$ARTIFACT_ROOT/probes/sae/dino_l11_topk32_exp4/imagenet_val/probe_logits.npz" \
  --task-type classification \
  --task-id imagenet_1k \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --output-dir "$ARTIFACT_ROOT/analysis/contribution/dino_l11_topk32_exp4/imagenet_val"

python -m feature_economy.cli.main rank-features \
  --config-root configs \
  --codes-npz "$ARTIFACT_ROOT/codes/dino_l11_topk32_exp4/imagenet_val/codes.npz" \
  --probe-logits-npz "$ARTIFACT_ROOT/probes/sae/dino_l11_topk32_exp4/imagenet_val/probe_logits.npz" \
  --contribution-npz "$ARTIFACT_ROOT/analysis/contribution/dino_l11_topk32_exp4/imagenet_val/contribution_scores.npz" \
  --ranking-method hybrid \
  --hybrid-alpha 0.5 \
  --task-id imagenet_1k \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --top-k 100 \
  --output-dir "$ARTIFACT_ROOT/analysis/ranking_hybrid/dino_l11_topk32_exp4/imagenet_val"

python -m feature_economy.cli.main compute-subset-usage \
  --config-root configs \
  --codes-npz "$ARTIFACT_ROOT/codes/dino_l11_topk32_exp4/imagenet_val/codes.npz" \
  --ranking-json "$ARTIFACT_ROOT/analysis/ranking/dino_l11_topk32_exp4/imagenet_val/task_feature_ranking.json" \
  --top-k 100 \
  --random-seed 0 \
  --output-dir "$ARTIFACT_ROOT/analysis/subset_usage/dino_l11_topk32_exp4/imagenet_val"

python -m feature_economy.cli.main ablate-features \
  --config-root configs \
  --backend linear-probe \
  --codes-npz "$ARTIFACT_ROOT/codes/dino_l11_topk32_exp4/imagenet_val/codes.npz" \
  --probe-logits-npz "$ARTIFACT_ROOT/probes/sae/dino_l11_topk32_exp4/imagenet_val/probe_logits.npz" \
  --ranking-json "$ARTIFACT_ROOT/analysis/ranking/dino_l11_topk32_exp4/imagenet_val/task_feature_ranking.json" \
  --task-type classification \
  --top-k 100 \
  --random-seed 0 \
  --output-dir "$ARTIFACT_ROOT/analysis/ablation/dino_l11_topk32_exp4/imagenet_val"
```

Index artifacts and export tables/figures:

```bash
python -m feature_economy.cli.main index-artifacts \
  --input-dir "$ARTIFACT_ROOT/analysis" \
  --output-json "$ARTIFACT_ROOT/index/analysis/artifact_index.json" \
  --output-csv "$ARTIFACT_ROOT/index/analysis/artifact_index.csv" \
  --require-valid

python -m feature_economy.cli.main make-tables \
  --artifact-index "$ARTIFACT_ROOT/index/analysis/artifact_index.json" \
  --output-dir "$ARTIFACT_ROOT/tables/analysis"

python -m feature_economy.cli.main make-figures \
  --table-dir "$ARTIFACT_ROOT/tables/analysis" \
  --output-dir "$ARTIFACT_ROOT/figures/analysis" \
  --formats png,pdf
```

## 4. Optional: Full Saved-Array Audit Bundle

The first public v1 saved-array audit bundle is distributed as split GitHub
Release assets. It contains archived paper-used features, SAE codes, probe
outputs, and generated tables. It is useful for auditing and debugging, but it
is not required for the default rerun path.

Download every part and checksum file:

```bash
gh release download v1-saved-array-20260513 \
  --repo sellerbubble/dino-ijepa-sae-feature-economy-repro \
  --pattern 'feature_economy_artifacts_v1*'
```

Reassemble and verify:

```bash
cat feature_economy_artifacts_v1.tar.gz.part-* > feature_economy_artifacts_v1.tar.gz
sha256sum -c feature_economy_artifacts_v1.tar.gz.sha256
tar -xzf feature_economy_artifacts_v1.tar.gz
export ARTIFACT_ROOT=$PWD/public_repro_feature_economy_v1_candidate
export PYTHONPATH=$PWD/src
```

Then verify the unpacked bundle and regenerate public tables:

```bash
python -m feature_economy.cli.main check-bundle \
  --run-plan-json "$ARTIFACT_ROOT/run_plan/reproduction_run_plan.json" \
  --artifact-root "$ARTIFACT_ROOT" \
  --output-json "$ARTIFACT_ROOT/run_plan/artifact_bundle_check_local.json" \
  --require-complete

python -m feature_economy.cli.main index-artifacts \
  --input-dir "$ARTIFACT_ROOT" \
  --output-json "$ARTIFACT_ROOT/index/artifact_index_local.json" \
  --output-csv "$ARTIFACT_ROOT/index/artifact_index_local.csv" \
  --require-valid

python -m feature_economy.cli.main make-tables \
  --artifact-index "$ARTIFACT_ROOT/index/artifact_index_local.json" \
  --output-dir "$ARTIFACT_ROOT/tables_local"
```

Expected current audit-bundle status:

- `66/66` run-plan rows complete;
- `190/190` indexed artifacts valid;
- generated tables for probe scores, Availability, Access, and Allocation.

## 5. What This Does Not Claim

This quickstart does not redistribute datasets, model weights, SAE checkpoints,
or private paper-scale training loops. It provides a public path for rerunning
the feature extraction, SAE-code extraction, probe, and AAA analysis chain under
documented configs. See `docs/public_release_audit_20260513.md` for the current
release boundary.
