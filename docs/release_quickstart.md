# Release Quickstart

This is the shortest path for checking that the public reproduction repo works
and for running the real saved-array evidence chain. For the complete command
reference, see `docs/canonical_chain_runbook.md`.

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
example of the public v1 artifact-first contract.

## 3. Real Saved-Array Vertical Slice

The public alpha is artifact-first. To reproduce the scientific chain, provide
real manifests, native feature arrays, SAE checkpoints, and dense targets.

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

## 4. What This Does Not Claim

This quickstart does not reproduce private paper-scale training loops from raw
datasets. It reproduces the public saved-array/lightweight analysis chain. See
`docs/public_release_audit_20260513.md` for the current release boundary.
