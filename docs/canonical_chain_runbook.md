# Canonical Paper Chain Runbook

This is the public how-to guide for reproducing the main Feature Economy
artifact chain from dataset manifests, model features, SAE codes, and task
targets.

It is intentionally cache-friendly but not cache-required. Full paper-scale
backbone extraction can be expensive and checkpoint-dependent, so the canonical
chain is split into:

1. Validate configs, runtime, manifests, checkpoints, and preprocessing.
2. Extract or provide saved native features.
3. Convert or provide lightweight SAE checkpoints.
4. Extract SAE codes.
5. Run native and SAE-code probes.
6. Compute Availability, Access, and Allocation artifacts.
7. Index artifacts and export paper-facing tables and overview figures.

The release boundary for this lightweight rerun path is documented in
`docs/public_v1_scope.md`. The optional saved-array audit bundle follows the
same artifact contracts but is not required for ordinary reruns.

For a tiny checkpoint-free validation of the same command surface, run:

```bash
cd public_repro
OUTPUT_ROOT=/tmp/feature_economy_public_smoke bash scripts/reproduce_smoke.sh
```

## Inputs

Set these paths for a real run:

```bash
export DATA_ROOT=/path/to/public_or_local_manifests
export SAE_ROOT=/path/to/sae_checkpoints
export ARTIFACT_ROOT=/path/to/feature_economy_artifacts
export PYTHONPATH=$PWD/src
```

Expected manifest files follow the contract in `docs/datasets.md`:

- ImageNet/CLEVR rows: `image`, `split`, `label`.
- NYUv2 rows: `image`, `split`, `depth`.
- ADE20K rows: `image`, `split`, `segmentation`.

Dense linear-probe backends also require explicit saved target arrays:

- NYUv2: `targets.npz` with `targets` shaped like `[N, H, W]`.
- ADE20K: `targets.npz` with `targets` shaped like `[N, H, W]`.

Saved feature/code arrays use the following public keys:

- Native features: `features.npz` with key `features`.
- SAE codes: `codes.npz` with key `codes`.
- Probe outputs: `probe_logits.npz` with `weights` and task-specific targets.

Before passing an externally exported or converted array to the next step, run
`feature-economy validate-arrays`. The full array contract is documented in
`docs/array_contracts.md`.

## Step 1: Validate Environment And Configs

```bash
python -m feature_economy.cli.main check-runtime --profile smoke --require
python -m feature_economy.cli.main check-runtime --profile experiments
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
```

The run plan is a machine-readable checklist of the configured model/task/SAE
matrix and expected artifact directories. It is useful before launching a large
run and after adding a new model, layer, task, or analysis module. The initial
bundle check will usually report missing rows before the run starts; rerun it
with `--require-complete` after all planned artifacts have been generated.

If SAE checkpoint paths should resolve on the current machine:

```bash
python -m feature_economy.cli.main check-models \
  --config-root configs \
  --require-resolved-checkpoints
```

Validate each task manifest before extraction:

```bash
python -m feature_economy.cli.main check-manifest \
  --manifest "$DATA_ROOT/imagenet/val_manifest.jsonl" \
  --task-type classification \
  --expected-split val

python -m feature_economy.cli.main check-manifest \
  --manifest "$DATA_ROOT/nyuv2/val_manifest.jsonl" \
  --task-type dense_depth \
  --expected-split val
```

## Step 2: Inspect Image Preprocessing

Run this once per model/task family before expensive extraction:

```bash
python -m feature_economy.cli.main inspect-images \
  --config-root configs \
  --manifest "$DATA_ROOT/imagenet/val_manifest.jsonl" \
  --task-type classification \
  --model-id dino_v2_base \
  --expected-split val \
  --limit 8 \
  --output-json "$ARTIFACT_ROOT/inspection/dino_imagenet_val.json"
```

The public v0 configs use a shared ImageNet-style resize/crop/normalize policy
for DINO and I-JEPA.

## Step 3: Extract Native Features

DINOv2-B/14 can be extracted through the HuggingFace backend:

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

I-JEPA and other local backbones can enter the same chain through the
TorchScript backend. Export a feature module that accepts normalized `NCHW`
image tensors and returns the desired target-layer features directly. See
`docs/export_torchscript_backbones.md` for the export contract and a toy
end-to-end example:

```bash
python -m feature_economy.cli.main extract-features \
  --config-root configs \
  --backend torchscript \
  --checkpoint "$ARTIFACT_ROOT/checkpoints/ijepa_l31_feature_module.pt" \
  --manifest "$DATA_ROOT/imagenet/val_manifest.jsonl" \
  --task-type classification \
  --model-id ijepa_vit_h14 \
  --expected-split val \
  --batch-size 16 \
  --device cuda \
  --output-dir "$ARTIFACT_ROOT/features/ijepa_vit_h14/imagenet_val_l31"
```

The public repo intentionally does not yet include an official raw I-JEPA
checkpoint loader. The TorchScript bridge keeps the reproduction contract
executable while avoiding private checkpoint-loader assumptions.

## Step 4: Convert SAE Checkpoints And Extract Codes

Convert full SAE checkpoints into the lightweight public format:

```bash
python -m feature_economy.cli.main convert-sae-checkpoint \
  --input-checkpoint "$SAE_ROOT/dino_l11_topk32_exp4/final_sae.pt" \
  --output-checkpoint "$ARTIFACT_ROOT/checkpoints/dino_l11_topk32_exp4_lightweight.npz"
```

Extract SAE codes:

```bash
python -m feature_economy.cli.main extract-sae-codes \
  --config-root configs \
  --backend linear-topk \
  --features-npz "$ARTIFACT_ROOT/features/dino_v2_base/imagenet_val_l11/features.npz" \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --checkpoint "$ARTIFACT_ROOT/checkpoints/dino_l11_topk32_exp4_lightweight.npz" \
  --output-dir "$ARTIFACT_ROOT/codes/dino_l11_topk32_exp4/imagenet_val"
```

Validate saved feature/code arrays before probe training:

```bash
python -m feature_economy.cli.main validate-arrays \
  --npz "$ARTIFACT_ROOT/features/dino_v2_base/imagenet_val_l11/features.npz" \
  --kind features \
  --task-type classification \
  --manifest "$DATA_ROOT/imagenet/val_manifest.jsonl" \
  --expected-split val

python -m feature_economy.cli.main validate-arrays \
  --npz "$ARTIFACT_ROOT/codes/dino_l11_topk32_exp4/imagenet_val/codes.npz" \
  --kind codes \
  --task-type classification \
  --manifest "$DATA_ROOT/imagenet/val_manifest.jsonl" \
  --expected-split val
```

## Step 5: Run Native And SAE-Code Probes

Classification or counting:

```bash
python -m feature_economy.cli.main probe-native \
  --backend linear-probe \
  --features-npz "$ARTIFACT_ROOT/features/dino_v2_base/imagenet_val_l11/features.npz" \
  --manifest "$DATA_ROOT/imagenet/val_manifest.jsonl" \
  --task-type classification \
  --task-id imagenet_1k \
  --model-id dino_v2_base \
  --expected-split val \
  --output-dir "$ARTIFACT_ROOT/probes/native/dino_v2_base/imagenet_val"

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

Depth:

```bash
python -m feature_economy.cli.main probe-sae \
  --backend linear-probe \
  --codes-npz "$ARTIFACT_ROOT/codes/dino_l11_topk32_exp4/nyuv2_val/codes.npz" \
  --manifest "$DATA_ROOT/nyuv2/val_manifest.jsonl" \
  --targets-npz "$DATA_ROOT/nyuv2/val_targets.npz" \
  --task-type dense_depth \
  --task-id nyuv2_depth \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --expected-split val \
  --output-dir "$ARTIFACT_ROOT/probes/sae/dino_l11_topk32_exp4/nyuv2_val"
```

Segmentation:

```bash
python -m feature_economy.cli.main probe-sae \
  --backend linear-probe \
  --codes-npz "$ARTIFACT_ROOT/codes/dino_l11_topk32_exp4/ade20k_val/codes.npz" \
  --manifest "$DATA_ROOT/ade20k/val_manifest.jsonl" \
  --targets-npz "$DATA_ROOT/ade20k/val_targets.npz" \
  --task-type dense_segmentation \
  --task-id ade20k_segmentation \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --expected-split val \
  --num-classes 150 \
  --ignore-index 255 \
  --output-dir "$ARTIFACT_ROOT/probes/sae/dino_l11_topk32_exp4/ade20k_val"
```

## Step 6: Availability

Availability measures which SAE features fire and how unevenly they are used.

```bash
python -m feature_economy.cli.main compute-usage \
  --config-root configs \
  --codes-npz "$ARTIFACT_ROOT/codes/dino_l11_topk32_exp4/imagenet_val/codes.npz" \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --dataset-id imagenet_1k \
  --split val \
  --output-dir "$ARTIFACT_ROOT/analysis/availability/dino_l11_topk32_exp4/imagenet_val"
```

## Step 7: Access

Access starts from the SAE-code probe weights, ranks task features, then compares
task-selected features to a fired-count bucket-matched random control.

```bash
python -m feature_economy.cli.main rank-features \
  --config-root configs \
  --codes-npz "$ARTIFACT_ROOT/codes/dino_l11_topk32_exp4/imagenet_val/codes.npz" \
  --probe-logits-npz "$ARTIFACT_ROOT/probes/sae/dino_l11_topk32_exp4/imagenet_val/probe_logits.npz" \
  --task-id imagenet_1k \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --top-k 100 \
  --output-dir "$ARTIFACT_ROOT/analysis/ranking/dino_l11_topk32_exp4/imagenet_val"
```

For Module E-style ranking sensitivity controls, first compute a validation
contribution score vector by ablating one SAE feature at a time under the saved
linear probe:

```bash
python -m feature_economy.cli.main compute-contributions \
  --config-root configs \
  --codes-npz "$ARTIFACT_ROOT/codes/dino_l11_topk32_exp4/imagenet_val/codes.npz" \
  --probe-logits-npz "$ARTIFACT_ROOT/probes/sae/dino_l11_topk32_exp4/imagenet_val/probe_logits.npz" \
  --task-type classification \
  --task-id imagenet_1k \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --output-dir "$ARTIFACT_ROOT/analysis/contribution/dino_l11_topk32_exp4/imagenet_val"

python -m feature_economy.cli.main validate-arrays \
  --npz "$ARTIFACT_ROOT/analysis/contribution/dino_l11_topk32_exp4/imagenet_val/contribution_scores.npz" \
  --kind contribution_scores

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
```

Then compute the selected-vs-random Access summary from whichever ranking JSON
you want to analyze:

```bash

python -m feature_economy.cli.main compute-subset-usage \
  --config-root configs \
  --codes-npz "$ARTIFACT_ROOT/codes/dino_l11_topk32_exp4/imagenet_val/codes.npz" \
  --ranking-json "$ARTIFACT_ROOT/analysis/ranking/dino_l11_topk32_exp4/imagenet_val/task_feature_ranking.json" \
  --top-k 100 \
  --random-seed 0 \
  --output-dir "$ARTIFACT_ROOT/analysis/subset_usage/dino_l11_topk32_exp4/imagenet_val"
```

## Step 8: Allocation

Allocation zeroes selected SAE feature channels and recomputes the saved linear
probe. The public delta convention is: positive `metric_delta` means performance
got worse after ablation.

```bash
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

Use the same command shape for `count_classification`, `dense_depth`, and
`dense_segmentation`; only `--task-type` changes because probe targets/classes
are already stored in `probe_logits.npz`.

## Step 9: Index Artifacts

Index each artifact family before building tables:

```bash
python -m feature_economy.cli.main index-artifacts \
  --input-dir "$ARTIFACT_ROOT/probes" \
  --output-json "$ARTIFACT_ROOT/index/probes/artifact_index.json" \
  --output-csv "$ARTIFACT_ROOT/index/probes/artifact_index.csv" \
  --require-valid

python -m feature_economy.cli.main index-artifacts \
  --input-dir "$ARTIFACT_ROOT/analysis" \
  --output-json "$ARTIFACT_ROOT/index/analysis/artifact_index.json" \
  --output-csv "$ARTIFACT_ROOT/index/analysis/artifact_index.csv" \
  --require-valid
```

## Step 10: Export Tables

```bash
python -m feature_economy.cli.main make-tables \
  --artifact-index "$ARTIFACT_ROOT/index/probes/artifact_index.json" \
  --output-dir "$ARTIFACT_ROOT/tables/probes"

python -m feature_economy.cli.main make-tables \
  --artifact-index "$ARTIFACT_ROOT/index/analysis/artifact_index.json" \
  --output-dir "$ARTIFACT_ROOT/tables/analysis"
```

Current table outputs:

- `probe_scores.csv`
- `availability_summary.csv`
- `subset_usage_summary.csv`
- `ablation_summary.csv`

## Step 11: Export Overview Figures

Install the optional figure dependency first:

```bash
pip install -e .[figures]
python -m feature_economy.cli.main check-runtime --profile figures --require
```

Then build overview figures from the exported CSV tables:

```bash
python -m feature_economy.cli.main make-figures \
  --table-dir "$ARTIFACT_ROOT/tables/analysis" \
  --output-dir "$ARTIFACT_ROOT/figures/analysis" \
  --formats png,pdf
```

Current figure outputs are lightweight reproducibility figures:

- `probe_scores.{png,pdf}`
- `availability_buckets.{png,pdf}`
- `subset_usage.{png,pdf}`
- `ablation_deltas.{png,pdf}`

## Current Limits

- DINOv2 HuggingFace feature extraction is implemented. I-JEPA/local backbone
  extraction is supported through exported TorchScript feature modules; the repo
  does not yet include an official raw I-JEPA checkpoint loader.
- Linear-probe backends are lightweight public reproductions, not necessarily
  exact paper training loops.
- Dense tasks currently consume explicit `targets.npz`; raw target-file loading
  from NYUv2/ADE20K manifests can be added later.
- Figure builders currently produce simple overview plots from public CSV
  tables, not the fully designed paper figures.
