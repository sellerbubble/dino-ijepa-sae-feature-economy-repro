# Public Experiment Matrix

This document is the compact map of the configured public reproduction matrix.
It is generated conceptually from `feature-economy plan-runs`; update it whenever
config changes alter stage coverage, row counts, or experiment scope.

Current command:

```bash
PYTHONPATH=src python -m feature_economy.cli.main plan-runs \
  --config-root configs \
  --output-json /tmp/feature_economy_run_plan.json \
  --output-csv /tmp/feature_economy_run_plan.csv
```

Current validated run-plan size: **149 rows**.

## Stage Coverage

| Stage | Rows | Role |
|---|---:|---|
| `native_probe` | 8 | Original frozen-backbone probe baseline for both models across four tasks. |
| `feature_extraction` | 17 | Native feature extraction for main tasks and layer sweeps. |
| `sae_code_extraction` | 17 | SAE-code extraction from saved native features. |
| `sae_probe` | 8 | SAE-code task probes used as feature-analysis substrate. |
| `availability` | 11 | Global SAE feature usage, including main and layer-sweep availability. |
| `contribution_scores` | 8 | Validation-contribution scores for ranking controls. |
| `feature_ranking` | 24 | Probe-weight, validation-contribution, and hybrid rankings. |
| `subset_usage` | 24 | Task-selected top-k versus matched random subset usage. |
| `feature_ablation` | 24 | SAE-channel allocation / causal-burden ablation. |
| `native_subspace_ablation` | 8 | Module F native-space bridge for NYUv2 final and second-last layers. |

## Task Coverage

| Task ID | Rows | Notes |
|---|---:|---|
| `imagenet_1k` | 28 | Classification native/SAE probes and ranking-control allocation. |
| `imagenet_1k_val` | 29 | Availability and layer-sweep usage on held-out ImageNet-style evaluation arrays. |
| `nyuv2_depth` | 36 | Depth probes, ranking controls, allocation, and Module F native ablation. |
| `ade20k_segmentation` | 28 | Segmentation probes, ranking controls, and allocation. |
| `clevr_count` | 28 | Image-only counting probes, ranking controls, and allocation. |

## Model And Layer Coverage

| Model | Main SAE Layer | Additional Layer-Sweep SAEs |
|---|---|---|
| `dino_v2_base` | `dino_l11_topk32_exp4` | `dino_l3_topk32_exp4`, `dino_l7_topk32_exp4`, `dino_l9_topk32_exp4`, `dino_l10_topk32_exp4` |
| `ijepa_vit_h14` | `ijepa_l31_topk32_exp4` | `ijepa_l8_topk32_exp4`, `ijepa_l20_topk32_exp4`, `ijepa_l30_topk32_exp4` |

The configured row split is 76 DINO rows and 73 I-JEPA rows. The asymmetry comes
from the current representative layer choices, not from different task coverage.

## Ranking-Control Coverage

For each model/task pair in the main SAE path, the public plan includes:

- `probe_weight`
- `validation_contribution`
- `hybrid`

Each method writes a ranking, subset-usage summary, and feature-ablation summary
under `analysis/ranking_controls/{sae_id}/{task_slug}/{method}/`.

## Advanced Module Coverage

`configs/advanced/module_f_native_ablation_nyuv2.yaml` defines:

| Run ID | Model | SAE | Top-k Values | Random Pool |
|---|---|---|---|---|
| `dino_l11_final` | `dino_v2_base` | `dino_l11_topk32_exp4` | 20, 100 | `exclude_topk` |
| `ijepa_l31_final` | `ijepa_vit_h14` | `ijepa_l31_topk32_exp4` | 20, 100 | `exclude_topk` |
| `dino_l10_second_last` | `dino_v2_base` | `dino_l10_topk32_exp4` | 20, 100 | `exclude_topk` |
| `ijepa_l30_second_last` | `ijepa_vit_h14` | `ijepa_l30_topk32_exp4` | 20, 100 | `exclude_topk` |

The command consumes saved native `features.npz`, a lightweight SAE checkpoint
with `decoder_weight`, hybrid ranking JSON, and SAE probe logits. It performs
projection removal in SAE runtime-normalized coordinates. NYUv2
perturbation-response analysis is intentionally outside the current public
default chain.

## Optional Trial Profiles

`scripts/prepare_real_artifact_slice.py` can generate focused artifact-manifest
workspaces for:

- `imagenet_v1_trial`, `dino_imagenet_v1_trial`, `ijepa_imagenet_v1_trial`
- `nyuv2_v1_trial`, `dino_nyuv2_v1_trial`, `ijepa_nyuv2_v1_trial`
- `ade20k_v1_trial`, `dino_ade20k_v1_trial`, `ijepa_ade20k_v1_trial`
- `clevr_count_v1_trial`, `dino_clevr_count_v1_trial`, `ijepa_clevr_count_v1_trial`
- `layer_sweep_v1_trial`
- `module_f_nyuv2_v1_trial`

These profiles are for staged real-artifact preparation and auditing. They are
not a substitute for the full 149-row paper-style run plan.
