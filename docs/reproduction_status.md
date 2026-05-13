# Reproduction Status

Date: 2026-05-13

This document separates what is executable now from the remaining gaps toward a
full paper-scale public reproduction.

Public v1 is scoped as a lightweight executable rerun repo with optional
saved-array audit support. See `docs/public_v1_scope.md` for what belongs in v1
and what should remain a future extension. As of 2026-05-13, the public command
surface, smoke tests, tiny complete bundle, and optional saved-array audit
bundle are available.

## Current Executable Smoke Chain

The public repo currently supports a dataset-free and checkpoint-free smoke
chain:

```text
configs
  -> check-configs
  -> smoke-probe
  -> smoke-analysis
  -> index-artifacts
  -> make-tables
```

These commands validate the public interfaces and artifact contracts. They do
not run DINO, I-JEPA, SAE encoders, or downstream probes.

## Optional Full Saved-Array Audit Candidate

The full candidate is an audit release bundle, not the default public
reproduction path. It contains saved features, SAE codes, probe outputs, and
analysis summaries for the configured DINO/I-JEPA public v1 matrix so that
paper-used intermediate arrays can be inspected if needed.

| Item | Status |
| --- | --- |
| Candidate root | `${PRIVATE_ARTIFACT_ROOT}/public_repro_feature_economy_v1_candidate` |
| Run-plan completion | `66/66` rows complete |
| Artifact index | `190/190` valid records |
| Generated CSV tables | probe scores, availability, subset usage, and ablation summaries |
| Metadata sanitizer | PASS, `0` remaining private-path hits |
| Packaged archive | `${PRIVATE_ARTIFACT_ROOT}/public_repro_release_assets/feature_economy_artifacts_v1.tar.gz` |
| Release split parts | 15 parts: `part-00` ... `part-13` at 512 MiB and `part-14` at approximately 361 MiB |
| Reassembly check | PASS, concatenated parts match archive SHA256 |

Default users should instead regenerate these files with the commands in
`docs/release_quickstart.md` and compare against `docs/expected_results.md`.

## Commands

| Command | Status | What it verifies | Scientific result? |
| --- | --- | --- | --- |
| `feature-economy check-configs` | executable | Model, SAE, task, and experiment config shape. | No |
| `feature-economy plan-runs` | executable | Expands configured model/task/SAE experiments into a JSON/CSV run matrix. | No |
| `feature-economy check-bundle` | executable | Checks a planned artifact root for required files per run-plan row. | No |
| `feature-economy check-runtime` | executable | Smoke or experiment dependency availability, with optional JSON report. | No |
| `feature-economy check-models` | executable | Model/SAE registry links and checkpoint placeholder resolution. | No |
| `feature-economy check-manifest` | executable | JSONL manifest fields and split assumptions. | No |
| `feature-economy validate-arrays` | executable | Public `.npz` contracts for features, SAE codes, probe logits, and dense targets. | No |
| `feature-economy inspect-images` | executable | Manifest path resolution and shared resize/crop/normalize preprocessing. | No |
| `feature-economy smoke-probe` | executable | Config-driven native/SAE probe summary contract. | No |
| `feature-economy smoke-analysis` | executable | Availability, ranking, access, and allocation artifact contracts. | No |
| `feature-economy compute-usage` | smoke and codes backend executable | Availability artifact command surface and real fired-count stats from `codes.npz`. | Yes when fed real SAE codes. |
| `feature-economy rank-features` | smoke and linear-probe backend executable | Task ranking artifact from smoke data or saved codes plus probe weights, validation contribution scores, or hybrid ranking. | Yes for probe-weight or contribution-backed ranking from real SAE-code probes. |
| `feature-economy compute-contributions` | linear-probe backend executable | Per-feature validation contribution scores from single-channel SAE ablations of a saved linear probe. | Yes when fed real SAE codes and probe outputs. |
| `feature-economy compute-subset-usage` | codes and ranking backend executable | Top-k task-selected subset usage and fired-count bucket-matched random control. | Yes for fired-count usage from real SAE codes and rankings. |
| `feature-economy ablate-features` | smoke and linear-probe backend executable | SAE feature zero-ablation with matched random control for saved linear probes. | Yes for lightweight classification/counting/depth/segmentation SAE-code probes. |
| `feature-economy probe-native` | fixture and linear-probe backend executable | Native probe output contract from fixture predictions or saved features plus labels/targets. | Yes for lightweight classification/counting/depth/segmentation probes. |
| `feature-economy probe-sae` | fixture and linear-probe backend executable | SAE-code probe summary from fixture predictions or saved codes plus labels/targets. | Yes for lightweight classification/counting/depth/segmentation probes. |
| `feature-economy extract-features` | fixture, HuggingFace, and TorchScript backends executable | Feature extraction artifact contract, `.npz` layout, DINO-style HF hidden-state extraction, and exported local/I-JEPA feature modules. | Yes only when run with real checkpoint/data or a TorchScript feature module. |
| `feature-economy extract-sae-codes` | fixture and linear-TopK backend executable | SAE-code artifact contract, `.npz` layout, and lightweight SAE inference. | Yes only when run with real features/checkpoint. |
| `feature-economy convert-sae-checkpoint` | executable | Standard SAE `W_enc`/`b_enc`/`b_dec` conversion to public lightweight `.npz`. | No, conversion utility only. |
| `feature-economy index-artifacts` | executable | Artifact validation and provenance registry. | No |
| `feature-economy make-tables` | executable | Probe, availability, subset/access, and ablation CSV tables. | No for smoke artifacts; yes only when fed real artifacts. |
| `feature-economy make-figures` | executable with `figures` extra | Overview PNG/PDF figures from public CSV tables. | No for smoke tables; yes only when fed real tables. |

## Paper Chain Coverage

| Paper chain step | Current public status | Needed for real reproduction |
| --- | --- | --- |
| Native backbone probes | Fixture evaluation and lightweight linear probes implemented for classification/counting/depth/segmentation saved arrays. | Add paper-scale torch training settings if closed-form ridge is insufficient. |
| SAE-code probes | Fixture evaluation and lightweight linear probes implemented for classification/counting/depth/segmentation saved arrays. | Add paper-scale torch training settings if closed-form ridge is insufficient. |
| Availability | Fired-count availability from saved `codes.npz` implemented. | Add usage-rate mode for dense spatial analyses if needed. |
| Access | Probe-weight, validation-contribution, and hybrid ranking plus fired-count bucket-matched random subset usage from saved SAE codes implemented. | Add dense usage-rate mode if needed for per-pixel task-specific streams. |
| Allocation | Linear-probe SAE feature zero-ablation with matched random control implemented for classification/counting/depth/segmentation saved arrays. | Add paper-scale task-specific settings. |
| Paper tables | Probe, availability, access/subset, and allocation CSV tables from indexed artifacts. | Add final paper formatting once real artifacts exist. |
| Paper figures | Lightweight overview figures from public CSV tables implemented. | Add final designed paper figure scripts if exact manuscript visuals are required. |
| Task metrics | Implemented and unit-tested for classification, depth, segmentation, and counting-style accuracy. | Add paper-specific metric variants only if needed. |
| Dataset manifest adapter | Metadata records, manifest-relative path resolution, image inspection, and explicit dense `targets.npz` probe input implemented. | Add optional target-file loaders if public datasets should be read directly from raw masks/maps. |
| Transform policy | Config-backed shared ImageNet resize/crop/normalize implemented with PIL/NumPy. | Reuse the same policy in torch-backed real runners. |
| Fixture native evaluation | Implemented for manifests with precomputed predictions. | Keep for smoke/CI; use linear probe backend for saved feature arrays. |
| Native linear probe | Closed-form ridge probe over saved `features.npz` implemented for classification/counting/depth/segmentation. | Add torch training loop if paper-scale ridge is not representative enough. |
| SAE linear probe | Closed-form ridge probe over saved `codes.npz` implemented for classification/counting/depth/segmentation. | Add torch training loop if paper-scale ridge is not representative enough. |
| Feature extraction | Fixture extraction, HuggingFace DINO-style extraction, and TorchScript local/I-JEPA feature-module extraction implemented with the same `.npz`/summary contract. | Add an official raw I-JEPA checkpoint loader only if exact raw-checkpoint reproduction becomes required. |
| SAE-code extraction | Fixture conversion, lightweight linear-TopK SAE encoding, and standard SAE checkpoint converter implemented. | Add dedicated gated SAE backend if needed. |

## Current Validation Commands

Run from `public_repro/`:

```bash
bash scripts/reproduce_smoke.sh
```

The script writes outputs under `/tmp/feature_economy_public_smoke` by default.
Set `OUTPUT_ROOT=/path/to/output` to choose another location.

To build one complete tiny vertical artifact bundle:

```bash
OUTPUT_ROOT=/tmp/feature_economy_tiny_bundle bash scripts/build_tiny_artifact_bundle.sh
```

This should end with `check-bundle --require-complete` passing for the tiny
DINO/ImageNet SAE slice.

For real lightweight reproduction, follow `docs/canonical_chain_runbook.md`.

Individual commands:

```bash
PYTHONPATH=src python -m unittest discover -s tests

PYTHONPATH=src python -m feature_economy.cli.main check-runtime \
  --profile smoke \
  --require \
  --json-output /tmp/feature_economy_runtime_smoke.json

PYTHONPATH=src python -m feature_economy.cli.main check-runtime \
  --profile experiments

PYTHONPATH=src python -m feature_economy.cli.main check-configs \
  --config-root configs

PYTHONPATH=src python -m feature_economy.cli.main check-models \
  --config-root configs

PYTHONPATH=src python -m feature_economy.cli.main check-manifest \
  --manifest tests/fixtures/tiny_nyuv2_manifest.jsonl \
  --task-type dense_depth \
  --expected-split val

PYTHONPATH=src python -m feature_economy.cli.main inspect-images \
  --config-root configs \
  --manifest /path/to/real_manifest.jsonl \
  --task-type classification \
  --model-id dino_v2_base \
  --expected-split val \
  --limit 8 \
  --output-json /tmp/feature_economy_image_inspection.json

PYTHONPATH=src python -m feature_economy.cli.main extract-features \
  --config-root configs \
  --backend fixture \
  --fixture-manifest tests/fixtures/tiny_imagenet_eval_manifest.jsonl \
  --task-type classification \
  --model-id dino_v2_base \
  --expected-split val \
  --output-dir /tmp/feature_economy_features_fixture

PYTHONPATH=src python -m feature_economy.cli.main extract-features \
  --config-root configs \
  --backend huggingface \
  --manifest /path/to/real_manifest.jsonl \
  --task-type classification \
  --model-id dino_v2_base \
  --expected-split val \
  --batch-size 16 \
  --device cuda \
  --output-dir /tmp/feature_economy_dino_features

PYTHONPATH=src python -m feature_economy.cli.main extract-features \
  --config-root configs \
  --backend torchscript \
  --checkpoint /path/to/ijepa_l31_feature_module.pt \
  --manifest /path/to/real_manifest.jsonl \
  --task-type classification \
  --model-id ijepa_vit_h14 \
  --expected-split val \
  --batch-size 16 \
  --device cuda \
  --output-dir /tmp/feature_economy_ijepa_features

PYTHONPATH=src python -m feature_economy.cli.main extract-sae-codes \
  --config-root configs \
  --backend fixture \
  --features-npz /tmp/feature_economy_features_fixture/features.npz \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --output-dir /tmp/feature_economy_codes_fixture

PYTHONPATH=src python -m feature_economy.cli.main extract-sae-codes \
  --config-root configs \
  --backend linear-topk \
  --features-npz /tmp/feature_economy_dino_features/features.npz \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --checkpoint /path/to/lightweight_sae.npz \
  --output-dir /tmp/feature_economy_dino_codes

PYTHONPATH=src python -m feature_economy.cli.main convert-sae-checkpoint \
  --input-checkpoint /path/to/full_sae.pt \
  --output-checkpoint /tmp/lightweight_sae.npz

PYTHONPATH=src python -m feature_economy.cli.main probe-native \
  --backend fixture \
  --fixture-manifest tests/fixtures/tiny_imagenet_eval_manifest.jsonl \
  --task-type classification \
  --task-id imagenet_1k \
  --model-id dino_v2_base \
  --expected-split val \
  --output-dir /tmp/feature_economy_probe_native_fixture

PYTHONPATH=src python -m feature_economy.cli.main probe-native \
  --backend linear-probe \
  --features-npz /tmp/feature_economy_dino_features/features.npz \
  --manifest /path/to/real_manifest.jsonl \
  --task-type classification \
  --task-id imagenet_1k \
  --model-id dino_v2_base \
  --expected-split val \
  --output-dir /tmp/feature_economy_dino_native_probe

PYTHONPATH=src python -m feature_economy.cli.main probe-sae \
  --backend fixture \
  --fixture-manifest tests/fixtures/tiny_clevr_count_eval_manifest.jsonl \
  --task-type count_classification \
  --task-id clevr_count \
  --model-id ijepa_vit_h14 \
  --sae-id ijepa_l31_topk32_exp4 \
  --expected-split val \
  --output-dir /tmp/feature_economy_probe_sae_fixture

PYTHONPATH=src python -m feature_economy.cli.main probe-sae \
  --backend linear-probe \
  --codes-npz /tmp/feature_economy_dino_codes/codes.npz \
  --manifest /path/to/real_manifest.jsonl \
  --task-type classification \
  --task-id imagenet_1k \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --expected-split val \
  --output-dir /tmp/feature_economy_dino_sae_probe

PYTHONPATH=src python -m feature_economy.cli.main smoke-probe \
  --config-root configs \
  --experiment-id paper_v0_native \
  --output-dir /tmp/feature_economy_smoke_probe

PYTHONPATH=src python -m feature_economy.cli.main smoke-analysis \
  --config-root configs \
  --output-dir /tmp/feature_economy_smoke_analysis

PYTHONPATH=src python -m feature_economy.cli.main compute-usage \
  --config-root configs \
  --output-dir /tmp/feature_economy_usage_smoke \
  --smoke

PYTHONPATH=src python -m feature_economy.cli.main compute-usage \
  --config-root configs \
  --codes-npz /tmp/feature_economy_dino_codes/codes.npz \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --dataset-id imagenet_1k \
  --split val \
  --output-dir /tmp/feature_economy_dino_availability

PYTHONPATH=src python -m feature_economy.cli.main rank-features \
  --config-root configs \
  --output-dir /tmp/feature_economy_rank_smoke \
  --smoke

PYTHONPATH=src python -m feature_economy.cli.main rank-features \
  --config-root configs \
  --codes-npz /tmp/feature_economy_dino_codes/codes.npz \
  --probe-logits-npz /tmp/feature_economy_dino_sae_probe/probe_logits.npz \
  --task-id imagenet_1k \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --output-dir /tmp/feature_economy_dino_ranking

PYTHONPATH=src python -m feature_economy.cli.main compute-subset-usage \
  --config-root configs \
  --codes-npz /tmp/feature_economy_dino_codes/codes.npz \
  --ranking-json /tmp/feature_economy_dino_ranking/task_feature_ranking.json \
  --top-k 100 \
  --random-seed 0 \
  --output-dir /tmp/feature_economy_dino_subset_usage

PYTHONPATH=src python -m feature_economy.cli.main ablate-features \
  --config-root configs \
  --backend linear-probe \
  --codes-npz /tmp/feature_economy_dino_codes/codes.npz \
  --probe-logits-npz /tmp/feature_economy_dino_sae_probe/probe_logits.npz \
  --ranking-json /tmp/feature_economy_dino_ranking/task_feature_ranking.json \
  --task-type classification \
  --top-k 100 \
  --random-seed 0 \
  --output-dir /tmp/feature_economy_dino_ablation

PYTHONPATH=src python -m feature_economy.cli.main ablate-features \
  --config-root configs \
  --output-dir /tmp/feature_economy_ablation_smoke \
  --smoke

PYTHONPATH=src python -m feature_economy.cli.main index-artifacts \
  --input-dir /tmp/feature_economy_smoke_probe \
  --output-json /tmp/feature_economy_probe_index/artifact_index.json \
  --output-csv /tmp/feature_economy_probe_index/artifact_index.csv \
  --require-valid

PYTHONPATH=src python -m feature_economy.cli.main make-tables \
  --artifact-index /tmp/feature_economy_probe_index/artifact_index.json \
  --output-dir /tmp/feature_economy_tables

PYTHONPATH=src python -m feature_economy.cli.main index-artifacts \
  --input-dir /tmp/feature_economy_smoke_analysis \
  --output-json /tmp/feature_economy_analysis_index/artifact_index.json \
  --output-csv /tmp/feature_economy_analysis_index/artifact_index.csv \
  --require-valid
```

`make-tables` writes every supported table type from an artifact index. Indexes
that contain only probe summaries produce a populated `probe_scores.csv`; indexes
that contain analysis artifacts produce populated availability, subset/access,
and ablation tables.

## Next Implementation Target

The next real migration should be a single vertical slice, not all tasks at
once. Recommended target:

```text
manifest fixture -> minimal dataset adapter -> native probe runner interface
```

The first real runner should still support a tiny fixture mode before running
full ImageNet/NYUv2/ADE20K/CLEVR data. This keeps the public repo testable on a
normal laptop and prevents private cluster assumptions from leaking into the
public API.

The metric functions are now available in `feature_economy.probes.metrics`.
They should be reused by future native and SAE-code runners rather than
reimplemented task by task.

`feature-economy probe-native --fixture-manifest ...` is the first public
runner-shaped path that computes metrics from manifest records and writes a
valid `native_probe_summary.json`. Tiny fixtures cover classification,
counting, depth, and segmentation. It is still a fixture evaluator, not a
backbone runner.

`feature-economy probe-sae --fixture-manifest ...` provides the matching
SAE-code probe output contract and writes `sae_probe_summary.json`. It is still
a fixture evaluator, not an SAE-code runner.

Fixture manifests:

- `tests/fixtures/tiny_imagenet_eval_manifest.jsonl`
- `tests/fixtures/tiny_clevr_count_eval_manifest.jsonl`
- `tests/fixtures/tiny_nyuv2_eval_manifest.jsonl`
- `tests/fixtures/tiny_ade20k_eval_manifest.jsonl`
