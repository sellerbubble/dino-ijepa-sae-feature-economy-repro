# DINO/I-JEPA SAE Feature Economy Reproduction

This directory is the curated public reproduction slice for the DINO/I-JEPA SAE
Feature Economy project.

Status: active public reproduction slice. It includes configs, artifact schemas,
tiny validation tests, fixture runners, full paper-style feature/code extraction
interfaces, probe and AAA analysis commands, artifact indexing, and table
export. Optional overview figure export is available with the `figures` extra.
It does not redistribute datasets, model checkpoints, SAE checkpoints, or final
designed paper figures.

## Goal

The public repo should let external readers rerun the main paper chain from
their own datasets, checkpoints, and SAE checkpoints:

```text
dataset/config -> feature extraction -> SAE codes -> native/SAE probes
  -> usage/ranking -> ablation -> paper-style table/figure
```

It should also make future extensions routine: new models, layers, SAEs, tasks,
and analysis modules should be added through configs and thin adapters rather
than by copying experiment scripts.

## What Is Included

- Config examples for DINOv2, I-JEPA, SAE checkpoints, tasks, and experiments.
- Artifact schema helpers for run manifests, probe summaries, feature rankings,
  and ablation summaries.
- Fixture, HuggingFace, TorchScript, and saved-array command backends for the
  paper evidence chain.
- A run-plan exporter that expands model/task/SAE configs into a JSON/CSV
  reproduction matrix.
- Documentation for datasets, checkpoints, artifact schemas, extension workflow,
  paper claim mapping, and the canonical paper chain.
- A TorchScript feature-module bridge for exported I-JEPA/local backbones.
- Tiny tests and smoke scripts that validate the command surface without model
  weights or datasets.
- A GitHub Actions CI template for the standalone public repo export.
- A standalone export script that copies this slice into a clean publishable
  repository tree and runs the public release hygiene checker.
- A release-asset packaging script for optional audit bundles. This is useful
  for archiving paper-used intermediate arrays, but it is not the default
  external reproduction path.

## What Is Not Included Yet

- Dataset downloads or redistributed raw data.
- Model or SAE checkpoints.
- Large paper-scale feature arrays, SAE-code arrays, or probe-logit caches by
  default.
- Raw official I-JEPA checkpoint loading. Exported I-JEPA feature modules can
  use the TorchScript backend.
- Private cluster-specific launchers and local absolute paths.
- Final designed paper figures.
- Private cluster launchers or local artifact paths.

Use `docs/reproduction_status.md` for current coverage and known gaps.
Use `docs/public_v1_scope.md` for the release boundary: public v1 is a full
paper-style rerun repo with smoke tests for code health and optional saved-array
audit support, not a private-cluster training-loop dump.

## Main Reproduction Path

Start with `docs/full_reproduction.md` for the default full paper-style rerun
path from prepared datasets, model checkpoints, SAE checkpoints, and GPU
resources. Use `docs/release_quickstart.md` for the short command path and smoke
checks. The expected outcome is approximate scientific agreement with the paper,
not bit-identical reproduction of private intermediate arrays.

For expected metric ranges and qualitative acceptance checks, see
`docs/expected_results.md`.

An optional full saved-array audit bundle is published as split GitHub Release
assets. It contains paper-used intermediate arrays and tables for traceability,
but ordinary users do not need it to rerun the experiments.

For the complete step-by-step paper-chain template, see
`docs/canonical_chain_runbook.md`.

For remote-server safety rules and non-destructive sync/output conventions, see
`docs/remote_safety.md`.

For the expected layout of public saved-array release assets, see
`docs/release_artifact_bundle_layout.md`.

For the first real saved-array release checklist, see
`docs/first_real_saved_array_bundle_plan.md`.

For local/I-JEPA feature-module export, see
`docs/export_torchscript_backbones.md`.

## Development Check

From this directory:

```bash
bash scripts/reproduce_smoke.sh
bash scripts/build_tiny_artifact_bundle.sh
python scripts/check_public_release.py --root .
bash scripts/export_public_repo.sh --output /tmp/feature_economy_repro_public_export --force
python scripts/create_artifact_export_manifest.py \
  --run-plan-json /tmp/feature_economy_release_plan/reproduction_run_plan.json \
  --output-csv /tmp/feature_economy_release_plan/artifact_export_manifest_template.csv
python scripts/filter_reproduction_run_plan.py \
  --input-json /tmp/feature_economy_release_plan/reproduction_run_plan.json \
  --output-json /tmp/feature_economy_release_plan/dino_imagenet_subset_run_plan.json \
  --model-id dino_v2_base \
  --task-id imagenet_1k \
  --task-id imagenet_1k_val \
  --sae-id "" \
  --sae-id dino_l11_topk32_exp4
python scripts/audit_artifact_export_manifest.py \
  --manifest-csv /tmp/feature_economy_release_plan/artifact_export_manifest_template.csv \
  --output-json /tmp/feature_economy_release_plan/artifact_export_manifest_audit.json
python scripts/copy_artifact_export_manifest.py \
  --manifest-csv /tmp/feature_economy_release_plan/artifact_export_manifest_template.csv \
  --artifact-root /tmp/feature_economy_artifacts_v1 \
  --dry-run
python scripts/create_artifact_bundle_readme.py \
  --run-plan-json /tmp/feature_economy_release_plan/reproduction_run_plan.json \
  --output-readme /tmp/feature_economy_artifacts_v1/README.md \
  --bundle-name feature_economy_artifacts_v1
bash scripts/stage_artifact_bundle_from_manifest.sh \
  --manifest-csv /tmp/feature_economy_release_plan/artifact_export_manifest_template.csv \
  --run-plan-json /tmp/feature_economy_release_plan/reproduction_run_plan.json \
  --artifact-root /tmp/feature_economy_artifacts_v1 \
  --release-output-dir /tmp/feature_economy_release_assets \
  --bundle-name feature_economy_artifacts_v1 \
  --dry-run
bash scripts/package_artifact_bundle.sh \
  --artifact-root /tmp/feature_economy_tiny_bundle \
  --run-plan-json /tmp/feature_economy_tiny_bundle/run_plan/tiny_dino_imagenet_run_plan.json \
  --output-dir /tmp/feature_economy_release_assets \
  --bundle-name feature_economy_tiny_bundle \
  --force
```

`reproduce_smoke.sh` checks the full public command surface. The tiny artifact
bundle script builds one complete DINO/ImageNet SAE slice and verifies it with
`check-bundle --require-complete`. The release checker verifies required files,
script permissions, Markdown fences, README links, and private-path hygiene.
The export script creates the clean standalone tree that can be initialized as
the public GitHub repository. The packaging script is the final local gate for
saved-array release assets before uploading them to a GitHub release or another
artifact host.

Or run individual checks:

```bash
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m feature_economy.cli.main check-runtime --profile smoke --require
PYTHONPATH=src python -m feature_economy.cli.main check-runtime --profile experiments
PYTHONPATH=src python -m feature_economy.cli.main check-runtime --profile figures
PYTHONPATH=src python -m feature_economy.cli.main check-configs --config-root configs
PYTHONPATH=src python -m feature_economy.cli.main plan-runs \
  --config-root configs \
  --output-json /tmp/feature_economy_run_plan.json \
  --output-csv /tmp/feature_economy_run_plan.csv
PYTHONPATH=src python -m feature_economy.cli.main check-bundle \
  --run-plan-json /tmp/feature_economy_run_plan.json \
  --artifact-root /tmp/feature_economy_artifacts \
  --output-json /tmp/feature_economy_bundle_check.json
PYTHONPATH=src python -m feature_economy.cli.main check-models --config-root configs
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
PYTHONPATH=src python -m feature_economy.cli.main eval-native-fixture \
  --manifest tests/fixtures/tiny_clevr_count_eval_manifest.jsonl \
  --task-type count_classification \
  --task-id clevr_count \
  --model-id dino_v2_base \
  --expected-split val \
  --output-dir /tmp/feature_economy_fixture_native
PYTHONPATH=src python -m feature_economy.cli.main eval-native-fixture \
  --manifest tests/fixtures/tiny_nyuv2_eval_manifest.jsonl \
  --task-type dense_depth \
  --task-id nyuv2_depth \
  --model-id dino_v2_base \
  --expected-split val \
  --output-dir /tmp/feature_economy_fixture_depth
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
PYTHONPATH=src python -m feature_economy.cli.main probe-native \
  --backend linear-probe \
  --features-npz /tmp/feature_economy_dino_depth_features/features.npz \
  --manifest /path/to/nyuv2_val_manifest.jsonl \
  --targets-npz /path/to/nyuv2_val_targets.npz \
  --task-type dense_depth \
  --task-id nyuv2_depth \
  --model-id dino_v2_base \
  --expected-split val \
  --output-dir /tmp/feature_economy_dino_depth_probe
PYTHONPATH=src python -m feature_economy.cli.main probe-sae \
  --backend linear-probe \
  --codes-npz /tmp/feature_economy_dino_ade_codes/codes.npz \
  --manifest /path/to/ade20k_val_manifest.jsonl \
  --targets-npz /path/to/ade20k_val_targets.npz \
  --task-type dense_segmentation \
  --task-id ade20k_segmentation \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --expected-split val \
  --num-classes 150 \
  --ignore-index 255 \
  --output-dir /tmp/feature_economy_dino_ade_probe
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
PYTHONPATH=src python -m feature_economy.cli.main validate-arrays \
  --npz /tmp/feature_economy_dino_codes/codes.npz \
  --kind codes \
  --task-type classification \
  --manifest /path/to/real_manifest.jsonl \
  --expected-split val
PYTHONPATH=src python -m feature_economy.cli.main convert-sae-checkpoint \
  --input-checkpoint /path/to/full_sae.pt \
  --output-checkpoint /tmp/lightweight_sae.npz
PYTHONPATH=src python -m feature_economy.cli.main smoke-probe \
  --config-root configs \
  --experiment-id paper_v0_native \
  --output-dir /tmp/feature_economy_smoke
PYTHONPATH=src python -m feature_economy.cli.main smoke-analysis \
  --config-root configs \
  --output-dir /tmp/feature_economy_analysis_smoke
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
  --backend linear-probe \
  --codes-npz /tmp/feature_economy_dino_ade_codes/codes.npz \
  --probe-logits-npz /tmp/feature_economy_dino_ade_probe/probe_logits.npz \
  --ranking-json /tmp/feature_economy_dino_ade_ranking/task_feature_ranking.json \
  --task-type dense_segmentation \
  --top-k 100 \
  --random-seed 0 \
  --output-dir /tmp/feature_economy_dino_ade_ablation
PYTHONPATH=src python -m feature_economy.cli.main ablate-features \
  --config-root configs \
  --output-dir /tmp/feature_economy_ablation_smoke \
  --smoke
PYTHONPATH=src python -m feature_economy.cli.main index-artifacts \
  --input-dir /tmp/feature_economy_smoke \
  --output-json /tmp/feature_economy_index/artifact_index.json \
  --output-csv /tmp/feature_economy_index/artifact_index.csv
PYTHONPATH=src python -m feature_economy.cli.main make-tables \
  --artifact-index /tmp/feature_economy_index/artifact_index.json \
  --output-dir /tmp/feature_economy_tables
PYTHONPATH=src python -m feature_economy.cli.main make-figures \
  --table-dir /tmp/feature_economy_tables \
  --output-dir /tmp/feature_economy_figures \
  --formats png,pdf
```

This checks the smoke schema, fixture runners, saved-array mini paths, artifact
indexing, and table export. It does not run paper-scale experiments.

For the current smoke coverage and remaining reproduction gaps, see
`docs/reproduction_status.md`. For staged implementation goals, see
`docs/public_repro_roadmap.md`.
