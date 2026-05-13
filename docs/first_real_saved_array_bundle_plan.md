# First Real Saved-Array Bundle Plan

Date: 2026-05-13

This document defines the first non-fixture artifact bundle that should be
prepared for the public DINO/I-JEPA SAE Feature Economy reproduction repo.

The purpose is not to publish raw datasets, model weights, or private cluster
logs. The purpose is to publish a portable saved-array bundle that lets an
external reader rerun the paper's main public analysis chain:

```text
native probe / SAE-code probe
  -> Availability
  -> Access
  -> Allocation
  -> artifact index
  -> CSV tables and overview figures
```

## Release Scope

The first real bundle should cover the public v1 matrix:

| Axis | Required values |
| --- | --- |
| Models | `dino_v2_base`, `ijepa_vit_h14` |
| SAEs | `dino_l11_topk32_exp4`, `ijepa_l31_topk32_exp4` |
| Tasks | `imagenet_1k`, `nyuv2_depth`, `ade20k_segmentation`, `clevr_count` |
| Availability split | `imagenet_1k_val` |
| Analysis axes | Availability, Access, Allocation |
| Release form | Saved arrays plus JSON/CSV summaries, not raw images or checkpoints |

Generate the exact run matrix from the public configs:

```bash
PYTHONPATH=src python -m feature_economy.cli.main plan-runs \
  --config-root configs \
  --output-json /tmp/feature_economy_release_plan/reproduction_run_plan.json \
  --output-csv /tmp/feature_economy_release_plan/reproduction_run_plan.csv
```

The current public v1 configs expand to `66` planned rows:

| Stage | Expected rows |
| --- | ---: |
| `availability` | 2 |
| `feature_extraction` | 8 |
| `sae_code_extraction` | 8 |
| `native_probe` | 8 |
| `sae_probe` | 8 |
| `feature_ranking` | 8 |
| `contribution_scores` | 8 |
| `subset_usage` | 8 |
| `feature_ablation` | 8 |

The generated `reproduction_run_plan.json` is the source of truth. If configs
change, regenerate the plan rather than editing row counts by hand.

For a partial release or first vertical-slice trial, filter the run plan instead
of hand-editing JSON:

```bash
python scripts/filter_reproduction_run_plan.py \
  --input-json /tmp/feature_economy_release_plan/reproduction_run_plan.json \
  --output-json /tmp/feature_economy_release_plan/dino_imagenet_subset_run_plan.json \
  --output-csv /tmp/feature_economy_release_plan/dino_imagenet_subset_run_plan.csv \
  --model-id dino_v2_base \
  --task-id imagenet_1k \
  --task-id imagenet_1k_val \
  --sae-id "" \
  --sae-id dino_l11_topk32_exp4
```

The empty `--sae-id ""` keeps native/feature-extraction rows whose public
`sae_id` field is intentionally blank.

For a first trial or task-level staging pass, maintainers can generate the full
plan, filtered slice plan, and pre-statused export manifest in one command:

```bash
python scripts/prepare_real_artifact_slice.py \
  --config-root configs \
  --profile dino_imagenet_v1_trial \
  --output-dir /tmp/feature_economy_release_plan/dino_imagenet_trial
```

Supported staging profiles are:

| Profile | Rows | Purpose |
| --- | ---: | --- |
| `dino_imagenet_v1_trial` | 9 | Single-model DINO/ImageNet vertical slice. |
| `ijepa_imagenet_v1_trial` | 9 | Single-model I-JEPA/ImageNet vertical slice. |
| `imagenet_v1_trial` | 18 | Combined DINO/I-JEPA ImageNet slice. |
| `nyuv2_v1_trial` | 16 | Combined DINO/I-JEPA NYUv2 depth slice. |
| `dino_nyuv2_v1_trial` | 8 | Single-model DINO/NYUv2 dense-depth slice. |
| `ijepa_nyuv2_v1_trial` | 8 | Single-model I-JEPA/NYUv2 dense-depth slice. |
| `ade20k_v1_trial` | 16 | Combined DINO/I-JEPA ADE20K segmentation slice. |
| `dino_ade20k_v1_trial` | 8 | Single-model DINO/ADE20K dense-segmentation slice. |
| `ijepa_ade20k_v1_trial` | 8 | Single-model I-JEPA/ADE20K dense-segmentation slice. |
| `clevr_count_v1_trial` | 16 | Combined DINO/I-JEPA CLEVR/Count slice. |
| `dino_clevr_count_v1_trial` | 8 | Single-model DINO/CLEVR/Count slice. |
| `ijepa_clevr_count_v1_trial` | 8 | Single-model I-JEPA/CLEVR/Count slice. |
| `full_v1_template` | 66 | Full public v1 saved-array release template. |

The generated manifest is still a handoff sheet, not proof that the artifact
slice is ready. Fill `source_artifact_dir` only after inspecting private or
remote artifacts, then run the audit/copy/staging commands below.

After generating the run plan, create the source-to-public export manifest:

```bash
python scripts/create_artifact_export_manifest.py \
  --run-plan-json /tmp/feature_economy_release_plan/reproduction_run_plan.json \
  --output-csv /tmp/feature_economy_release_plan/artifact_export_manifest_template.csv \
  --output-json /tmp/feature_economy_release_plan/artifact_export_manifest_template.json
```

The export manifest is the handoff sheet for private artifact collection. Fill
`source_artifact_dir`, `source_artifact_note`, `conversion_needed`, and `notes`
before copying anything into the public bundle. Keep `public_artifact_dir` and
`required_files` unchanged unless the public run plan is regenerated.

Before copying files, audit the filled manifest:

```bash
python scripts/audit_artifact_export_manifest.py \
  --manifest-csv /tmp/feature_economy_release_plan/artifact_export_manifest_template.csv \
  --artifact-root "$ARTIFACT_ROOT" \
  --output-json /tmp/feature_economy_release_plan/artifact_export_manifest_audit.json
```

Use `--require-ready` only after all non-omitted rows are expected to be ready.
The audit is intentionally read-only; it checks source directories and required
files without modifying either the private workbench or the public bundle.

After the audit passes for the rows you intend to export, copy ready rows into
the public artifact root:

```bash
python scripts/copy_artifact_export_manifest.py \
  --manifest-csv /tmp/feature_economy_release_plan/artifact_export_manifest_template.csv \
  --artifact-root "$ARTIFACT_ROOT" \
  --output-json /tmp/feature_economy_release_plan/artifact_export_copy_report.json
```

The copy step only copies files listed in `required_files` for rows marked
`READY` or `COPIED`; it skips `TODO`, `OMIT`, and `NEEDS_CONVERSION` rows by
default. Use `--dry-run` before a large export and `--force` only when you mean
to replace an existing public artifact.

For a full release staging pass, use the one-command wrapper:

```bash
bash scripts/stage_artifact_bundle_from_manifest.sh \
  --manifest-csv /tmp/feature_economy_release_plan/artifact_export_manifest_template.csv \
  --run-plan-json /tmp/feature_economy_release_plan/reproduction_run_plan.json \
  --artifact-root "$ARTIFACT_ROOT" \
  --release-output-dir /tmp/feature_economy_release_assets \
  --bundle-name feature_economy_artifacts_v1 \
  --sanitize-replace "${PRIVATE_ARTIFACT_PREFIX}/=${ARTIFACT_ROOT}/" \
  --force
```

This wrapper runs audit, copy, `check-bundle --require-complete`,
`index-artifacts --require-valid`, `make-tables`, and
`package_artifact_bundle.sh` in order. Use a subset run plan if you are staging
a partial release; the full public v1 run plan expects all `66` rows.

## Required Artifact Root Layout

Use the layout defined in `docs/release_artifact_bundle_layout.md`:

```text
feature_economy_artifacts_v1/
  run_plan/
  manifests/
  features/
  codes/
  probes/
  analysis/
  index/
  tables/
  figures/
  README.md
```

The first real bundle may omit `figures/` only if figure generation is not part
of the release asset. It should not omit `tables/`, because CSV tables are the
lowest-friction way for readers to inspect the paper chain.

## Minimum Per-Stage Files

For every planned row, `check-bundle --require-complete` enforces the required
files below:

| Stage | Required files |
| --- | --- |
| `feature_extraction` | `features.npz`, `feature_extraction_summary.json`, `run_manifest.json` |
| `sae_code_extraction` | `codes.npz`, `sae_code_summary.json`, `run_manifest.json` |
| `native_probe` | `native_probe_summary.json`, `probe_logits.npz`, `run_manifest.json` |
| `sae_probe` | `sae_probe_summary.json`, `probe_logits.npz`, `run_manifest.json` |
| `availability` | `availability_summary.json`, `run_manifest.json` |
| `feature_ranking` | `task_feature_ranking.json`, `run_manifest.json` |
| `contribution_scores` | `contribution_scores.npz`, `contribution_scores_summary.json`, `run_manifest.json` |
| `subset_usage` | `subset_usage_summary.json`, `run_manifest.json` |
| `feature_ablation` | `feature_ablation_summary.json`, `run_manifest.json` |

Recommended but not required next to arrays:

- `array_validation.json` for `features.npz`;
- `array_validation.json` for `codes.npz`;
- `array_validation.json` for `probe_logits.npz`;
- `array_validation.json` for `contribution_scores.npz`.

## What Should Be Exported From The Private Workbench

For public v1, prefer exporting already-materialized saved arrays and summaries
over rerunning private training scripts. Each exported artifact should preserve:

- the same public `model_id`, `task_id`, and `sae_id` names used in `configs/`;
- a `run_manifest.json` with command, code version, inputs, and output path;
- portable paths or clearly marked non-portable provenance fields;
- metrics that match the corresponding summary schema;
- arrays that pass `validate-arrays` under the public contract.

If a private artifact uses a different naming convention, convert it at the
bundle boundary rather than changing public config identifiers.

## Acceptance Gates

The real bundle is publishable only after these commands pass from the
standalone public repo root:

```bash
export ARTIFACT_ROOT=/path/to/feature_economy_artifacts_v1
export PYTHONPATH=$PWD/src

python -m feature_economy.cli.main check-bundle \
  --run-plan-json "$ARTIFACT_ROOT/run_plan/reproduction_run_plan.json" \
  --artifact-root "$ARTIFACT_ROOT" \
  --output-json "$ARTIFACT_ROOT/run_plan/artifact_bundle_check_final.json" \
  --require-complete

python -m feature_economy.cli.main index-artifacts \
  --input-dir "$ARTIFACT_ROOT" \
  --output-json "$ARTIFACT_ROOT/index/artifact_index.json" \
  --output-csv "$ARTIFACT_ROOT/index/artifact_index.csv" \
  --require-valid

python -m feature_economy.cli.main make-tables \
  --artifact-index "$ARTIFACT_ROOT/index/artifact_index.json" \
  --output-dir "$ARTIFACT_ROOT/tables"

bash scripts/package_artifact_bundle.sh \
  --artifact-root "$ARTIFACT_ROOT" \
  --run-plan-json "$ARTIFACT_ROOT/run_plan/reproduction_run_plan.json" \
  --sanitize-replace "${PRIVATE_ARTIFACT_PREFIX}/=${ARTIFACT_ROOT}/" \
  --output-dir /tmp/feature_economy_release_assets \
  --bundle-name feature_economy_artifacts_v1 \
  --force
```

Optional figure gate:

```bash
bash scripts/package_artifact_bundle.sh \
  --artifact-root "$ARTIFACT_ROOT" \
  --run-plan-json "$ARTIFACT_ROOT/run_plan/reproduction_run_plan.json" \
  --sanitize-replace "${PRIVATE_ARTIFACT_PREFIX}/=${ARTIFACT_ROOT}/" \
  --output-dir /tmp/feature_economy_release_assets \
  --bundle-name feature_economy_artifacts_v1_with_figures \
  --with-figures \
  --force
```

## Release README Requirements

The artifact bundle `README.md` should include:

- release date and paper version;
- exact public repo commit used for validation;
- whether figures are included;
- dataset and checkpoint redistribution limitations;
- a short description of each top-level directory;
- commands to verify checksum, unpack, run `check-bundle`, index artifacts, and
  regenerate tables;
- known caveats, especially that public v1 is an artifact-first saved-array
  reproduction rather than a raw-data training reproduction.

Generate the initial README with:

```bash
python scripts/create_artifact_bundle_readme.py \
  --run-plan-json "$ARTIFACT_ROOT/run_plan/reproduction_run_plan.json" \
  --output-readme "$ARTIFACT_ROOT/README.md" \
  --bundle-name feature_economy_artifacts_v1 \
  --public-repo-commit "$(git rev-parse --short HEAD)"
```

Review the generated README before packaging, especially the dataset,
checkpoint, and license notes.

## Non-Goals For The First Real Bundle

Do not block the first real bundle on:

- raw ImageNet, NYUv2, ADE20K, or CLEVR redistribution;
- raw I-JEPA checkpoint loading;
- exact private cluster launchers;
- every second-last-layer or appendix-only experiment;
- final manuscript figure styling.

Those can become later releases once the first public saved-array chain is
auditable.

## Open Export Checklist

Before the first real bundle can be packaged, fill this checklist with concrete
artifact paths or mark the item intentionally omitted:

| Item | Status | Evidence path or note |
| --- | --- | --- |
| Generated public `reproduction_run_plan.json` copied into `run_plan/` | TODO |  |
| Generated `artifact_export_manifest_template.csv/json` reviewed | TODO |  |
| Filled export manifest passes read-only audit | TODO |  |
| Ready rows copied into public artifact root | TODO |  |
| Bundle `README.md` written | TODO |  |
| DINO native probe artifacts for 4 tasks | TODO |  |
| I-JEPA native probe artifacts for 4 tasks | TODO |  |
| DINO SAE-code probe artifacts for 4 tasks | TODO |  |
| I-JEPA SAE-code probe artifacts for 4 tasks | TODO |  |
| DINO feature/code artifacts for 4 task eval splits | TODO |  |
| I-JEPA feature/code artifacts for 4 task eval splits | TODO |  |
| DINO ImageNet-val availability artifacts | TODO |  |
| I-JEPA ImageNet-val availability artifacts | TODO |  |
| Feature ranking artifacts for 8 model/task pairs | TODO |  |
| Contribution score artifacts for 8 model/task pairs | TODO |  |
| Subset usage artifacts for 8 model/task pairs | TODO |  |
| Feature ablation artifacts for 8 model/task pairs | TODO |  |
| `check-bundle --require-complete` passes | TODO |  |
| `index-artifacts --require-valid` passes | TODO |  |
| `make-tables` regenerates CSVs | TODO |  |
| `package_artifact_bundle.sh` emits sanitized archive/checksum/manifest | TODO |  |
| Checksum verified after unpacking | TODO |  |

## Full Public V1 Release Candidate

The first full saved-array release candidate was staged on 2026-05-13 at:

```text
${PRIVATE_ARTIFACT_ROOT}/public_repro_feature_economy_v1_candidate
```

This candidate merges the validated single-task slices into the full public v1
run matrix:

| Gate | Result |
| --- | --- |
| Run-plan rows | `66/66` complete |
| Artifact index | `190/190` valid records |
| Generated tables | `probe_scores.csv`, `availability_summary.csv`, `subset_usage_summary.csv`, `ablation_summary.csv` |
| Candidate root size | approximately 7.4 GiB |
| Metadata sanitizer | PASS, `0` remaining private-path hits |

The packaged release assets were written to:

```text
${PRIVATE_ARTIFACT_ROOT}/public_repro_release_assets
```

Release files:

| File | Size / status |
| --- | --- |
| `feature_economy_artifacts_v1.tar.gz` | approximately 7.4 GiB |
| `feature_economy_artifacts_v1.tar.gz.sha256` | archive checksum |
| `feature_economy_artifacts_v1_release_manifest.json` | release manifest |
| `feature_economy_artifacts_v1.tar.gz.part-00` ... `part-13` | 512 MiB each |
| `feature_economy_artifacts_v1.tar.gz.part-14` | approximately 361 MiB |
| `feature_economy_artifacts_v1.tar.gz.parts.sha256` | per-part checksums |

The split parts were also checked by concatenating them and comparing the
reassembled SHA256 against `feature_economy_artifacts_v1.tar.gz.sha256`.
For future releases, use the scripted helper instead of manual `split`:

```bash
python scripts/split_release_archive.py \
  --archive "${RELEASE_OUTPUT_DIR}/feature_economy_artifacts_v1.tar.gz" \
  --part-size 512M \
  --force
```

## Current Remote Trial Status

These trial bundles are smoke tests for the public saved-array contract. They
are not the final release assets, but they prove that the public analysis chain
can be generated from real private checkpoints and then checked by the public
repo tools.

| Trial | Status | Artifact root | Log |
| --- | --- | --- | --- |
| DINO ImageNet full saved-array candidate | PASS | `${PRIVATE_ARTIFACT_ROOT}/public_repro_dino_imagenet_trial` | `${PRIVATE_LOG_ROOT}/public_repro_dino_imagenet_trial` |
| I-JEPA ImageNet full saved-array candidate | PASS | `${PRIVATE_ARTIFACT_ROOT}/public_repro_ijepa_imagenet_trial` | `${PRIVATE_LOG_ROOT}/public_repro_ijepa_imagenet_trial` |
| DINO NYUv2 8-image dense smoke | PASS | `${PRIVATE_ARTIFACT_ROOT}/public_repro_dino_nyuv2_trial_smoke` | `${PRIVATE_LOG_ROOT}/public_repro_dino_nyuv2_trial_smoke/run.log` |
| I-JEPA NYUv2 8-image dense smoke | PASS | `${PRIVATE_ARTIFACT_ROOT}/public_repro_ijepa_nyuv2_trial_smoke` | `${PRIVATE_LOG_ROOT}/public_repro_ijepa_nyuv2_trial_smoke/run.log` |
| DINO NYUv2 full-val saved-array candidate | PASS | `${PRIVATE_ARTIFACT_ROOT}/public_repro_dino_nyuv2_trial` | `${PRIVATE_LOG_ROOT}/public_repro_dino_nyuv2_trial/run.log` |
| I-JEPA NYUv2 full-val saved-array candidate | PASS | `${PRIVATE_ARTIFACT_ROOT}/public_repro_ijepa_nyuv2_trial` | `${PRIVATE_LOG_ROOT}/public_repro_ijepa_nyuv2_trial/run.log` |
| DINO ADE20K 8-image dense smoke | PASS | `${PRIVATE_ARTIFACT_ROOT}/public_repro_dino_ade20k_trial_smoke` | `${PRIVATE_LOG_ROOT}/public_repro_dino_ade20k_trial_smoke/run.log` |
| I-JEPA ADE20K 8-image dense smoke | PASS | `${PRIVATE_ARTIFACT_ROOT}/public_repro_ijepa_ade20k_trial_smoke` | `${PRIVATE_LOG_ROOT}/public_repro_ijepa_ade20k_trial_smoke/run.log` |
| DINO ADE20K full-val saved-array candidate | PASS | `${PRIVATE_ARTIFACT_ROOT}/public_repro_dino_ade20k_trial` | `${PRIVATE_LOG_ROOT}/public_repro_dino_ade20k_trial/run.log` |
| I-JEPA ADE20K full-val saved-array candidate | PASS | `${PRIVATE_ARTIFACT_ROOT}/public_repro_ijepa_ade20k_trial` | `${PRIVATE_LOG_ROOT}/public_repro_ijepa_ade20k_trial/run.log` |
| DINO CLEVR/Count 8-image smoke | PASS | `${PRIVATE_ARTIFACT_ROOT}/public_repro_dino_clevr_count_trial_smoke` | `${PRIVATE_LOG_ROOT}/public_repro_dino_clevr_count_trial_smoke/run.log` |
| I-JEPA CLEVR/Count 8-image smoke | PASS | `${PRIVATE_ARTIFACT_ROOT}/public_repro_ijepa_clevr_count_trial_smoke` | `${PRIVATE_LOG_ROOT}/public_repro_ijepa_clevr_count_trial_smoke/run.log` |
| DINO CLEVR/Count full-val saved-array candidate | PASS | `${PRIVATE_ARTIFACT_ROOT}/public_repro_dino_clevr_count_trial` | `${PRIVATE_LOG_ROOT}/public_repro_dino_clevr_count_trial/run.log` |
| I-JEPA CLEVR/Count full-val saved-array candidate | PASS | `${PRIVATE_ARTIFACT_ROOT}/public_repro_ijepa_clevr_count_trial` | `${PRIVATE_LOG_ROOT}/public_repro_ijepa_clevr_count_trial/run.log` |

NYUv2 smoke validation on 2026-05-13 confirmed both models produce complete
single-model public bundles: `8/8` planned rows complete and `25/25` indexed
artifacts valid. The bridge writes a public-schema `val_manifest.jsonl` beside
the exported features so the public CLI does not depend on the private NYUv2
formal manifest field names.

NYUv2 full-val validation on 2026-05-13 then confirmed the same chain on the
654-image formal validation split for both DINO and I-JEPA. Each full-val
candidate completed `8/8` planned rows and indexed `25/25` valid artifacts.

ADE20K smoke validation on 2026-05-13 confirmed both models produce complete
single-model public dense-segmentation bundles on an 8-image validation slice:
`8/8` planned rows complete and `25/25` indexed artifacts valid. The bridge
writes a public-schema `val_manifest.jsonl` beside the exported features and
stores segmentation targets in the public `targets.npz` contract.

ADE20K full-val validation on 2026-05-13 then confirmed the same chain on the
2000-image validation split for both DINO and I-JEPA. Each full-val candidate
completed `8/8` planned rows and indexed `25/25` valid artifacts. The current
private artifact sizes are approximately 2.1 GiB for DINO and 3.0 GiB for
I-JEPA before final cross-task packaging.

CLEVR/Count smoke validation on 2026-05-13 confirmed both models produce
complete single-model public count-classification bundles on an 8-image
validation slice: `8/8` planned rows complete and `24/24` indexed artifacts
valid. Count classification has no dense target array, so the indexed artifact
count is one lower than dense depth/segmentation slices.

CLEVR/Count full-val validation on 2026-05-13 then confirmed the same chain on
the 15000-image validation split for both DINO and I-JEPA. Each full-val
candidate completed `8/8` planned rows and indexed `24/24` valid artifacts.
The current private artifact sizes are approximately 49 MiB for DINO and
76 MiB for I-JEPA before final cross-task packaging.
