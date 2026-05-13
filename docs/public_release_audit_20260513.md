# Public Reproduction Release Audit

Date: 2026-05-13

This audit checks the current `public_repro/` slice against the goal of a
public, executable DINO/I-JEPA SAE Feature Economy reproduction repo.

## Objective Restated As Release Criteria

The public repo should let external readers:

1. run a checkpoint-free smoke chain that validates the command surface;
2. prepare real dataset manifests and inspect shared image preprocessing;
3. extract or provide DINO/I-JEPA native features under a stable array contract;
4. convert or provide SAE checkpoints and extract SAE codes;
5. run native and SAE-code probes for classification, counting, depth, and
   segmentation saved arrays;
6. compute AAA Feature Economy artifacts: Availability, Access, and Allocation;
7. index artifacts, then export paper-facing CSV tables and overview figures;
8. add new models, layers, tasks, SAEs, and analyses without copying private
   workbench scripts.

The current public target is a full paper-style rerun chain with smoke tests for
code health and optional saved-array audit support. It is not intended to
redistribute private data, checkpoints, or one-off private cluster launchers.
The intended public v1 boundary is fixed in `docs/public_v1_scope.md`.

## Prompt-To-Artifact Checklist

| Requirement | Current evidence | Verification status | Remaining gap |
| --- | --- | --- | --- |
| One-command smoke chain | `scripts/reproduce_smoke.sh` | Verified by full smoke run on 2026-05-13. It runs tests, config/model/manifest checks, fixture probes, dense saved-array probes, feature/code extraction, toy TorchScript extraction, AAA analyses, artifact indexing, tables, and optional figures. | Smoke uses tiny fixtures, not scientific results. |
| Tiny complete artifact bundle | `scripts/build_tiny_artifact_bundle.sh` | Verified locally on 2026-05-13. It builds one DINO/ImageNet SAE vertical slice, passes `check-bundle --require-complete` with 9/9 rows complete, indexes 24/24 valid artifacts, and exports CSV tables. | Uses fixture features/codes and tiny labels; it demonstrates contract completion, not scientific performance. |
| Expected results | `docs/expected_results.md` | Defines reference scores, qualitative acceptance bands, and the difference between approximate reruns and bit-identical audit bundles. | Bands are practical sanity checks, not formal statistical confidence intervals. |
| Release artifact bundle layout | `docs/release_artifact_bundle_layout.md` | Documents expected top-level directories, required files by run-plan stage, validation commands, and publishing checklist for optional saved-array audit assets. | Does not itself provide real paper artifacts. |
| First real saved-array bundle plan | `docs/first_real_saved_array_bundle_plan.md` | Defines the public v1 model/task/SAE matrix, expected 66 run-plan rows, task-level staging profiles, acceptance gates, and an export checklist for the optional audit release asset. Updated with the first full public v1 candidate: `66/66` rows complete, `190/190` valid indexed artifacts, sanitized archive packaged, and 15 GitHub-release-sized split parts created. | Keep this document synchronized with future release asset layouts. |
| Artifact export manifest template | `scripts/create_artifact_export_manifest.py` | Converts a generated run plan into a CSV/JSON handoff sheet with one row per public artifact target, required files, source placeholder, conversion flag, and validation hint. | Maintainers still need to fill concrete source artifact paths from the private workbench. |
| Artifact export manifest audit | `scripts/audit_artifact_export_manifest.py` | Read-only audit of a filled export manifest; checks status values, source directories, required source files, and resolved public target paths. | Does not copy files; it only gates whether the manifest is safe to act on. |
| Artifact export copy gate | `scripts/copy_artifact_export_manifest.py` | Copies only required files for manifest rows marked `READY` or `COPIED`, with dry-run support and overwrite refusal by default. | Does not convert artifact formats; rows marked `NEEDS_CONVERSION` remain a separate step unless explicitly included. |
| Run-plan subset filter and profiles | `scripts/filter_reproduction_run_plan.py`, `scripts/prepare_real_artifact_slice.py` | Creates partial release run plans from the canonical full run matrix. `prepare_real_artifact_slice.py` supports single-model ImageNet, combined ImageNet, NYUv2, ADE20K, CLEVR/Count, and full-v1 profiles. | Profiles generate staging manifests, not proof that scientific artifacts are ready. |
| Artifact bundle README generator | `scripts/create_artifact_bundle_readme.py` | Generates a bundle `README.md` from the run plan and optional release manifest, including scope, stage counts, verification commands, checksum instructions, and caveats. | Maintainers should still review dataset/checkpoint/license wording before publishing. |
| One-command artifact staging | `scripts/stage_artifact_bundle_from_manifest.sh` | Chains manifest audit, controlled copy, bundle completeness check, artifact indexing, table export, README generation, sanitizer-aware release packaging, and optional figure export. | Requires a filled manifest whose rows match the run plan; partial releases need a subset run plan. |
| Release hygiene checker | `scripts/check_public_release.py` | Verified locally on 2026-05-13. Checks required files, executable bits, Markdown fences, README links, and private-path markers with an explicit allowlist. | Does not prove scientific metric correctness. |
| Standalone repo export | `scripts/export_public_repo.sh` | Copies the curated public slice into a clean standalone working tree, excludes caches/generated local artifact directories, and runs the release hygiene checker on the exported tree. Verified on 2026-05-13 by exporting to `/tmp`, running 113 unit tests, running `check_public_release.py`, and building the tiny artifact bundle from the exported tree. | Does not initialize git, push to GitHub, or attach real saved-array release assets. |
| Saved-array release packaging | `scripts/package_artifact_bundle.sh`, `scripts/split_release_archive.py` | Validates a completed artifact root, rebuilds index/tables, scans or rewrites text metadata for private absolute paths, and emits a tar.gz archive, SHA256 checksum, release manifest, and checksummed split parts. The first full public v1 saved-array candidate packaged successfully on 2026-05-13 with `0` remaining private-path hits and was uploaded to the public GitHub Release. | This is an optional audit path, not the default user workflow. |
| Standalone public CI | `.github/workflows/ci.yml` | Runs unit tests, release hygiene, `scripts/reproduce_smoke.sh`, and `scripts/build_tiny_artifact_bundle.sh` when `public_repro` is exported as a repository root. | Not executed by GitHub until the public repo/export is created. |
| Runtime/config/model gates | `check-runtime`, `check-configs`, `check-models` | Command help inspected; smoke executes all three. | Real checkpoint existence is only enforced when users pass `--require-resolved-checkpoints`. |
| Run matrix planning | `plan-runs`, `reproduction_run_plan.json`, `reproduction_run_plan.csv` | Unit-tested and smoke-tested; expands native, SAE, availability, Access, and Allocation rows from public configs. | This is a launch checklist, not an executor. |
| Full profile launcher | `scripts/run_full_profile.sh` | Dry-run verified locally on 2026-05-13 for `dino_imagenet_l11`, `ijepa_imagenet_l31`, `dino_nyuv2_l11`, and `ijepa_nyuv2_l31`. It chains runtime/config/model/manifest gates, feature extraction, dense target export when needed, native and SAE probes, SAE-code extraction, Availability, Access, Allocation, artifact indexing, tables, and optional figures. | Real-slice remote validation has covered the matched ImageNet vertical profiles and both NYUv2 dense profiles. Additional layers, tasks, and SAE settings should extend this launcher or add sibling profiles under the same artifact contract. |
| Artifact bundle completeness | `check-bundle`, `artifact_bundle_check.json` | Unit-tested; checks required files for every run-plan row and can fail with `--require-complete`. | Checks file presence, not scientific correctness of metrics. |
| Dataset manifest gate | `check-manifest`, `ManifestDataset`, `inspect-images`, `export-targets` | Smoke validates a dense-depth manifest; docs describe ImageNet/CLEVR/NYUv2/ADE20K manifest rows. Dense profiles can export manifest-referenced depth/mask files into the public `targets.npz` contract. | Real public dataset download/preprocessing scripts are not included. |
| Shared transform policy | `configs/models/*.yaml`, `models/transforms.py`, `inspect-images` | Implemented with PIL/NumPy resize, center crop, RGB conversion, ImageNet normalization. | Public exactness depends on users using the same configs for exported feature modules. |
| DINO feature extraction | `extract-features --backend huggingface` | CLI is implemented and documented; output contract is shared with fixture/TorchScript backends. | Not smoke-tested with a real HuggingFace checkpoint because smoke remains checkpoint-light. |
| I-JEPA/local feature extraction | `extract-features --backend huggingface/torchscript`, `IJEPA_HF_NAME_OR_PATH`, `docs/export_torchscript_backbones.md`, `scripts/export_torchscript_toy_feature_module.py` | Smoke exports a toy TorchScript feature module, extracts features from real tiny PNGs, validates arrays, and indexes the artifacts. The full-profile launcher can also use a transformers-compatible local I-JEPA checkpoint. | Exact compatibility depends on the user's installed `transformers` version and checkpoint format. TorchScript remains the stable fallback. |
| Array contracts | `docs/array_contracts.md`, `validate-arrays`, `artifacts/arrays.py` | Unit-tested for features, codes, probe logits, dense targets, row mismatch; smoke validates `features`, `codes`, and dense `probe_logits`. | Does not validate every possible private artifact variant, only the public contract. |
| SAE-code extraction | `extract-sae-codes --backend linear-topk`, `convert-sae-checkpoint`, `docs/checkpoints.md` | Fixture and lightweight TopK paths implemented; smoke runs fixture code extraction and array validation. | Gated SAE checkpoints intentionally rejected until a dedicated public backend exists. |
| Native probes | `probe-native --backend fixture/linear-probe` | Unit-tested for classification and dense segmentation; smoke runs fixture ImageNet and dense-depth saved-array probes. Classification/counting token maps are mean-pooled before closed-form ridge so real DINO token features do not get flattened into an oversized readout. | Closed-form ridge is a lightweight public probe, not the private paper-scale trainer. |
| SAE-code probes | `probe-sae --backend fixture/linear-probe` | Unit-tested and smoke-tested for counting fixture and dense segmentation saved-array probes. Classification/counting SAE token maps are mean-pooled before probing so saved weights stay aligned with the SAE feature dimension for ranking and ablation. | Same paper-scale trainer caveat. |
| Availability | `compute-usage`, `availability_summary.json` schema, table/figure consumers | Unit-tested and smoke-tested with fixture and saved-code paths. | Dense usage-rate conventions may need a dedicated mode if exact paper artifacts require per-pixel rates. |
| Access | `rank-features`, `compute-contributions`, `compute-subset-usage`, `task_feature_ranking.json`, `contribution_scores.npz`, `subset_usage_summary.json` | Unit-tested and smoke-tested, including matched random controls. `rank-features` supports `probe_weight`, `validation_contribution`, and `hybrid` ranking methods; `compute-contributions` writes validation-contribution scores from saved linear probes. The full-profile classification launcher defaults to scalable `true_class_logit_drop`, with `exact_metric_drop` available for small audit runs. | Contribution scoring uses the public linear probe, not the private paper trainer. |
| Allocation | `ablate-features --backend linear-probe`, `feature_ablation_summary.json` | Unit-tested for classification, depth, segmentation; smoke runs classification and dense-segmentation ablations. | Native-space ablation and perturbation modules from the paper are not in the public slice yet. |
| Artifact provenance | `index-artifacts`, `run_manifest.json`, schema validators, `current_git_commit()` | Smoke indexes each artifact family with `--require-valid`; array validation reports are valid index records; run manifests record the current git commit or `FEATURE_ECONOMY_GIT_COMMIT` override. | Falls back to `"unknown"` only when no git metadata or override is available. |
| Tables | `make-tables`, `paper/tables.py` | Smoke exports probe, availability, subset usage, and ablation CSV tables. | Final manuscript formatting is separate from reproducibility CSV export. |
| Figures | `make-figures`, `paper/figures.py` | Smoke exports overview figures when matplotlib is installed. | Final designed paper figures are not included. |
| Full reproduction guide | `docs/full_reproduction.md` | Defines the default full paper-style rerun path, required inputs, outputs, resource expectations, and acceptance checks. | Paper-style runners still need to be ported and validated module by module. |
| Quickstart path | `docs/release_quickstart.md` | Provides a 5-minute smoke path, a compact full vertical slice template, and an optional audit-bundle path. | Still points to the full runbook for multi-task/multi-model runs. |
| Extension path | `docs/extending_models_tasks.md`, config directories, TorchScript export guide | Docs describe adding models, SAEs, tasks, and analyses through configs/adapters. | Needs more examples once a second real model/layer/task is added publicly. |

## Release Verdict

Current status: **public alpha has executable command surfaces and smoke tests;
the default target is now full paper-style rerun, while the first full public v1
saved-array release candidate remains an optional audit artifact**.

It is suitable for:

- validating the public command surface;
- demonstrating the paper evidence chain on fixtures and user-generated arrays;
- allowing external users to generate or plug in DINO/I-JEPA features, SAE
  codes, and probe outputs under documented contracts;
- extending the project through configs and thin adapters.

It is not yet suitable for claiming:

- bit-identical reproduction of private paper-scale intermediate arrays;
- official raw I-JEPA checkpoint loading;
- equivalence to every private training-loop launcher;
- final manuscript figure reproduction.

## Highest-Value Next Actions

1. Export the standalone repository tree with `scripts/export_public_repo.sh`,
   initialize a public GitHub repository from that tree, and let the standalone
   CI run before publishing saved-array assets.
2. Run `scripts/run_full_profile.sh dino_imagenet_l11` on a clean GPU machine
   with real manifests, a DINO checkpoint, and an SAE checkpoint.
3. Extend the same full-profile launcher pattern to ADE20K and CLEVR/Count,
   then run clean real-slice validations for those task families.
4. Revisit exact paper-scale probe training only if the escalation rule in
   `docs/public_v1_scope.md` is satisfied.
