# Expected Results And Acceptance Bands

This repo is designed for rerunning the DINO/I-JEPA SAE Feature Economy
experiments from public code, configs, user-provided datasets, and user-provided
checkpoints. It does not require users to download paper-scale feature caches by
default.

The expected outcome is **close scientific agreement**, not bit-identical
numbers. Small differences can come from hardware, image decoding libraries,
probe seeds, exact checkpoint variants, SAE checkpoints, and whether users run
closed-form lightweight probes or paper-scale trainers.

## What Should Reproduce

A successful rerun should recover the main qualitative pattern:

- DINO has stronger native frozen-probe transfer than I-JEPA on the configured
  ImageNet-1K, NYUv2, ADE20K, and CLEVR/Count settings.
- In SAE feature space, DINO uses a broader set of moderately active features.
- I-JEPA has more inactive or rarely used features and places more task burden
  on smaller high-usage feature sets.
- Task-selected top-k features should differ from bucket-matched random
  features.
- Removing high-usage task-recruited features should hurt more than removing
  matched random features, with the clearest concentration in I-JEPA.

If these signs do not appear, first check dataset splits, image transforms,
model layer IDs, SAE checkpoints, and ranking method.

## Reference Scores

These values are paper-reference anchors. Treat them as expected-regime checks,
not strict pass/fail thresholds.

| Task | Metric | DINO native | I-JEPA native | DINO SAE-code | I-JEPA SAE-code |
| --- | --- | ---: | ---: | ---: | ---: |
| ImageNet-1K | top-1 | 0.806 | 0.704 | 0.688 | 0.433 |
| ImageNet-1K | top-5 | 0.959 | 0.898 | 0.892 | 0.667 |
| NYUv2 depth | RMSE, lower is better | 0.776 | 0.888 | 0.902 | 1.008 |
| NYUv2 depth | delta1 | 0.652 | 0.590 | 0.562 | 0.517 |
| ADE20K segmentation | mIoU | 0.299 | 0.218 | 0.378 | 0.209 |
| CLEVR/Count | accuracy | 0.612 | 0.541 | 0.407 | 0.379 |

The SAE-code scores are not intended to beat native probes. They define the
feature-level coordinate system used for Availability, Access, and Allocation
analysis.

## Suggested Acceptance Bands

Use these as practical sanity checks when rerunning the full chain:

- Native and SAE-code probe scores should usually be within a few absolute
  points for classification/counting accuracy and segmentation mIoU if the same
  checkpoints, splits, transforms, and probe recipe are used.
- Depth RMSE may vary more across probe recipes, but the DINO-vs-I-JEPA ordering
  should remain stable under matched settings.
- Availability statistics should preserve the broad contrast: DINO more evenly
  uses its SAE basis; I-JEPA has a larger low-fired tail and a hotter recruited
  subset.
- Allocation statistics should preserve the contrast between task-recruited
  high-usage features and bucket-matched random controls.

When numbers fall outside these bands, report the model checkpoint, layer, SAE
setting, dataset split, transform config, ranking method, probe recipe, random
seed, and hardware/backend details.

## Optional Audit Bundle

The GitHub Release `v1-saved-array-20260513` contains a large split tarball with
paper-used saved arrays and generated tables. It exists for audit and debugging:
users can inspect the evidence chain or compare their rerun artifacts against
the archived version.

The audit bundle is not the primary reproduction path. New experiments should
prefer generating their own features, SAE codes, probes, rankings, and ablation
artifacts with the commands in `docs/release_quickstart.md` and
`docs/canonical_chain_runbook.md`.
