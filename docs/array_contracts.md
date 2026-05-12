# Public Array Contracts

The public reproduction chain passes saved `.npz` arrays between backbone
extraction, SAE encoding, linear probes, ranking, and ablation. These contracts
make external exports auditable before they enter the paper-analysis pipeline.

Validate any exported or converted array with:

```bash
feature-economy validate-arrays \
  --npz /path/to/features.npz \
  --kind features \
  --task-type classification \
  --manifest /path/to/val_manifest.jsonl \
  --expected-split val
```

## `features.npz`

Required keys:

- `features`: floating array with shape `[num_examples, ..., feature_dim]`.

The first dimension should match the manifest row count when `--manifest` is
provided. Global features usually have shape `[N, D]`; dense token/spatial maps
can use shape `[N, H, W, D]` or another task-specific middle shape.

## `codes.npz`

Required keys:

- `codes`: floating array with shape `[num_examples, ..., code_dim]`.

The shape should align with the corresponding feature or task target layout.
Classification/counting probes usually consume `[N, code_dim]`; dense probes
consume `[N, H, W, code_dim]`.

## `targets.npz`

Required keys:

- `targets`: dense target array with shape `[num_examples, ...]`.

For `dense_depth`, `targets` must be floating. For `dense_segmentation`,
`targets` must be integer labels. The first dimension should match the dense
task manifest when `--manifest` is provided.

## `probe_logits.npz`

Classification and counting probes require:

- `logits`: floating array with shape `[num_examples, num_classes]`;
- `weights`: floating array with shape `[feature_dim + 1, num_classes]`;
- `classes`: one-dimensional class-id array;
- `labels`: one-dimensional label array.

Dense depth probes require:

- `prediction`: floating array with the same shape as `targets`;
- `weights`: floating array with shape `[feature_dim + 1]`;
- `targets`: floating depth target array.

Dense segmentation probes require:

- `logits`: floating array with shape `target_shape + [num_classes]`;
- `weights`: floating array with shape `[feature_dim + 1, num_classes]`;
- `classes`: one-dimensional class-id array;
- `targets`: integer segmentation target array.

These probe outputs are the shared substrate for feature ranking and feature
ablation. If the validator fails, fix the upstream export rather than patching
downstream analysis scripts around the mismatch.

## `contribution_scores.npz`

`rank-features --ranking-method validation_contribution` and
`rank-features --ranking-method hybrid` accept an optional contribution score
file. The public `compute-contributions` command writes this file by
single-feature zero-ablation of a saved linear probe. Required key by default:

- `validation_contribution_score`: floating array with shape `[code_dim]`.

Use `--contribution-key` if the score array is stored under another key. Larger
scores are treated as more task-relevant. The hybrid ranking combines normalized
probe-weight score and normalized contribution score with `--hybrid-alpha`.

Validate the file before ranking:

```bash
python -m feature_economy.cli.main validate-arrays \
  --npz /path/to/contribution_scores.npz \
  --kind contribution_scores
```
