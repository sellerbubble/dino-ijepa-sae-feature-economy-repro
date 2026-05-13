# Datasets

The public repo should provide dataset adapters and manifest schemas, not
redistribute restricted datasets.

Paper v0 tasks:

- ImageNet-1K classification.
- NYUv2 depth.
- ADE20K semantic segmentation.
- CLEVR/Count image-only counting from scene annotations.

Each task should expose train/validation manifests with explicit split
assumptions and metric definitions.

Tiny fixtures should be used for CI and schema tests.

Current manifest contract:

- Common fields: `image`, `split`.
- Classification/counting: add `label`.
- NYUv2-style depth: add `depth`.
- ADE20K-style segmentation: add `segmentation`.

For CLEVR/Count, use the image-only count definition from official scene
annotations:

```text
object_count = len(scene["objects"])
label = object_count - 3
```

This yields labels `0..7` for object counts `3..10`. Do not derive labels from
question JSON files; those define a question-conditioned task rather than the
paper's CLEVR/Count profile.

For dense profiles, `depth` and `segmentation` point to manifest-relative or
absolute target files. The public `export-targets` command supports `.npy`,
`.npz`, and image files, then writes a probe-ready `targets.npz` aligned to the
spatial feature grid:

```bash
feature-economy export-targets \
  --manifest /path/to/nyuv2/val_manifest.jsonl \
  --task-type dense_depth \
  --expected-split val \
  --features-npz /path/to/features.npz \
  --output-dir /path/to/targets/nyuv2_depth/val
```

For official ADE20K annotations, masks use `0` for background/ignored pixels
and `1..150` for classes. The public full-profile runner converts this to the
probe convention `0..149` plus ignore index `255`:

```bash
feature-economy export-targets \
  --manifest /path/to/ade20k/val_manifest.jsonl \
  --task-type dense_segmentation \
  --expected-split val \
  --features-npz /path/to/features.npz \
  --segmentation-ignore-value 0 \
  --segmentation-label-offset -1 \
  --segmentation-output-ignore-index 255 \
  --output-dir /path/to/targets/ade20k_segmentation/val
```

If your segmentation masks are already stored as `0..149` with `255` ignored
pixels, omit the three segmentation remapping flags.

Validation command:

```bash
feature-economy check-manifest \
  --manifest tests/fixtures/tiny_nyuv2_manifest.jsonl \
  --task-type dense_depth \
  --expected-split val
```

Image preprocessing inspection command:

```bash
feature-economy inspect-images \
  --config-root configs \
  --manifest /path/to/manifest.jsonl \
  --task-type classification \
  --model-id dino_v2_base \
  --expected-split val \
  --limit 8 \
  --output-json /tmp/image_inspection.json
```

Current adapter boundary:

- `ManifestDataset` reads validated JSONL rows into metadata-only
  `ManifestRecord` objects.
- It resolves relative paths against the manifest directory.
- `inspect-images` loads RGB images and applies the model transform policy
  using PIL/NumPy. It does not instantiate torch, DINO, I-JEPA, or SAE modules.
- Future task adapters should build on this manifest layer rather than parsing
  task-specific JSONL files independently.
