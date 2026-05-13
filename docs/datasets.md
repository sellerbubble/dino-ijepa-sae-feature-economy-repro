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
