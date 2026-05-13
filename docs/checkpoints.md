# Checkpoints

The public repo will not ship full model or SAE checkpoints by default.

Expected checkpoint classes:

- DINOv2 backbone checkpoints.
- I-JEPA backbone checkpoints.
- ImageNet-trained SAE checkpoints for selected layers.
- Frozen task-probe checkpoints when reproducing table/figure generation from
  saved artifacts.

Configs should use environment variables such as `${SAE_ROOT}` and
`${ARTIFACT_ROOT}` rather than private absolute paths.

Validation:

```bash
feature-economy check-models --config-root configs
```

To require all checkpoint placeholders to resolve on the current machine:

```bash
SAE_ROOT=/path/to/saes \
feature-economy check-models \
  --config-root configs \
  --require-resolved-checkpoints
```

The registry validates model/SAE links and exposes resolved SAE checkpoint
paths without instantiating any model weights.

## TorchScript Backbone Feature Module Contract

`feature-economy extract-features --backend torchscript` is the public bridge for
I-JEPA and other local backbones whose raw checkpoint loaders are not included
in this repo. See `docs/export_torchscript_backbones.md` for a full export
guide and toy contract smoke. The TorchScript module should:

- accept one argument: normalized image tensors with shape `[batch, 3, crop, crop]`;
- use the model transform policy from `configs/models/*.yaml`;
- return the target-layer features directly as a tensor, tuple/list, or dict;
- keep layer selection inside the exported module, while `--layer` records the
  layer identity in the public artifact metadata.

Tuple/list outputs can be selected with `--output-index`; dict outputs can be
selected with `--output-key`. This keeps raw model-loading code outside the
public runner while preserving the same `features.npz`,
`feature_extraction_summary.json`, and `run_manifest.json` contract used by the
HuggingFace DINO backend.

## Lightweight SAE Checkpoint Contract

`feature-economy extract-sae-codes --backend linear-topk` accepts a small
inference-only SAE checkpoint. The public dependency-light format is `.npz`
with these arrays:

- `encoder_weight`: shape `[input_dim, code_dim]`.
- `encoder_bias`: shape `[code_dim]`, optional and defaults to zeros.
- `decoder_bias`: shape `[input_dim]`, optional and defaults to zeros.
- `decoder_weight`: shape `[code_dim, input_dim]`, optional for SAE-code
  extraction but required for Module F native-subspace ablation.

The backend applies the SAE runtime normalization from the SAE config, subtracts
`decoder_bias`, applies the linear encoder, ReLU, and TopK sparsification. This
format is intentionally narrower than the private training checkpoint format so
public runs do not need the full SAE training package tree.

Module F native-subspace ablation additionally needs the SAE decoder directions
because it removes projections onto selected decoder directions in the SAE
runtime-normalized hidden coordinate system. For current paper SAEs this means
`layer_norm(x) - decoder_bias`; after projection removal the command inverts the
normalization and re-encodes through the frozen SAE/probe path. Checkpoints
without `decoder_weight` can still run ordinary SAE-code extraction, but
`feature-economy ablate-native-subspace` will reject them.

To convert a full checkpoint that stores standard SAE weights as `W_enc`,
`b_enc`, and `b_dec`:

```bash
feature-economy convert-sae-checkpoint \
  --input-checkpoint /path/to/final_sae.pt \
  --output-checkpoint /path/to/lightweight_sae.npz
```

The converter includes lightweight pickle stubs for legacy `vit_prisma`
checkpoints so public reproduction can extract tensor attributes without
installing the full SAE training package tree. This path is intended only for
dependency-light inference conversion; if a checkpoint uses a different
activation mechanism, add a dedicated public backend instead of silently
pretending it is a linear TopK SAE.

Gated SAE checkpoints are rejected by default because their activation path is
not equivalent to the linear TopK backend. Add a dedicated gated backend before
using gated checkpoints for paper reproduction.

The registry also returns a `TransformPolicy` for each model/SAE pair. Public v0
uses a shared ImageNet-style resize/crop/normalize policy so DINO and I-JEPA
runner ports can avoid silently diverging preprocessing choices.

Before running real backbone or SAE extraction, check optional experiment
dependencies:

```bash
feature-economy check-runtime --profile experiments --require
```

For DINO HuggingFace extraction on offline clusters, pre-download the model
directory on a machine with network access, copy it to the target machine, and
pass it as an override:

```bash
feature-economy extract-features \
  --backend huggingface \
  --model-id dino_v2_base \
  --hf-name-or-path /path/to/facebook/dinov2-base \
  --local-files-only \
  ...
```
