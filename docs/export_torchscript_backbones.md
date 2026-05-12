# Exporting TorchScript Backbone Feature Modules

The public reproduction repo has two real feature-extraction paths:

- `--backend huggingface` for models exposed through HuggingFace hidden states,
  such as the public DINOv2 config.
- `--backend torchscript` for local checkpoints, including I-JEPA variants or
  future ViTs whose raw loaders are not part of this repo.

The TorchScript path is intentionally a feature-module contract, not a private
checkpoint loader. The exported module receives already-normalized images and
returns the target-layer features that should be saved to `features.npz`.

## Contract

Your exported module should:

- accept one positional input: `pixel_values` with shape `[batch, 3, crop, crop]`;
- assume the transform in `configs/models/<model_id>.yaml` has already been
  applied by `feature-economy extract-features`;
- run in eval mode, with stochastic augmentations and dropout disabled;
- return target-layer features directly as a tensor, tuple/list, or dict;
- make layer choice explicit in the exported module name or output directory.

Common output shapes:

- global/classification features: `[batch, feature_dim]`;
- token features: `[batch, tokens, feature_dim]`;
- dense maps: `[batch, height, width, feature_dim]`.

After extraction, validate the saved array:

```bash
python -m feature_economy.cli.main validate-arrays \
  --npz "$ARTIFACT_ROOT/features/ijepa_vit_h14/imagenet_val_l31/features.npz" \
  --kind features \
  --task-type classification \
  --manifest "$DATA_ROOT/imagenet/val_manifest.jsonl" \
  --expected-split val
```

## Toy Contract Smoke

This toy module is not a scientific model. It only demonstrates the public
TorchScript interface:

```bash
python - <<'PY'
import json
from pathlib import Path
from PIL import Image

root = Path("/tmp/feature_economy_torchscript_toy_inputs")
(root / "images").mkdir(parents=True, exist_ok=True)
rows = []
for index, label in enumerate([0, 1], start=1):
    image_path = root / "images" / f"toy_{index:04d}.png"
    Image.new("RGB", (260, 260), color=(64 * index, 32 * index, 16 * index)).save(image_path)
    rows.append({"image": str(image_path), "split": "val", "label": label})
(root / "manifest.jsonl").write_text(
    "\n".join(json.dumps(row) for row in rows) + "\n",
    encoding="utf-8",
)
PY

python scripts/export_torchscript_toy_feature_module.py \
  --output /tmp/tiny_feature_module.pt \
  --feature-dim 3

python -m feature_economy.cli.main extract-features \
  --config-root configs \
  --backend torchscript \
  --checkpoint /tmp/tiny_feature_module.pt \
  --manifest /tmp/feature_economy_torchscript_toy_inputs/manifest.jsonl \
  --task-type classification \
  --model-id ijepa_vit_h14 \
  --expected-split val \
  --batch-size 1 \
  --output-dir /tmp/feature_economy_torchscript_toy_features
```

## Adapting I-JEPA Or Another Local ViT

Wrap your local model in a small `torch.nn.Module` that owns the raw checkpoint
loader outside this repo and returns only the layer representation used by the
public pipeline. A typical adapter looks like:

```python
import torch


class LocalVitFeatureModule(torch.nn.Module):
    def __init__(self, backbone, layer_index: int):
        super().__init__()
        self.backbone = backbone.eval()
        self.layer_index = layer_index

    def forward(self, pixel_values):
        # Replace this block with the model-specific hidden-state API.
        # The returned tensor should already be the representation you want to
        # save as `features`: global, token, or dense-map features.
        hidden_states = self.backbone(pixel_values, return_hidden_states=True)
        return hidden_states[self.layer_index]


backbone = load_your_local_backbone_checkpoint("/path/to/checkpoint")
module = LocalVitFeatureModule(backbone, layer_index=31).eval()
example = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
traced = torch.jit.trace(module, example)
traced.save("/path/to/ijepa_l31_feature_module.pt")
```

Some local backbones cannot be traced cleanly because they return Python objects
or depend on dynamic control flow. In that case, export a thinner wrapper whose
`forward` returns a plain tensor, or use `torch.jit.script` if scripting works
better for that model.

## Extraction Command

Once exported:

```bash
python -m feature_economy.cli.main extract-features \
  --config-root configs \
  --backend torchscript \
  --checkpoint "$ARTIFACT_ROOT/checkpoints/ijepa_l31_feature_module.pt" \
  --manifest "$DATA_ROOT/imagenet/val_manifest.jsonl" \
  --task-type classification \
  --model-id ijepa_vit_h14 \
  --layer 31 \
  --expected-split val \
  --batch-size 16 \
  --device cuda \
  --output-dir "$ARTIFACT_ROOT/features/ijepa_vit_h14/imagenet_val_l31"
```

If the module returns a tuple/list, select an output with `--output-index`. If it
returns a dict, select an output with `--output-key`.

## Failure Modes To Check First

- Transform mismatch: DINO and I-JEPA should use the same resize/crop/normalize
  policy unless a deliberate experiment says otherwise.
- Layer ambiguity: record whether the module returns the final, second-last, or
  another representative layer.
- Special-token ambiguity: document whether features include CLS/register
  tokens or patch tokens only.
- Shape mismatch: run `validate-arrays` before SAE-code extraction or probing.
- Device mismatch: use `--device cpu` for debugging and `--device cuda` only
  after the CPU path writes a valid `features.npz`.
