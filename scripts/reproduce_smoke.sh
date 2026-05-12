#!/usr/bin/env bash
# Role: run the public reproduction smoke chain.
# Status: public smoke runner
# Used by: public_repro README and CI-style local validation
# Inputs: public_repro configs and tests/fixtures
# Outputs: smoke artifacts, artifact indexes, CSV tables, and optional figures under OUTPUT_ROOT
# Safe to move/delete?: keep; this is the public reproduction package's end-to-end smoke runner.
# Notes: This does not run DINO, I-JEPA, SAE encoders, or full datasets.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/feature_economy_public_smoke}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

rm -rf "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

python -m unittest discover -s tests

python -m feature_economy.cli.main check-runtime \
  --profile smoke \
  --require \
  --json-output "${OUTPUT_ROOT}/runtime_smoke.json"

python -m feature_economy.cli.main check-configs \
  --config-root configs

python -m feature_economy.cli.main plan-runs \
  --config-root configs \
  --output-json "${OUTPUT_ROOT}/run_plan/reproduction_run_plan.json" \
  --output-csv "${OUTPUT_ROOT}/run_plan/reproduction_run_plan.csv"

python -m feature_economy.cli.main check-bundle \
  --run-plan-json "${OUTPUT_ROOT}/run_plan/reproduction_run_plan.json" \
  --artifact-root "${OUTPUT_ROOT}" \
  --output-json "${OUTPUT_ROOT}/run_plan/artifact_bundle_check_initial.json"

python -m feature_economy.cli.main check-models \
  --config-root configs

python -m feature_economy.cli.main check-manifest \
  --manifest tests/fixtures/tiny_nyuv2_manifest.jsonl \
  --task-type dense_depth \
  --expected-split val

python -m feature_economy.cli.main eval-native-fixture \
  --manifest tests/fixtures/tiny_clevr_count_eval_manifest.jsonl \
  --task-type count_classification \
  --task-id clevr_count \
  --model-id dino_v2_base \
  --expected-split val \
  --output-dir "${OUTPUT_ROOT}/fixture_native_clevr"

python -m feature_economy.cli.main probe-native \
  --fixture-manifest tests/fixtures/tiny_imagenet_eval_manifest.jsonl \
  --task-type classification \
  --task-id imagenet_1k \
  --model-id dino_v2_base \
  --expected-split val \
  --output-dir "${OUTPUT_ROOT}/probe_native_fixture_imagenet"

python -m feature_economy.cli.main probe-sae \
  --fixture-manifest tests/fixtures/tiny_clevr_count_eval_manifest.jsonl \
  --task-type count_classification \
  --task-id clevr_count \
  --model-id ijepa_vit_h14 \
  --sae-id ijepa_l31_topk32_exp4 \
  --expected-split val \
  --output-dir "${OUTPUT_ROOT}/probe_sae_fixture_clevr"

mkdir -p "${OUTPUT_ROOT}/dense_probe_tiny_inputs"
python - <<PY
import json
from pathlib import Path
import numpy as np

root = Path("${OUTPUT_ROOT}") / "dense_probe_tiny_inputs"
depth = np.asarray(
    [
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 3.0], [4.0, 5.0]],
    ],
    dtype=np.float32,
)
depth_features = depth[..., None]
seg = np.asarray(
    [
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
    ],
    dtype=np.int64,
)
seg_codes = np.stack([1 - seg, seg], axis=-1).astype(np.float32)
np.savez_compressed(root / "depth_features.npz", features=depth_features)
np.savez_compressed(root / "depth_targets.npz", targets=depth)
np.savez_compressed(root / "seg_codes.npz", codes=seg_codes)
np.savez_compressed(root / "seg_targets.npz", targets=seg)
depth_rows = [
    {"image": "images/nyu_a.jpg", "depth": "depth/a.npy", "split": "val"},
    {"image": "images/nyu_b.jpg", "depth": "depth/b.npy", "split": "val"},
]
seg_rows = [
    {"image": "images/ade_a.jpg", "segmentation": "seg/a.png", "split": "val"},
    {"image": "images/ade_b.jpg", "segmentation": "seg/b.png", "split": "val"},
]
(root / "depth_manifest.jsonl").write_text(
    "\n".join(json.dumps(row) for row in depth_rows) + "\n",
    encoding="utf-8",
)
(root / "seg_manifest.jsonl").write_text(
    "\n".join(json.dumps(row) for row in seg_rows) + "\n",
    encoding="utf-8",
)
dense_ranking = {
    "record_type": "task_feature_ranking",
    "task_id": "ade20k_segmentation",
    "model_id": "ijepa_vit_h14",
    "sae_id": "ijepa_l31_topk32_exp4",
    "ranking_method": "probe_weight",
    "rows": [
        {
            "feature_id": 1,
            "task_rank": 1,
            "ranking_score": 2.0,
            "probe_weight_score": 2.0,
            "validation_contribution_score": 0.0,
            "mean_activation": 0.5,
            "mean_positive_activation": 1.0,
        }
    ],
}
(root / "seg_task_feature_ranking.json").write_text(
    json.dumps(dense_ranking, indent=2) + "\n",
    encoding="utf-8",
)
PY

python -m feature_economy.cli.main probe-native \
  --backend linear-probe \
  --features-npz "${OUTPUT_ROOT}/dense_probe_tiny_inputs/depth_features.npz" \
  --manifest "${OUTPUT_ROOT}/dense_probe_tiny_inputs/depth_manifest.jsonl" \
  --targets-npz "${OUTPUT_ROOT}/dense_probe_tiny_inputs/depth_targets.npz" \
  --task-type dense_depth \
  --task-id nyuv2_depth \
  --model-id dino_v2_base \
  --expected-split val \
  --output-dir "${OUTPUT_ROOT}/probe_native_dense_depth"

python -m feature_economy.cli.main probe-sae \
  --backend linear-probe \
  --codes-npz "${OUTPUT_ROOT}/dense_probe_tiny_inputs/seg_codes.npz" \
  --manifest "${OUTPUT_ROOT}/dense_probe_tiny_inputs/seg_manifest.jsonl" \
  --targets-npz "${OUTPUT_ROOT}/dense_probe_tiny_inputs/seg_targets.npz" \
  --task-type dense_segmentation \
  --task-id ade20k_segmentation \
  --model-id ijepa_vit_h14 \
  --sae-id ijepa_l31_topk32_exp4 \
  --expected-split val \
  --num-classes 2 \
  --output-dir "${OUTPUT_ROOT}/probe_sae_dense_seg"

python -m feature_economy.cli.main extract-features \
  --config-root configs \
  --fixture-manifest tests/fixtures/tiny_imagenet_eval_manifest.jsonl \
  --task-type classification \
  --model-id dino_v2_base \
  --expected-split val \
  --output-dir "${OUTPUT_ROOT}/features_fixture_imagenet"

if python - <<'PY'
import importlib.util
raise SystemExit(
    0
    if importlib.util.find_spec("torch") and importlib.util.find_spec("PIL")
    else 1
)
PY
then
  mkdir -p "${OUTPUT_ROOT}/torchscript_toy_inputs/images"
  python - <<PY
import json
from pathlib import Path
from PIL import Image

root = Path("${OUTPUT_ROOT}") / "torchscript_toy_inputs"
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
    --output "${OUTPUT_ROOT}/checkpoints/tiny_feature_module.pt" \
    --feature-dim 3

  python -m feature_economy.cli.main extract-features \
    --config-root configs \
    --backend torchscript \
    --checkpoint "${OUTPUT_ROOT}/checkpoints/tiny_feature_module.pt" \
    --manifest "${OUTPUT_ROOT}/torchscript_toy_inputs/manifest.jsonl" \
    --task-type classification \
    --model-id ijepa_vit_h14 \
    --expected-split val \
    --batch-size 1 \
    --output-dir "${OUTPUT_ROOT}/features_torchscript_toy"

  python -m feature_economy.cli.main validate-arrays \
    --npz "${OUTPUT_ROOT}/features_torchscript_toy/features.npz" \
    --kind features \
    --task-type classification \
    --manifest "${OUTPUT_ROOT}/torchscript_toy_inputs/manifest.jsonl" \
    --expected-split val \
    --json-output "${OUTPUT_ROOT}/features_torchscript_toy/array_validation.json"
else
  echo "Skipping TorchScript toy extraction: torch or Pillow is not installed."
fi

python -m feature_economy.cli.main extract-sae-codes \
  --config-root configs \
  --features-npz "${OUTPUT_ROOT}/features_fixture_imagenet/features.npz" \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --output-dir "${OUTPUT_ROOT}/codes_fixture_imagenet"

python -m feature_economy.cli.main validate-arrays \
  --npz "${OUTPUT_ROOT}/features_fixture_imagenet/features.npz" \
  --kind features \
  --task-type classification \
  --manifest tests/fixtures/tiny_imagenet_eval_manifest.jsonl \
  --expected-split val \
  --json-output "${OUTPUT_ROOT}/features_fixture_imagenet/array_validation.json"

python -m feature_economy.cli.main validate-arrays \
  --npz "${OUTPUT_ROOT}/codes_fixture_imagenet/codes.npz" \
  --kind codes \
  --task-type classification \
  --manifest tests/fixtures/tiny_imagenet_eval_manifest.jsonl \
  --expected-split val \
  --json-output "${OUTPUT_ROOT}/codes_fixture_imagenet/array_validation.json"

python -m feature_economy.cli.main validate-arrays \
  --npz "${OUTPUT_ROOT}/probe_sae_dense_seg/probe_logits.npz" \
  --kind probe_logits \
  --task-type dense_segmentation \
  --manifest "${OUTPUT_ROOT}/dense_probe_tiny_inputs/seg_manifest.jsonl" \
  --expected-split val \
  --json-output "${OUTPUT_ROOT}/probe_sae_dense_seg/array_validation.json"

python -m feature_economy.cli.main smoke-probe \
  --config-root configs \
  --experiment-id paper_v0_native \
  --output-dir "${OUTPUT_ROOT}/smoke_probe_native"

python -m feature_economy.cli.main smoke-analysis \
  --config-root configs \
  --output-dir "${OUTPUT_ROOT}/smoke_analysis"

python -m feature_economy.cli.main compute-usage \
  --config-root configs \
  --output-dir "${OUTPUT_ROOT}/usage_smoke" \
  --smoke

python -m feature_economy.cli.main rank-features \
  --config-root configs \
  --output-dir "${OUTPUT_ROOT}/rank_smoke" \
  --smoke

mkdir -p "${OUTPUT_ROOT}/subset_usage_tiny_inputs"
python - <<PY
import json
from pathlib import Path
import numpy as np

root = Path("${OUTPUT_ROOT}") / "subset_usage_tiny_inputs"
np.savez_compressed(
    root / "codes.npz",
    codes=np.asarray(
        [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    ),
)
weights = np.asarray(
    [
        [2.0, 0.0],
        [0.0, 0.0],
        [0.0, 2.0],
        [0.1, 0.1],
        [0.0, 0.0],
        [0.1, 0.1],
        [0.0, 0.0],
    ],
    dtype=np.float32,
)
codes = np.load(root / "codes.npz")["codes"]
labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
classes = np.asarray([0, 1], dtype=np.int64)
logits = np.concatenate([codes, np.ones((codes.shape[0], 1), dtype=np.float32)], axis=1) @ weights
np.savez_compressed(
    root / "probe_logits.npz",
    logits=logits,
    weights=weights,
    classes=classes,
    labels=labels,
)
ranking = {
    "record_type": "task_feature_ranking",
    "task_id": "tiny_cls",
    "model_id": "dino_v2_base",
    "sae_id": "dino_l11_topk32_exp4",
    "ranking_method": "probe_weight",
    "rows": [
        {
            "feature_id": 0,
            "task_rank": 1,
            "ranking_score": 2.0,
            "probe_weight_score": 2.0,
            "validation_contribution_score": 0.0,
            "mean_activation": 1.0,
            "mean_positive_activation": 1.0,
        },
        {
            "feature_id": 2,
            "task_rank": 2,
            "ranking_score": 1.0,
            "probe_weight_score": 1.0,
            "validation_contribution_score": 0.0,
            "mean_activation": 0.5,
            "mean_positive_activation": 1.0,
        },
    ],
}
(root / "task_feature_ranking.json").write_text(json.dumps(ranking, indent=2) + "\n")
PY

python -m feature_economy.cli.main compute-subset-usage \
  --config-root configs \
  --codes-npz "${OUTPUT_ROOT}/subset_usage_tiny_inputs/codes.npz" \
  --ranking-json "${OUTPUT_ROOT}/subset_usage_tiny_inputs/task_feature_ranking.json" \
  --top-k 2 \
  --random-seed 0 \
  --high-usage-threshold 3 \
  --output-dir "${OUTPUT_ROOT}/subset_usage_tiny"

python -m feature_economy.cli.main compute-contributions \
  --config-root configs \
  --codes-npz "${OUTPUT_ROOT}/subset_usage_tiny_inputs/codes.npz" \
  --probe-logits-npz "${OUTPUT_ROOT}/subset_usage_tiny_inputs/probe_logits.npz" \
  --task-type classification \
  --task-id tiny_cls \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --output-dir "${OUTPUT_ROOT}/contribution_tiny"

python -m feature_economy.cli.main validate-arrays \
  --npz "${OUTPUT_ROOT}/contribution_tiny/contribution_scores.npz" \
  --kind contribution_scores \
  --json-output "${OUTPUT_ROOT}/contribution_tiny/array_validation.json"

python -m feature_economy.cli.main rank-features \
  --config-root configs \
  --codes-npz "${OUTPUT_ROOT}/subset_usage_tiny_inputs/codes.npz" \
  --probe-logits-npz "${OUTPUT_ROOT}/subset_usage_tiny_inputs/probe_logits.npz" \
  --contribution-npz "${OUTPUT_ROOT}/contribution_tiny/contribution_scores.npz" \
  --ranking-method hybrid \
  --hybrid-alpha 0.5 \
  --task-id tiny_cls \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --top-k 2 \
  --output-dir "${OUTPUT_ROOT}/rank_tiny_hybrid"

python -m feature_economy.cli.main ablate-features \
  --config-root configs \
  --backend linear-probe \
  --codes-npz "${OUTPUT_ROOT}/subset_usage_tiny_inputs/codes.npz" \
  --probe-logits-npz "${OUTPUT_ROOT}/subset_usage_tiny_inputs/probe_logits.npz" \
  --ranking-json "${OUTPUT_ROOT}/subset_usage_tiny_inputs/task_feature_ranking.json" \
  --task-type classification \
  --top-k 2 \
  --random-seed 0 \
  --output-dir "${OUTPUT_ROOT}/ablation_tiny"

python -m feature_economy.cli.main ablate-features \
  --config-root configs \
  --backend linear-probe \
  --codes-npz "${OUTPUT_ROOT}/dense_probe_tiny_inputs/seg_codes.npz" \
  --probe-logits-npz "${OUTPUT_ROOT}/probe_sae_dense_seg/probe_logits.npz" \
  --ranking-json "${OUTPUT_ROOT}/dense_probe_tiny_inputs/seg_task_feature_ranking.json" \
  --task-type dense_segmentation \
  --top-k 1 \
  --random-seed 0 \
  --output-dir "${OUTPUT_ROOT}/ablation_tiny_dense_seg"

python -m feature_economy.cli.main ablate-features \
  --config-root configs \
  --output-dir "${OUTPUT_ROOT}/ablation_smoke" \
  --smoke

python -m feature_economy.cli.main index-artifacts \
  --input-dir "${OUTPUT_ROOT}/fixture_native_clevr" \
  --output-json "${OUTPUT_ROOT}/fixture_index/artifact_index.json" \
  --output-csv "${OUTPUT_ROOT}/fixture_index/artifact_index.csv" \
  --require-valid

python -m feature_economy.cli.main index-artifacts \
  --input-dir "${OUTPUT_ROOT}/probe_native_fixture_imagenet" \
  --output-json "${OUTPUT_ROOT}/probe_native_fixture_index/artifact_index.json" \
  --output-csv "${OUTPUT_ROOT}/probe_native_fixture_index/artifact_index.csv" \
  --require-valid

python -m feature_economy.cli.main index-artifacts \
  --input-dir "${OUTPUT_ROOT}/probe_sae_fixture_clevr" \
  --output-json "${OUTPUT_ROOT}/probe_sae_fixture_index/artifact_index.json" \
  --output-csv "${OUTPUT_ROOT}/probe_sae_fixture_index/artifact_index.csv" \
  --require-valid

python -m feature_economy.cli.main index-artifacts \
  --input-dir "${OUTPUT_ROOT}/probe_native_dense_depth" \
  --output-json "${OUTPUT_ROOT}/probe_native_dense_depth_index/artifact_index.json" \
  --output-csv "${OUTPUT_ROOT}/probe_native_dense_depth_index/artifact_index.csv" \
  --require-valid

python -m feature_economy.cli.main index-artifacts \
  --input-dir "${OUTPUT_ROOT}/probe_sae_dense_seg" \
  --output-json "${OUTPUT_ROOT}/probe_sae_dense_seg_index/artifact_index.json" \
  --output-csv "${OUTPUT_ROOT}/probe_sae_dense_seg_index/artifact_index.csv" \
  --require-valid

python -m feature_economy.cli.main index-artifacts \
  --input-dir "${OUTPUT_ROOT}/features_fixture_imagenet" \
  --output-json "${OUTPUT_ROOT}/features_fixture_index/artifact_index.json" \
  --output-csv "${OUTPUT_ROOT}/features_fixture_index/artifact_index.csv" \
  --require-valid

if [[ -d "${OUTPUT_ROOT}/features_torchscript_toy" ]]; then
  python -m feature_economy.cli.main index-artifacts \
    --input-dir "${OUTPUT_ROOT}/features_torchscript_toy" \
    --output-json "${OUTPUT_ROOT}/features_torchscript_toy_index/artifact_index.json" \
    --output-csv "${OUTPUT_ROOT}/features_torchscript_toy_index/artifact_index.csv" \
    --require-valid
fi

python -m feature_economy.cli.main index-artifacts \
  --input-dir "${OUTPUT_ROOT}/codes_fixture_imagenet" \
  --output-json "${OUTPUT_ROOT}/codes_fixture_index/artifact_index.json" \
  --output-csv "${OUTPUT_ROOT}/codes_fixture_index/artifact_index.csv" \
  --require-valid

python -m feature_economy.cli.main index-artifacts \
  --input-dir "${OUTPUT_ROOT}/subset_usage_tiny" \
  --output-json "${OUTPUT_ROOT}/subset_usage_tiny_index/artifact_index.json" \
  --output-csv "${OUTPUT_ROOT}/subset_usage_tiny_index/artifact_index.csv" \
  --require-valid

python -m feature_economy.cli.main index-artifacts \
  --input-dir "${OUTPUT_ROOT}/contribution_tiny" \
  --output-json "${OUTPUT_ROOT}/contribution_tiny_index/artifact_index.json" \
  --output-csv "${OUTPUT_ROOT}/contribution_tiny_index/artifact_index.csv" \
  --require-valid

python -m feature_economy.cli.main index-artifacts \
  --input-dir "${OUTPUT_ROOT}/rank_tiny_hybrid" \
  --output-json "${OUTPUT_ROOT}/rank_tiny_hybrid_index/artifact_index.json" \
  --output-csv "${OUTPUT_ROOT}/rank_tiny_hybrid_index/artifact_index.csv" \
  --require-valid

python -m feature_economy.cli.main index-artifacts \
  --input-dir "${OUTPUT_ROOT}/ablation_tiny" \
  --output-json "${OUTPUT_ROOT}/ablation_tiny_index/artifact_index.json" \
  --output-csv "${OUTPUT_ROOT}/ablation_tiny_index/artifact_index.csv" \
  --require-valid

python -m feature_economy.cli.main index-artifacts \
  --input-dir "${OUTPUT_ROOT}/ablation_tiny_dense_seg" \
  --output-json "${OUTPUT_ROOT}/ablation_tiny_dense_seg_index/artifact_index.json" \
  --output-csv "${OUTPUT_ROOT}/ablation_tiny_dense_seg_index/artifact_index.csv" \
  --require-valid

python -m feature_economy.cli.main index-artifacts \
  --input-dir "${OUTPUT_ROOT}/smoke_probe_native" \
  --output-json "${OUTPUT_ROOT}/probe_index/artifact_index.json" \
  --output-csv "${OUTPUT_ROOT}/probe_index/artifact_index.csv" \
  --require-valid

python -m feature_economy.cli.main index-artifacts \
  --input-dir "${OUTPUT_ROOT}/smoke_analysis" \
  --output-json "${OUTPUT_ROOT}/analysis_index/artifact_index.json" \
  --output-csv "${OUTPUT_ROOT}/analysis_index/artifact_index.csv" \
  --require-valid

python -m feature_economy.cli.main make-tables \
  --artifact-index "${OUTPUT_ROOT}/probe_index/artifact_index.json" \
  --output-dir "${OUTPUT_ROOT}/probe_tables"

python -m feature_economy.cli.main make-tables \
  --artifact-index "${OUTPUT_ROOT}/analysis_index/artifact_index.json" \
  --output-dir "${OUTPUT_ROOT}/analysis_tables"

if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("matplotlib") else 1)
PY
then
  python -m feature_economy.cli.main make-figures \
    --table-dir "${OUTPUT_ROOT}/analysis_tables" \
    --output-dir "${OUTPUT_ROOT}/analysis_figures" \
    --formats png
else
  echo "Skipping figure smoke: matplotlib is not installed."
fi

echo "Smoke reproduction complete: ${OUTPUT_ROOT}"
