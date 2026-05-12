#!/usr/bin/env bash
# Role: build a tiny complete artifact bundle for one DINO/ImageNet SAE slice.
# Status: public example runner
# Used by: public v1 scope documentation and release sanity checks
# Inputs: public_repro configs and tests/fixtures/tiny_imagenet_eval_manifest.jsonl
# Outputs: a self-contained tiny artifact bundle under OUTPUT_ROOT
# Safe to move/delete?: keep; this demonstrates plan-runs/check-bundle completion semantics.
# Notes: This is not a scientific result. It uses fixture features/codes and tiny labels.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/feature_economy_tiny_bundle}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

rm -rf "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/run_plan"

python - <<PY
import json
from pathlib import Path

root = Path("${OUTPUT_ROOT}")
rows = [
    ("feature_extraction", "extract-features", "", "\${ARTIFACT_ROOT}/features/dino_v2_base/imagenet_1k/l11"),
    ("sae_code_extraction", "extract-sae-codes", "dino_l11_topk32_exp4", "\${ARTIFACT_ROOT}/codes/dino_l11_topk32_exp4/imagenet_1k/val"),
    ("native_probe", "probe-native", "", "\${ARTIFACT_ROOT}/probes/native/dino_v2_base/imagenet_1k/val"),
    ("sae_probe", "probe-sae", "dino_l11_topk32_exp4", "\${ARTIFACT_ROOT}/probes/sae/dino_l11_topk32_exp4/imagenet_1k/val"),
    ("availability", "compute-usage", "dino_l11_topk32_exp4", "\${ARTIFACT_ROOT}/analysis/availability/dino_l11_topk32_exp4/imagenet_1k_val"),
    ("feature_ranking", "rank-features", "dino_l11_topk32_exp4", "\${ARTIFACT_ROOT}/analysis/ranking/dino_l11_topk32_exp4/imagenet_1k/val"),
    ("contribution_scores", "compute-contributions", "dino_l11_topk32_exp4", "\${ARTIFACT_ROOT}/analysis/contribution/dino_l11_topk32_exp4/imagenet_1k/val"),
    ("subset_usage", "compute-subset-usage", "dino_l11_topk32_exp4", "\${ARTIFACT_ROOT}/analysis/subset_usage/dino_l11_topk32_exp4/imagenet_1k/val"),
    ("feature_ablation", "ablate-features", "dino_l11_topk32_exp4", "\${ARTIFACT_ROOT}/analysis/ablation/dino_l11_topk32_exp4/imagenet_1k/val"),
]
plan = {
    "record_type": "reproduction_run_plan",
    "config_root": "configs",
    "num_rows": len(rows),
    "stages": sorted({stage for stage, _, _, _ in rows}),
    "rows": [
        {
            "stage": stage,
            "experiment_id": "tiny_dino_imagenet_bundle",
            "task_id": "imagenet_1k",
            "task_type": "classification",
            "model_id": "dino_v2_base",
            "sae_id": sae_id,
            "command": command,
            "artifact_dir": artifact_dir,
        }
        for stage, command, sae_id, artifact_dir in rows
    ],
}
(root / "run_plan" / "tiny_dino_imagenet_run_plan.json").write_text(
    json.dumps(plan, indent=2) + "\n",
    encoding="utf-8",
)
PY

python -m feature_economy.cli.main check-bundle \
  --run-plan-json "${OUTPUT_ROOT}/run_plan/tiny_dino_imagenet_run_plan.json" \
  --artifact-root "${OUTPUT_ROOT}" \
  --output-json "${OUTPUT_ROOT}/run_plan/artifact_bundle_check_initial.json"

FEATURE_DIR="${OUTPUT_ROOT}/features/dino_v2_base/imagenet_1k/l11"
CODE_DIR="${OUTPUT_ROOT}/codes/dino_l11_topk32_exp4/imagenet_1k/val"
NATIVE_PROBE_DIR="${OUTPUT_ROOT}/probes/native/dino_v2_base/imagenet_1k/val"
SAE_PROBE_DIR="${OUTPUT_ROOT}/probes/sae/dino_l11_topk32_exp4/imagenet_1k/val"
AVAILABILITY_DIR="${OUTPUT_ROOT}/analysis/availability/dino_l11_topk32_exp4/imagenet_1k_val"
RANKING_DIR="${OUTPUT_ROOT}/analysis/ranking/dino_l11_topk32_exp4/imagenet_1k/val"
CONTRIBUTION_DIR="${OUTPUT_ROOT}/analysis/contribution/dino_l11_topk32_exp4/imagenet_1k/val"
SUBSET_USAGE_DIR="${OUTPUT_ROOT}/analysis/subset_usage/dino_l11_topk32_exp4/imagenet_1k/val"
ABLATION_DIR="${OUTPUT_ROOT}/analysis/ablation/dino_l11_topk32_exp4/imagenet_1k/val"
MANIFEST="tests/fixtures/tiny_imagenet_eval_manifest.jsonl"

python -m feature_economy.cli.main extract-features \
  --config-root configs \
  --backend fixture \
  --fixture-manifest "${MANIFEST}" \
  --task-type classification \
  --model-id dino_v2_base \
  --expected-split val \
  --output-dir "${FEATURE_DIR}"

python -m feature_economy.cli.main validate-arrays \
  --npz "${FEATURE_DIR}/features.npz" \
  --kind features \
  --task-type classification \
  --manifest "${MANIFEST}" \
  --expected-split val \
  --json-output "${FEATURE_DIR}/array_validation.json"

python -m feature_economy.cli.main extract-sae-codes \
  --config-root configs \
  --backend fixture \
  --features-npz "${FEATURE_DIR}/features.npz" \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --output-dir "${CODE_DIR}"

python -m feature_economy.cli.main validate-arrays \
  --npz "${CODE_DIR}/codes.npz" \
  --kind codes \
  --task-type classification \
  --manifest "${MANIFEST}" \
  --expected-split val \
  --json-output "${CODE_DIR}/array_validation.json"

python -m feature_economy.cli.main probe-native \
  --backend linear-probe \
  --features-npz "${FEATURE_DIR}/features.npz" \
  --manifest "${MANIFEST}" \
  --task-type classification \
  --task-id imagenet_1k \
  --model-id dino_v2_base \
  --expected-split val \
  --output-dir "${NATIVE_PROBE_DIR}"

python -m feature_economy.cli.main probe-sae \
  --backend linear-probe \
  --codes-npz "${CODE_DIR}/codes.npz" \
  --manifest "${MANIFEST}" \
  --task-type classification \
  --task-id imagenet_1k \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --expected-split val \
  --output-dir "${SAE_PROBE_DIR}"

python -m feature_economy.cli.main compute-usage \
  --config-root configs \
  --codes-npz "${CODE_DIR}/codes.npz" \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --dataset-id imagenet_1k \
  --split val \
  --output-dir "${AVAILABILITY_DIR}"

python -m feature_economy.cli.main rank-features \
  --config-root configs \
  --codes-npz "${CODE_DIR}/codes.npz" \
  --probe-logits-npz "${SAE_PROBE_DIR}/probe_logits.npz" \
  --task-id imagenet_1k \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --top-k 2 \
  --output-dir "${RANKING_DIR}"

python -m feature_economy.cli.main compute-contributions \
  --config-root configs \
  --codes-npz "${CODE_DIR}/codes.npz" \
  --probe-logits-npz "${SAE_PROBE_DIR}/probe_logits.npz" \
  --task-type classification \
  --task-id imagenet_1k \
  --model-id dino_v2_base \
  --sae-id dino_l11_topk32_exp4 \
  --output-dir "${CONTRIBUTION_DIR}"

python -m feature_economy.cli.main validate-arrays \
  --npz "${CONTRIBUTION_DIR}/contribution_scores.npz" \
  --kind contribution_scores \
  --json-output "${CONTRIBUTION_DIR}/array_validation.json"

python -m feature_economy.cli.main compute-subset-usage \
  --config-root configs \
  --codes-npz "${CODE_DIR}/codes.npz" \
  --ranking-json "${RANKING_DIR}/task_feature_ranking.json" \
  --top-k 2 \
  --random-seed 0 \
  --output-dir "${SUBSET_USAGE_DIR}"

python -m feature_economy.cli.main ablate-features \
  --config-root configs \
  --backend linear-probe \
  --codes-npz "${CODE_DIR}/codes.npz" \
  --probe-logits-npz "${SAE_PROBE_DIR}/probe_logits.npz" \
  --ranking-json "${RANKING_DIR}/task_feature_ranking.json" \
  --task-type classification \
  --top-k 2 \
  --random-seed 0 \
  --output-dir "${ABLATION_DIR}"

python -m feature_economy.cli.main check-bundle \
  --run-plan-json "${OUTPUT_ROOT}/run_plan/tiny_dino_imagenet_run_plan.json" \
  --artifact-root "${OUTPUT_ROOT}" \
  --output-json "${OUTPUT_ROOT}/run_plan/artifact_bundle_check_final.json" \
  --require-complete

python -m feature_economy.cli.main index-artifacts \
  --input-dir "${OUTPUT_ROOT}" \
  --output-json "${OUTPUT_ROOT}/index/artifact_index.json" \
  --output-csv "${OUTPUT_ROOT}/index/artifact_index.csv" \
  --require-valid

python -m feature_economy.cli.main make-tables \
  --artifact-index "${OUTPUT_ROOT}/index/artifact_index.json" \
  --output-dir "${OUTPUT_ROOT}/tables"

echo "Tiny artifact bundle complete: ${OUTPUT_ROOT}"
