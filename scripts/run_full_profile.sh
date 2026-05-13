#!/usr/bin/env bash
# Role: launch a full paper-style public reproduction profile.
# Status: canonical public launcher; starts with dino_imagenet_l11 and dry-run support.
# Used by: docs/full_reproduction.md and remote validation plans.
# Inputs: DATA_ROOT, SAE_ROOT, ARTIFACT_ROOT, optional DEVICE/BATCH_SIZE/DRY_RUN.
# Outputs: feature, code, probe, analysis, index, table, figure, and log artifacts.
# Safe to move/delete?: keep stable; this is the public full-rerun entrypoint.
# Notes: never writes into the repository checkout unless ARTIFACT_ROOT points there.
set -euo pipefail

PROFILE="${1:-dino_imagenet_l11}"
DRY_RUN="${DRY_RUN:-0}"
CONFIG_ROOT="${CONFIG_ROOT:-configs}"
DATA_ROOT="${DATA_ROOT:-}"
SAE_ROOT="${SAE_ROOT:-}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DTYPE="${DTYPE:-float32}"
RIDGE="${RIDGE:-0.001}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
MAKE_FIGURES="${MAKE_FIGURES:-1}"

usage() {
  cat <<'EOF'
Usage:
  DATA_ROOT=/path/to/manifests \
  SAE_ROOT=/path/to/sae_checkpoints \
  ARTIFACT_ROOT=/path/to/output \
  bash scripts/run_full_profile.sh dino_imagenet_l11

Environment:
  DRY_RUN=1             Print commands without executing them.
  DEVICE=cuda           Feature extraction device.
  BATCH_SIZE=16         Feature extraction batch size.
  DTYPE=float32         Feature dtype: float16 or float32.
  RIDGE=0.001           Closed-form probe ridge value.
  MAX_EXAMPLES=1000     Optional small real-data slice.
  LOCAL_FILES_ONLY=1    Pass --local-files-only to HuggingFace extraction.
  DINO_HF_NAME_OR_PATH  Optional local DINO checkpoint directory or HF model id.
  MAKE_FIGURES=0        Skip overview figure export.

Supported profiles:
  dino_imagenet_l11
EOF
}

if [[ "${PROFILE}" == "-h" || "${PROFILE}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$PROFILE" != "dino_imagenet_l11" ]]; then
  echo "Unsupported profile: $PROFILE" >&2
  usage >&2
  exit 2
fi

if [[ "$DRY_RUN" != "1" ]]; then
  for name in DATA_ROOT SAE_ROOT ARTIFACT_ROOT; do
    if [[ -z "${!name:-}" ]]; then
      echo "Missing required environment variable: $name" >&2
      usage >&2
      exit 2
    fi
  done
fi

DATA_ROOT="${DATA_ROOT:-/path/to/manifests}"
SAE_ROOT="${SAE_ROOT:-/path/to/sae_checkpoints}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/path/to/output_artifacts}"

MODEL_ID="dino_v2_base"
SAE_ID="dino_l11_topk32_exp4"
TASK_ID="imagenet_1k"
TASK_TYPE="classification"
SPLIT="val"
MANIFEST="$DATA_ROOT/imagenet/val_manifest.jsonl"
SAE_CHECKPOINT="$SAE_ROOT/$SAE_ID/final_sae.pt"
LIGHTWEIGHT_SAE="$ARTIFACT_ROOT/checkpoints/${SAE_ID}_lightweight.npz"
FEATURE_DIR="$ARTIFACT_ROOT/features/$MODEL_ID/imagenet_val_l11"
CODES_DIR="$ARTIFACT_ROOT/codes/$SAE_ID/imagenet_val"
NATIVE_PROBE_DIR="$ARTIFACT_ROOT/probes/native/$MODEL_ID/imagenet_val"
SAE_PROBE_DIR="$ARTIFACT_ROOT/probes/sae/$SAE_ID/imagenet_val"
AVAILABILITY_DIR="$ARTIFACT_ROOT/analysis/availability/$SAE_ID/imagenet_val"
RANKING_DIR="$ARTIFACT_ROOT/analysis/ranking/$SAE_ID/imagenet_val"
CONTRIBUTION_DIR="$ARTIFACT_ROOT/analysis/contribution/$SAE_ID/imagenet_val"
RANKING_HYBRID_DIR="$ARTIFACT_ROOT/analysis/ranking_hybrid/$SAE_ID/imagenet_val"
SUBSET_USAGE_DIR="$ARTIFACT_ROOT/analysis/subset_usage/$SAE_ID/imagenet_val"
ABLATION_DIR="$ARTIFACT_ROOT/analysis/ablation/$SAE_ID/imagenet_val"
INDEX_DIR="$ARTIFACT_ROOT/index/analysis"
TABLE_DIR="$ARTIFACT_ROOT/tables/analysis"
FIGURE_DIR="$ARTIFACT_ROOT/figures/analysis"
RUN_PLAN_JSON="$ARTIFACT_ROOT/run_plan/reproduction_run_plan.json"
RUN_PLAN_CSV="$ARTIFACT_ROOT/run_plan/reproduction_run_plan.csv"

run() {
  echo
  echo "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

run_shell() {
  echo
  echo "+ $*"
  if [[ "$DRY_RUN" != "1" ]]; then
    bash -lc "$*"
  fi
}

optional_args=()
if [[ -n "$MAX_EXAMPLES" ]]; then
  optional_args+=(--max-examples "$MAX_EXAMPLES")
fi
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  optional_args+=(--local-files-only)
fi
if [[ -n "${DINO_HF_NAME_OR_PATH:-}" ]]; then
  optional_args+=(--hf-name-or-path "$DINO_HF_NAME_OR_PATH")
fi

run mkdir -p \
  "$ARTIFACT_ROOT/run_plan" \
  "$ARTIFACT_ROOT/checkpoints" \
  "$FEATURE_DIR" \
  "$CODES_DIR" \
  "$NATIVE_PROBE_DIR" \
  "$SAE_PROBE_DIR" \
  "$AVAILABILITY_DIR" \
  "$RANKING_DIR" \
  "$CONTRIBUTION_DIR" \
  "$RANKING_HYBRID_DIR" \
  "$SUBSET_USAGE_DIR" \
  "$ABLATION_DIR" \
  "$INDEX_DIR" \
  "$TABLE_DIR" \
  "$FIGURE_DIR" \
  "$ARTIFACT_ROOT/logs"

run python -m feature_economy.cli.main check-runtime \
  --profile experiments \
  --require \
  --json-output "$ARTIFACT_ROOT/logs/runtime_check.json"

run python -m feature_economy.cli.main check-configs \
  --config-root "$CONFIG_ROOT"

run python -m feature_economy.cli.main plan-runs \
  --config-root "$CONFIG_ROOT" \
  --output-json "$RUN_PLAN_JSON" \
  --output-csv "$RUN_PLAN_CSV"

run python -m feature_economy.cli.main check-models \
  --config-root "$CONFIG_ROOT"

run python -m feature_economy.cli.main check-manifest \
  --manifest "$MANIFEST" \
  --task-type "$TASK_TYPE" \
  --expected-split "$SPLIT"

if [[ ${#optional_args[@]} -gt 0 ]]; then
  run python -m feature_economy.cli.main extract-features \
    --config-root "$CONFIG_ROOT" \
    --backend huggingface \
    --manifest "$MANIFEST" \
    --task-type "$TASK_TYPE" \
    --model-id "$MODEL_ID" \
    --expected-split "$SPLIT" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --output-dir "$FEATURE_DIR" \
    "${optional_args[@]}"
else
  run python -m feature_economy.cli.main extract-features \
    --config-root "$CONFIG_ROOT" \
    --backend huggingface \
    --manifest "$MANIFEST" \
    --task-type "$TASK_TYPE" \
    --model-id "$MODEL_ID" \
    --expected-split "$SPLIT" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --output-dir "$FEATURE_DIR"
fi

run python -m feature_economy.cli.main validate-arrays \
  --npz "$FEATURE_DIR/features.npz" \
  --kind features \
  --task-type "$TASK_TYPE" \
  --manifest "$MANIFEST" \
  --expected-split "$SPLIT"

run python -m feature_economy.cli.main probe-native \
  --backend linear-probe \
  --features-npz "$FEATURE_DIR/features.npz" \
  --manifest "$MANIFEST" \
  --task-type "$TASK_TYPE" \
  --task-id "$TASK_ID" \
  --model-id "$MODEL_ID" \
  --expected-split "$SPLIT" \
  --ridge "$RIDGE" \
  --output-dir "$NATIVE_PROBE_DIR"

run python -m feature_economy.cli.main convert-sae-checkpoint \
  --input-checkpoint "$SAE_CHECKPOINT" \
  --output-checkpoint "$LIGHTWEIGHT_SAE"

run python -m feature_economy.cli.main extract-sae-codes \
  --config-root "$CONFIG_ROOT" \
  --backend linear-topk \
  --features-npz "$FEATURE_DIR/features.npz" \
  --model-id "$MODEL_ID" \
  --sae-id "$SAE_ID" \
  --checkpoint "$LIGHTWEIGHT_SAE" \
  --output-dir "$CODES_DIR"

run python -m feature_economy.cli.main validate-arrays \
  --npz "$CODES_DIR/codes.npz" \
  --kind codes \
  --task-type "$TASK_TYPE" \
  --manifest "$MANIFEST" \
  --expected-split "$SPLIT"

run python -m feature_economy.cli.main probe-sae \
  --backend linear-probe \
  --codes-npz "$CODES_DIR/codes.npz" \
  --manifest "$MANIFEST" \
  --task-type "$TASK_TYPE" \
  --task-id "$TASK_ID" \
  --model-id "$MODEL_ID" \
  --sae-id "$SAE_ID" \
  --expected-split "$SPLIT" \
  --ridge "$RIDGE" \
  --output-dir "$SAE_PROBE_DIR"

run python -m feature_economy.cli.main compute-usage \
  --config-root "$CONFIG_ROOT" \
  --codes-npz "$CODES_DIR/codes.npz" \
  --model-id "$MODEL_ID" \
  --sae-id "$SAE_ID" \
  --dataset-id "${TASK_ID}_${SPLIT}" \
  --split "$SPLIT" \
  --output-dir "$AVAILABILITY_DIR"

run python -m feature_economy.cli.main rank-features \
  --config-root "$CONFIG_ROOT" \
  --codes-npz "$CODES_DIR/codes.npz" \
  --probe-logits-npz "$SAE_PROBE_DIR/probe_logits.npz" \
  --task-id "$TASK_ID" \
  --model-id "$MODEL_ID" \
  --sae-id "$SAE_ID" \
  --top-k 100 \
  --output-dir "$RANKING_DIR"

run python -m feature_economy.cli.main compute-contributions \
  --config-root "$CONFIG_ROOT" \
  --codes-npz "$CODES_DIR/codes.npz" \
  --probe-logits-npz "$SAE_PROBE_DIR/probe_logits.npz" \
  --task-type "$TASK_TYPE" \
  --task-id "$TASK_ID" \
  --model-id "$MODEL_ID" \
  --sae-id "$SAE_ID" \
  --output-dir "$CONTRIBUTION_DIR"

run python -m feature_economy.cli.main rank-features \
  --config-root "$CONFIG_ROOT" \
  --codes-npz "$CODES_DIR/codes.npz" \
  --probe-logits-npz "$SAE_PROBE_DIR/probe_logits.npz" \
  --contribution-npz "$CONTRIBUTION_DIR/contribution_scores.npz" \
  --ranking-method hybrid \
  --hybrid-alpha 0.5 \
  --task-id "$TASK_ID" \
  --model-id "$MODEL_ID" \
  --sae-id "$SAE_ID" \
  --top-k 100 \
  --output-dir "$RANKING_HYBRID_DIR"

run python -m feature_economy.cli.main compute-subset-usage \
  --config-root "$CONFIG_ROOT" \
  --codes-npz "$CODES_DIR/codes.npz" \
  --ranking-json "$RANKING_HYBRID_DIR/task_feature_ranking.json" \
  --top-k 100 \
  --random-seed 0 \
  --output-dir "$SUBSET_USAGE_DIR"

run python -m feature_economy.cli.main ablate-features \
  --config-root "$CONFIG_ROOT" \
  --backend linear-probe \
  --codes-npz "$CODES_DIR/codes.npz" \
  --probe-logits-npz "$SAE_PROBE_DIR/probe_logits.npz" \
  --ranking-json "$RANKING_HYBRID_DIR/task_feature_ranking.json" \
  --task-type "$TASK_TYPE" \
  --top-k 100 \
  --random-seed 0 \
  --output-dir "$ABLATION_DIR"

run python -m feature_economy.cli.main index-artifacts \
  --input-dir "$ARTIFACT_ROOT" \
  --output-json "$INDEX_DIR/artifact_index.json" \
  --output-csv "$INDEX_DIR/artifact_index.csv" \
  --require-valid

run python -m feature_economy.cli.main make-tables \
  --artifact-index "$INDEX_DIR/artifact_index.json" \
  --output-dir "$TABLE_DIR"

if [[ "$MAKE_FIGURES" == "1" ]]; then
  run python -m feature_economy.cli.main make-figures \
    --table-dir "$TABLE_DIR" \
    --output-dir "$FIGURE_DIR" \
    --formats png,pdf
fi

run_shell "cat > '$ARTIFACT_ROOT/run_profile_summary.txt' <<EOF
profile=$PROFILE
model_id=$MODEL_ID
sae_id=$SAE_ID
task_id=$TASK_ID
artifact_root=$ARTIFACT_ROOT
manifest=$MANIFEST
sae_checkpoint=$SAE_CHECKPOINT
EOF"

echo
echo "Full profile completed: $PROFILE"
echo "Artifact root: $ARTIFACT_ROOT"
