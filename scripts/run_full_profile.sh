#!/usr/bin/env bash
# Role: launch a full paper-style public reproduction profile.
# Status: canonical public launcher; supports ImageNet and NYUv2 full-profile slices.
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
CONTRIBUTION_SCORING_METHOD="${CONTRIBUTION_SCORING_METHOD:-auto}"
IJEPA_TORCHSCRIPT_CHECKPOINT="${IJEPA_TORCHSCRIPT_CHECKPOINT:-}"
TORCHSCRIPT_OUTPUT_KEY="${TORCHSCRIPT_OUTPUT_KEY:-}"
TORCHSCRIPT_OUTPUT_INDEX="${TORCHSCRIPT_OUTPUT_INDEX:-0}"

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
  IJEPA_HF_NAME_OR_PATH
                         Optional local I-JEPA checkpoint directory or HF model id.
                         If set, ijepa_imagenet_l31 uses HuggingFace extraction.
  IJEPA_TORCHSCRIPT_CHECKPOINT
                         TorchScript feature module for ijepa_imagenet_l31 when
                         IJEPA_HF_NAME_OR_PATH is not set.
  TORCHSCRIPT_OUTPUT_KEY Optional dict key for TorchScript outputs.
  TORCHSCRIPT_OUTPUT_INDEX=0
                         Tuple/list output index for TorchScript outputs.
  CONTRIBUTION_SCORING_METHOD=auto
                         Feature contribution scoring method. `auto` uses
                         true_class_logit_drop for ImageNet/CLEVR-style
                         classification and weight_activation for dense tasks.
                         Use exact_metric_drop for small exact audit runs.
  MAKE_FIGURES=0        Skip overview figure export.

Supported profiles:
  dino_imagenet_l11
  ijepa_imagenet_l31
  dino_nyuv2_l11
  ijepa_nyuv2_l31
EOF
}

if [[ "${PROFILE}" == "-h" || "${PROFILE}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$PROFILE" != "dino_imagenet_l11" \
  && "$PROFILE" != "ijepa_imagenet_l31" \
  && "$PROFILE" != "dino_nyuv2_l11" \
  && "$PROFILE" != "ijepa_nyuv2_l31" ]]; then
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

SPLIT="val"
case "$PROFILE" in
  dino_imagenet_l11|ijepa_imagenet_l31)
    TASK_ID="imagenet_1k"
    TASK_TYPE="classification"
    TASK_SLUG="imagenet_val"
    MANIFEST="$DATA_ROOT/imagenet/val_manifest.jsonl"
    ;;
  dino_nyuv2_l11|ijepa_nyuv2_l31)
    TASK_ID="nyuv2_depth"
    TASK_TYPE="dense_depth"
    TASK_SLUG="nyuv2_val"
    MANIFEST="$DATA_ROOT/nyuv2/val_manifest.jsonl"
    ;;
esac

if [[ "$PROFILE" == "dino_imagenet_l11" || "$PROFILE" == "dino_nyuv2_l11" ]]; then
  MODEL_ID="dino_v2_base"
  SAE_ID="dino_l11_topk32_exp4"
  LAYER="11"
  FEATURE_BACKEND="huggingface"
  FEATURE_DIR="$ARTIFACT_ROOT/features/$MODEL_ID/${TASK_SLUG}_l11"
elif [[ "$PROFILE" == "ijepa_imagenet_l31" || "$PROFILE" == "ijepa_nyuv2_l31" ]]; then
  MODEL_ID="ijepa_vit_h14"
  SAE_ID="ijepa_l31_topk32_exp4"
  LAYER="31"
  FEATURE_DIR="$ARTIFACT_ROOT/features/$MODEL_ID/${TASK_SLUG}_l31"
  if [[ -n "${IJEPA_HF_NAME_OR_PATH:-}" ]]; then
    FEATURE_BACKEND="huggingface"
  else
    FEATURE_BACKEND="torchscript"
    if [[ "$DRY_RUN" != "1" && -z "$IJEPA_TORCHSCRIPT_CHECKPOINT" ]]; then
      echo "Missing required environment variable for $PROFILE: IJEPA_TORCHSCRIPT_CHECKPOINT or IJEPA_HF_NAME_OR_PATH" >&2
      usage >&2
      exit 2
    fi
    IJEPA_TORCHSCRIPT_CHECKPOINT="${IJEPA_TORCHSCRIPT_CHECKPOINT:-/path/to/ijepa_l31_feature_module.pt}"
  fi
fi
if [[ "$CONTRIBUTION_SCORING_METHOD" == "auto" ]]; then
  if [[ "$TASK_TYPE" == "classification" || "$TASK_TYPE" == "count_classification" ]]; then
    CONTRIBUTION_SCORING_METHOD="true_class_logit_drop"
  else
    CONTRIBUTION_SCORING_METHOD="weight_activation"
  fi
fi
SAE_CHECKPOINT="$SAE_ROOT/$SAE_ID/final_sae.pt"
LIGHTWEIGHT_SAE="$ARTIFACT_ROOT/checkpoints/${SAE_ID}_lightweight.npz"
CODES_DIR="$ARTIFACT_ROOT/codes/$SAE_ID/$TASK_SLUG"
TARGET_DIR="$ARTIFACT_ROOT/targets/$TASK_ID/$SPLIT"
NATIVE_PROBE_DIR="$ARTIFACT_ROOT/probes/native/$MODEL_ID/$TASK_SLUG"
SAE_PROBE_DIR="$ARTIFACT_ROOT/probes/sae/$SAE_ID/$TASK_SLUG"
AVAILABILITY_DIR="$ARTIFACT_ROOT/analysis/availability/$SAE_ID/$TASK_SLUG"
RANKING_DIR="$ARTIFACT_ROOT/analysis/ranking/$SAE_ID/$TASK_SLUG"
CONTRIBUTION_DIR="$ARTIFACT_ROOT/analysis/contribution/$SAE_ID/$TASK_SLUG"
RANKING_HYBRID_DIR="$ARTIFACT_ROOT/analysis/ranking_hybrid/$SAE_ID/$TASK_SLUG"
SUBSET_USAGE_DIR="$ARTIFACT_ROOT/analysis/subset_usage/$SAE_ID/$TASK_SLUG"
ABLATION_DIR="$ARTIFACT_ROOT/analysis/ablation/$SAE_ID/$TASK_SLUG"
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
torchscript_optional_args=()
if [[ -n "$MAX_EXAMPLES" ]]; then
  optional_args+=(--max-examples "$MAX_EXAMPLES")
  torchscript_optional_args+=(--max-examples "$MAX_EXAMPLES")
fi
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  optional_args+=(--local-files-only)
fi
if [[ "$MODEL_ID" == "dino_v2_base" && -n "${DINO_HF_NAME_OR_PATH:-}" ]]; then
  optional_args+=(--hf-name-or-path "$DINO_HF_NAME_OR_PATH")
fi
if [[ "$MODEL_ID" == "ijepa_vit_h14" && -n "${IJEPA_HF_NAME_OR_PATH:-}" ]]; then
  optional_args+=(--hf-name-or-path "$IJEPA_HF_NAME_OR_PATH")
fi
if [[ -n "$TORCHSCRIPT_OUTPUT_KEY" ]]; then
  torchscript_optional_args+=(--output-key "$TORCHSCRIPT_OUTPUT_KEY")
fi
torchscript_optional_args+=(--output-index "$TORCHSCRIPT_OUTPUT_INDEX")

run mkdir -p \
  "$ARTIFACT_ROOT/run_plan" \
  "$ARTIFACT_ROOT/checkpoints" \
  "$FEATURE_DIR" \
  "$CODES_DIR" \
  "$TARGET_DIR" \
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

if [[ "$FEATURE_BACKEND" == "huggingface" && ${#optional_args[@]} -gt 0 ]]; then
  run python -m feature_economy.cli.main extract-features \
    --config-root "$CONFIG_ROOT" \
    --backend huggingface \
    --manifest "$MANIFEST" \
    --task-type "$TASK_TYPE" \
    --model-id "$MODEL_ID" \
    --layer "$LAYER" \
    --expected-split "$SPLIT" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --output-dir "$FEATURE_DIR" \
    "${optional_args[@]}"
elif [[ "$FEATURE_BACKEND" == "huggingface" ]]; then
  run python -m feature_economy.cli.main extract-features \
    --config-root "$CONFIG_ROOT" \
    --backend huggingface \
    --manifest "$MANIFEST" \
    --task-type "$TASK_TYPE" \
    --model-id "$MODEL_ID" \
    --layer "$LAYER" \
    --expected-split "$SPLIT" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --output-dir "$FEATURE_DIR"
elif [[ "$FEATURE_BACKEND" == "torchscript" ]]; then
  run python -m feature_economy.cli.main extract-features \
    --config-root "$CONFIG_ROOT" \
    --backend torchscript \
    --checkpoint "$IJEPA_TORCHSCRIPT_CHECKPOINT" \
    --manifest "$MANIFEST" \
    --task-type "$TASK_TYPE" \
    --model-id "$MODEL_ID" \
    --layer "$LAYER" \
    --expected-split "$SPLIT" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --output-dir "$FEATURE_DIR" \
    "${torchscript_optional_args[@]}"
fi

run python -m feature_economy.cli.main validate-arrays \
  --npz "$FEATURE_DIR/features.npz" \
  --kind features \
  --task-type "$TASK_TYPE" \
  --manifest "$MANIFEST" \
  --expected-split "$SPLIT"

probe_target_args=()
if [[ "$TASK_TYPE" == "dense_depth" || "$TASK_TYPE" == "dense_segmentation" ]]; then
  run python -m feature_economy.cli.main export-targets \
    --manifest "$MANIFEST" \
    --task-type "$TASK_TYPE" \
    --expected-split "$SPLIT" \
    --features-npz "$FEATURE_DIR/features.npz" \
    --output-dir "$TARGET_DIR"

  run python -m feature_economy.cli.main validate-arrays \
    --npz "$TARGET_DIR/targets.npz" \
    --kind targets \
    --task-type "$TASK_TYPE" \
    --manifest "$MANIFEST" \
    --expected-split "$SPLIT"

  probe_target_args=(--targets-npz "$TARGET_DIR/targets.npz")
fi

run python -m feature_economy.cli.main probe-native \
  --backend linear-probe \
  --features-npz "$FEATURE_DIR/features.npz" \
  --manifest "$MANIFEST" \
  --task-type "$TASK_TYPE" \
  --task-id "$TASK_ID" \
  --model-id "$MODEL_ID" \
  --expected-split "$SPLIT" \
  --ridge "$RIDGE" \
  "${probe_target_args[@]}" \
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
  "${probe_target_args[@]}" \
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
  --scoring-method "$CONTRIBUTION_SCORING_METHOD" \
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
task_type=$TASK_TYPE
feature_backend=$FEATURE_BACKEND
contribution_scoring_method=$CONTRIBUTION_SCORING_METHOD
artifact_root=$ARTIFACT_ROOT
manifest=$MANIFEST
sae_checkpoint=$SAE_CHECKPOINT
layer=$LAYER
EOF"

echo
echo "Full profile completed: $PROFILE"
echo "Artifact root: $ARTIFACT_ROOT"
