#!/usr/bin/env bash
# Role: launch a full paper-style public reproduction profile.
# Status: canonical public launcher for ImageNet, NYUv2, ADE20K, and CLEVR/Count slices.
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
PROBE_BACKEND="${PROBE_BACKEND:-paper-scale-torch}"
PROBE_EPOCHS="${PROBE_EPOCHS:-20}"
PROBE_BATCH_SIZE="${PROBE_BATCH_SIZE:-512}"
PROBE_LR="${PROBE_LR:-0.001}"
PROBE_WEIGHT_DECAY="${PROBE_WEIGHT_DECAY:-0.0001}"
DECODER_HIDDEN_CHANNELS="${DECODER_HIDDEN_CHANNELS:-256}"
TARGET_KEY="${TARGET_KEY:-targets}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
MAKE_FIGURES="${MAKE_FIGURES:-1}"
CONTRIBUTION_SCORING_METHOD="${CONTRIBUTION_SCORING_METHOD:-auto}"
IJEPA_TORCHSCRIPT_CHECKPOINT="${IJEPA_TORCHSCRIPT_CHECKPOINT:-}"
TORCHSCRIPT_OUTPUT_KEY="${TORCHSCRIPT_OUTPUT_KEY:-}"
TORCHSCRIPT_OUTPUT_INDEX="${TORCHSCRIPT_OUTPUT_INDEX:-0}"
RANKING_METHODS="${RANKING_METHODS:-probe_weight validation_contribution hybrid}"
HYBRID_ALPHA="${HYBRID_ALPHA:-0.5}"

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
  PROBE_BACKEND=paper-scale-torch
                         Full rerun default. Use linear-probe for lightweight diagnostics.
  RIDGE=0.001           Closed-form probe ridge value when PROBE_BACKEND=linear-probe.
  PROBE_EPOCHS=20       Paper-scale torch probe epochs.
  PROBE_BATCH_SIZE=512  Paper-scale torch probe batch size.
  PROBE_LR=0.001        Paper-scale torch probe learning rate.
  PROBE_WEIGHT_DECAY=0.0001
  DECODER_HIDDEN_CHANNELS=256
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
  RANKING_METHODS="probe_weight validation_contribution hybrid"
                         Alternative ranking-control sweep. Each method writes
                         ranking, subset-usage, and ablation artifacts under
                         analysis/ranking_controls/.
  HYBRID_ALPHA=0.5       Probe-weight mixture weight for hybrid ranking.
  MAKE_FIGURES=0        Skip overview figure export.

Supported profiles:
  dino_imagenet_l11
  ijepa_imagenet_l31
  dino_nyuv2_l11
  ijepa_nyuv2_l31
  dino_ade20k_l11
  ijepa_ade20k_l31
  dino_clevr_count_l11
  ijepa_clevr_count_l31
EOF
}

if [[ "${PROFILE}" == "-h" || "${PROFILE}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$PROFILE" != "dino_imagenet_l11" \
  && "$PROFILE" != "ijepa_imagenet_l31" \
  && "$PROFILE" != "dino_nyuv2_l11" \
  && "$PROFILE" != "ijepa_nyuv2_l31" \
  && "$PROFILE" != "dino_ade20k_l11" \
  && "$PROFILE" != "ijepa_ade20k_l31" \
  && "$PROFILE" != "dino_clevr_count_l11" \
  && "$PROFILE" != "ijepa_clevr_count_l31" ]]; then
  echo "Unsupported profile: $PROFILE" >&2
  usage >&2
  exit 2
fi
if [[ "$PROBE_BACKEND" != "paper-scale-torch" && "$PROBE_BACKEND" != "linear-probe" ]]; then
  echo "Unsupported PROBE_BACKEND: $PROBE_BACKEND" >&2
  echo "Use PROBE_BACKEND=paper-scale-torch or PROBE_BACKEND=linear-probe." >&2
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

TRAIN_SPLIT="train"
VAL_SPLIT="val"
SPLIT="$VAL_SPLIT"
NUM_CLASSES="${NUM_CLASSES:-}"
IGNORE_INDEX="${IGNORE_INDEX:-}"
SEGMENTATION_IGNORE_VALUE="${SEGMENTATION_IGNORE_VALUE:-}"
SEGMENTATION_LABEL_OFFSET="${SEGMENTATION_LABEL_OFFSET:-}"
case "$PROFILE" in
  dino_imagenet_l11|ijepa_imagenet_l31)
    TASK_ID="imagenet_1k"
    TASK_TYPE="classification"
    TRAIN_TASK_SLUG="imagenet_train"
    VAL_TASK_SLUG="imagenet_val"
    TRAIN_MANIFEST="$DATA_ROOT/imagenet/train_manifest.jsonl"
    VAL_MANIFEST="$DATA_ROOT/imagenet/val_manifest.jsonl"
    ;;
  dino_nyuv2_l11|ijepa_nyuv2_l31)
    TASK_ID="nyuv2_depth"
    TASK_TYPE="dense_depth"
    TRAIN_TASK_SLUG="nyuv2_train"
    VAL_TASK_SLUG="nyuv2_val"
    TRAIN_MANIFEST="$DATA_ROOT/nyuv2/train_manifest.jsonl"
    VAL_MANIFEST="$DATA_ROOT/nyuv2/val_manifest.jsonl"
    ;;
  dino_ade20k_l11|ijepa_ade20k_l31)
    TASK_ID="ade20k_segmentation"
    TASK_TYPE="dense_segmentation"
    TRAIN_TASK_SLUG="ade20k_train"
    VAL_TASK_SLUG="ade20k_val"
    TRAIN_MANIFEST="$DATA_ROOT/ade20k/train_manifest.jsonl"
    VAL_MANIFEST="$DATA_ROOT/ade20k/val_manifest.jsonl"
    NUM_CLASSES="${NUM_CLASSES:-150}"
    IGNORE_INDEX="${IGNORE_INDEX:-255}"
    # Official ADE20K annotations use 0 for background/ignore and 1..150
    # for semantic classes. Public probes expect 0..149 plus IGNORE_INDEX.
    SEGMENTATION_IGNORE_VALUE="${SEGMENTATION_IGNORE_VALUE:-0}"
    SEGMENTATION_LABEL_OFFSET="${SEGMENTATION_LABEL_OFFSET:--1}"
    ;;
  dino_clevr_count_l11|ijepa_clevr_count_l31)
    TASK_ID="clevr_count"
    TASK_TYPE="count_classification"
    TRAIN_TASK_SLUG="clevr_count_train"
    VAL_TASK_SLUG="clevr_count_val"
    TRAIN_MANIFEST="$DATA_ROOT/clevr_count/train_manifest.jsonl"
    VAL_MANIFEST="$DATA_ROOT/clevr_count/val_manifest.jsonl"
    ;;
esac
TASK_SLUG="$VAL_TASK_SLUG"
MANIFEST="$VAL_MANIFEST"

if [[ "$PROFILE" == "dino_imagenet_l11" || "$PROFILE" == "dino_nyuv2_l11" || "$PROFILE" == "dino_ade20k_l11" || "$PROFILE" == "dino_clevr_count_l11" ]]; then
  MODEL_ID="dino_v2_base"
  SAE_ID="dino_l11_topk32_exp4"
  LAYER="11"
  FEATURE_BACKEND="huggingface"
  TRAIN_FEATURE_DIR="$ARTIFACT_ROOT/features/$MODEL_ID/${TRAIN_TASK_SLUG}_l11"
  VAL_FEATURE_DIR="$ARTIFACT_ROOT/features/$MODEL_ID/${VAL_TASK_SLUG}_l11"
elif [[ "$PROFILE" == "ijepa_imagenet_l31" || "$PROFILE" == "ijepa_nyuv2_l31" || "$PROFILE" == "ijepa_ade20k_l31" || "$PROFILE" == "ijepa_clevr_count_l31" ]]; then
  MODEL_ID="ijepa_vit_h14"
  SAE_ID="ijepa_l31_topk32_exp4"
  LAYER="31"
  TRAIN_FEATURE_DIR="$ARTIFACT_ROOT/features/$MODEL_ID/${TRAIN_TASK_SLUG}_l31"
  VAL_FEATURE_DIR="$ARTIFACT_ROOT/features/$MODEL_ID/${VAL_TASK_SLUG}_l31"
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
FEATURE_DIR="$VAL_FEATURE_DIR"
if [[ "$CONTRIBUTION_SCORING_METHOD" == "auto" ]]; then
  if [[ "$TASK_TYPE" == "classification" || "$TASK_TYPE" == "count_classification" ]]; then
    CONTRIBUTION_SCORING_METHOD="true_class_logit_drop"
  else
    CONTRIBUTION_SCORING_METHOD="weight_activation"
  fi
fi
SEGMENTATION_LABEL_OFFSET="${SEGMENTATION_LABEL_OFFSET:-0}"
SAE_CHECKPOINT="$SAE_ROOT/$SAE_ID/final_sae.pt"
LIGHTWEIGHT_SAE="$ARTIFACT_ROOT/checkpoints/${SAE_ID}_lightweight.npz"
CODES_DIR="$ARTIFACT_ROOT/codes/$SAE_ID/$TASK_SLUG"
TRAIN_CODES_DIR="$ARTIFACT_ROOT/codes/$SAE_ID/$TRAIN_TASK_SLUG"
VAL_CODES_DIR="$ARTIFACT_ROOT/codes/$SAE_ID/$VAL_TASK_SLUG"
TARGET_DIR="$ARTIFACT_ROOT/targets/$TASK_ID/$SPLIT"
TRAIN_TARGET_DIR="$ARTIFACT_ROOT/targets/$TASK_ID/$TRAIN_SPLIT"
VAL_TARGET_DIR="$ARTIFACT_ROOT/targets/$TASK_ID/$VAL_SPLIT"
NATIVE_PROBE_DIR="$ARTIFACT_ROOT/probes/native/$MODEL_ID/$TASK_SLUG"
SAE_PROBE_DIR="$ARTIFACT_ROOT/probes/sae/$SAE_ID/$TASK_SLUG"
AVAILABILITY_DIR="$ARTIFACT_ROOT/analysis/availability/$SAE_ID/$TASK_SLUG"
CONTRIBUTION_DIR="$ARTIFACT_ROOT/analysis/contribution/$SAE_ID/$TASK_SLUG"
RANKING_CONTROL_ROOT="$ARTIFACT_ROOT/analysis/ranking_controls/$SAE_ID/$TASK_SLUG"
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
  "$TRAIN_FEATURE_DIR" \
  "$FEATURE_DIR" \
  "$TRAIN_CODES_DIR" \
  "$CODES_DIR" \
  "$TRAIN_TARGET_DIR" \
  "$TARGET_DIR" \
  "$NATIVE_PROBE_DIR" \
  "$SAE_PROBE_DIR" \
  "$AVAILABILITY_DIR" \
  "$CONTRIBUTION_DIR" \
  "$RANKING_CONTROL_ROOT" \
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

if [[ "$PROBE_BACKEND" == "paper-scale-torch" ]]; then
  run python -m feature_economy.cli.main check-manifest \
    --manifest "$TRAIN_MANIFEST" \
    --task-type "$TASK_TYPE" \
    --expected-split "$TRAIN_SPLIT"
fi

extract_features_split() {
  local split="$1"
  local manifest="$2"
  local output_dir="$3"
  if [[ "$FEATURE_BACKEND" == "huggingface" && ${#optional_args[@]} -gt 0 ]]; then
    run python -m feature_economy.cli.main extract-features \
      --config-root "$CONFIG_ROOT" \
      --backend huggingface \
      --manifest "$manifest" \
      --task-type "$TASK_TYPE" \
      --model-id "$MODEL_ID" \
      --layer "$LAYER" \
      --expected-split "$split" \
      --batch-size "$BATCH_SIZE" \
      --device "$DEVICE" \
      --dtype "$DTYPE" \
      --output-dir "$output_dir" \
      "${optional_args[@]}"
  elif [[ "$FEATURE_BACKEND" == "huggingface" ]]; then
    run python -m feature_economy.cli.main extract-features \
      --config-root "$CONFIG_ROOT" \
      --backend huggingface \
      --manifest "$manifest" \
      --task-type "$TASK_TYPE" \
      --model-id "$MODEL_ID" \
      --layer "$LAYER" \
      --expected-split "$split" \
      --batch-size "$BATCH_SIZE" \
      --device "$DEVICE" \
      --dtype "$DTYPE" \
      --output-dir "$output_dir"
  elif [[ "$FEATURE_BACKEND" == "torchscript" ]]; then
    run python -m feature_economy.cli.main extract-features \
      --config-root "$CONFIG_ROOT" \
      --backend torchscript \
      --checkpoint "$IJEPA_TORCHSCRIPT_CHECKPOINT" \
      --manifest "$manifest" \
      --task-type "$TASK_TYPE" \
      --model-id "$MODEL_ID" \
      --layer "$LAYER" \
      --expected-split "$split" \
      --batch-size "$BATCH_SIZE" \
      --device "$DEVICE" \
      --dtype "$DTYPE" \
      --output-dir "$output_dir" \
      "${torchscript_optional_args[@]}"
  fi
}

if [[ "$PROBE_BACKEND" == "paper-scale-torch" ]]; then
  extract_features_split "$TRAIN_SPLIT" "$TRAIN_MANIFEST" "$TRAIN_FEATURE_DIR"
fi
extract_features_split "$VAL_SPLIT" "$VAL_MANIFEST" "$VAL_FEATURE_DIR"

run python -m feature_economy.cli.main validate-arrays \
  --npz "$FEATURE_DIR/features.npz" \
  --kind features \
  --task-type "$TASK_TYPE" \
  --manifest "$MANIFEST" \
  --expected-split "$SPLIT"

if [[ "$PROBE_BACKEND" == "paper-scale-torch" ]]; then
  run python -m feature_economy.cli.main validate-arrays \
    --npz "$TRAIN_FEATURE_DIR/features.npz" \
    --kind features \
    --task-type "$TASK_TYPE" \
    --manifest "$TRAIN_MANIFEST" \
    --expected-split "$TRAIN_SPLIT"
fi

linear_probe_target_args=()
paper_probe_target_args=(--target-key "$TARGET_KEY")
if [[ "$TASK_TYPE" == "dense_depth" || "$TASK_TYPE" == "dense_segmentation" ]]; then
  export_targets_split() {
    local split="$1"
    local manifest="$2"
    local feature_dir="$3"
    local target_dir="$4"
    export_target_args=(
      --manifest "$manifest"
      --task-type "$TASK_TYPE"
      --expected-split "$split"
      --features-npz "$feature_dir/features.npz"
      --output-dir "$target_dir"
    )
    if [[ "$TASK_TYPE" == "dense_segmentation" ]]; then
      export_target_args+=(
        --segmentation-output-ignore-index "$IGNORE_INDEX"
        --segmentation-label-offset "$SEGMENTATION_LABEL_OFFSET"
      )
      if [[ -n "$SEGMENTATION_IGNORE_VALUE" ]]; then
        export_target_args+=(--segmentation-ignore-value "$SEGMENTATION_IGNORE_VALUE")
      fi
    fi
    run python -m feature_economy.cli.main export-targets \
      "${export_target_args[@]}"
    run python -m feature_economy.cli.main validate-arrays \
      --npz "$target_dir/targets.npz" \
      --kind targets \
      --task-type "$TASK_TYPE" \
      --manifest "$manifest" \
      --expected-split "$split"
  }
  if [[ "$PROBE_BACKEND" == "paper-scale-torch" ]]; then
    export_targets_split "$TRAIN_SPLIT" "$TRAIN_MANIFEST" "$TRAIN_FEATURE_DIR" "$TRAIN_TARGET_DIR"
  fi
  export_targets_split "$VAL_SPLIT" "$VAL_MANIFEST" "$VAL_FEATURE_DIR" "$VAL_TARGET_DIR"

  linear_probe_target_args=(--targets-npz "$VAL_TARGET_DIR/targets.npz")
  paper_probe_target_args=(
    --target-key "$TARGET_KEY"
    --train-targets-npz "$TRAIN_TARGET_DIR/targets.npz"
    --val-targets-npz "$VAL_TARGET_DIR/targets.npz"
  )
  if [[ "$TASK_TYPE" == "dense_segmentation" ]]; then
    linear_probe_target_args+=(--num-classes "$NUM_CLASSES" --ignore-index "$IGNORE_INDEX")
    paper_probe_target_args+=(--num-classes "$NUM_CLASSES" --ignore-index "$IGNORE_INDEX")
  fi
fi

if [[ "$PROBE_BACKEND" == "paper-scale-torch" ]]; then
  run python -m feature_economy.cli.main probe-native \
    --backend paper-scale-torch \
    --train-features-npz "$TRAIN_FEATURE_DIR/features.npz" \
    --val-features-npz "$VAL_FEATURE_DIR/features.npz" \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --task-type "$TASK_TYPE" \
    --task-id "$TASK_ID" \
    --model-id "$MODEL_ID" \
    --expected-split "$VAL_SPLIT" \
    --epochs "$PROBE_EPOCHS" \
    --batch-size "$PROBE_BATCH_SIZE" \
    --lr "$PROBE_LR" \
    --weight-decay "$PROBE_WEIGHT_DECAY" \
    --decoder-hidden-channels "$DECODER_HIDDEN_CHANNELS" \
    --device "$DEVICE" \
    "${paper_probe_target_args[@]}" \
    --output-dir "$NATIVE_PROBE_DIR"
elif [[ ${#linear_probe_target_args[@]} -gt 0 ]]; then
  run python -m feature_economy.cli.main probe-native \
    --backend linear-probe \
    --features-npz "$FEATURE_DIR/features.npz" \
    --manifest "$MANIFEST" \
    --task-type "$TASK_TYPE" \
    --task-id "$TASK_ID" \
    --model-id "$MODEL_ID" \
    --expected-split "$SPLIT" \
    --ridge "$RIDGE" \
    "${linear_probe_target_args[@]}" \
    --output-dir "$NATIVE_PROBE_DIR"
else
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
fi

run python -m feature_economy.cli.main convert-sae-checkpoint \
  --input-checkpoint "$SAE_CHECKPOINT" \
  --output-checkpoint "$LIGHTWEIGHT_SAE"

extract_codes_split() {
  local feature_dir="$1"
  local output_dir="$2"
  run python -m feature_economy.cli.main extract-sae-codes \
    --config-root "$CONFIG_ROOT" \
    --backend linear-topk \
    --features-npz "$feature_dir/features.npz" \
    --model-id "$MODEL_ID" \
    --sae-id "$SAE_ID" \
    --checkpoint "$LIGHTWEIGHT_SAE" \
    --output-dir "$output_dir"
}

if [[ "$PROBE_BACKEND" == "paper-scale-torch" ]]; then
  extract_codes_split "$TRAIN_FEATURE_DIR" "$TRAIN_CODES_DIR"
fi
extract_codes_split "$VAL_FEATURE_DIR" "$VAL_CODES_DIR"

run python -m feature_economy.cli.main validate-arrays \
  --npz "$CODES_DIR/codes.npz" \
  --kind codes \
  --task-type "$TASK_TYPE" \
  --manifest "$MANIFEST" \
  --expected-split "$SPLIT"

if [[ "$PROBE_BACKEND" == "paper-scale-torch" ]]; then
  run python -m feature_economy.cli.main validate-arrays \
    --npz "$TRAIN_CODES_DIR/codes.npz" \
    --kind codes \
    --task-type "$TASK_TYPE" \
    --manifest "$TRAIN_MANIFEST" \
    --expected-split "$TRAIN_SPLIT"
fi

if [[ "$PROBE_BACKEND" == "paper-scale-torch" ]]; then
  run python -m feature_economy.cli.main probe-sae \
    --backend paper-scale-torch \
    --train-codes-npz "$TRAIN_CODES_DIR/codes.npz" \
    --val-codes-npz "$VAL_CODES_DIR/codes.npz" \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --task-type "$TASK_TYPE" \
    --task-id "$TASK_ID" \
    --model-id "$MODEL_ID" \
    --sae-id "$SAE_ID" \
    --expected-split "$VAL_SPLIT" \
    --epochs "$PROBE_EPOCHS" \
    --batch-size "$PROBE_BATCH_SIZE" \
    --lr "$PROBE_LR" \
    --weight-decay "$PROBE_WEIGHT_DECAY" \
    --decoder-hidden-channels "$DECODER_HIDDEN_CHANNELS" \
    --device "$DEVICE" \
    "${paper_probe_target_args[@]}" \
    --output-dir "$SAE_PROBE_DIR"
elif [[ ${#linear_probe_target_args[@]} -gt 0 ]]; then
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
    "${linear_probe_target_args[@]}" \
    --output-dir "$SAE_PROBE_DIR"
else
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
fi

run python -m feature_economy.cli.main compute-usage \
  --config-root "$CONFIG_ROOT" \
  --codes-npz "$CODES_DIR/codes.npz" \
  --model-id "$MODEL_ID" \
  --sae-id "$SAE_ID" \
  --dataset-id "${TASK_ID}_${SPLIT}" \
  --split "$SPLIT" \
  --output-dir "$AVAILABILITY_DIR"

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

for ranking_method in $RANKING_METHODS; do
  if [[ "$ranking_method" != "probe_weight" \
    && "$ranking_method" != "validation_contribution" \
    && "$ranking_method" != "hybrid" ]]; then
    echo "Unsupported ranking method in RANKING_METHODS: $ranking_method" >&2
    exit 2
  fi
  ranking_dir="$RANKING_CONTROL_ROOT/$ranking_method/ranking"
  subset_usage_dir="$RANKING_CONTROL_ROOT/$ranking_method/subset_usage"
  ablation_dir="$RANKING_CONTROL_ROOT/$ranking_method/ablation"
  run mkdir -p "$ranking_dir" "$subset_usage_dir" "$ablation_dir"

  ranking_args=(
    --config-root "$CONFIG_ROOT"
    --codes-npz "$CODES_DIR/codes.npz"
    --probe-logits-npz "$SAE_PROBE_DIR/probe_logits.npz"
    --task-id "$TASK_ID"
    --model-id "$MODEL_ID"
    --sae-id "$SAE_ID"
    --top-k 100
    --ranking-method "$ranking_method"
  )
  if [[ "$ranking_method" == "validation_contribution" || "$ranking_method" == "hybrid" ]]; then
    ranking_args+=(--contribution-npz "$CONTRIBUTION_DIR/contribution_scores.npz")
  fi
  if [[ "$ranking_method" == "hybrid" ]]; then
    ranking_args+=(--hybrid-alpha "$HYBRID_ALPHA")
  fi
  run python -m feature_economy.cli.main rank-features \
    "${ranking_args[@]}" \
    --output-dir "$ranking_dir"

  run python -m feature_economy.cli.main compute-subset-usage \
    --config-root "$CONFIG_ROOT" \
    --codes-npz "$CODES_DIR/codes.npz" \
    --ranking-json "$ranking_dir/task_feature_ranking.json" \
    --top-k 100 \
    --random-seed 0 \
    --output-dir "$subset_usage_dir"

  run python -m feature_economy.cli.main ablate-features \
    --config-root "$CONFIG_ROOT" \
    --backend linear-probe \
    --codes-npz "$CODES_DIR/codes.npz" \
    --probe-logits-npz "$SAE_PROBE_DIR/probe_logits.npz" \
    --ranking-json "$ranking_dir/task_feature_ranking.json" \
    --task-type "$TASK_TYPE" \
    --top-k 100 \
    --random-seed 0 \
    --output-dir "$ablation_dir"
done

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
probe_backend=$PROBE_BACKEND
probe_epochs=$PROBE_EPOCHS
probe_batch_size=$PROBE_BATCH_SIZE
ranking_methods=$RANKING_METHODS
hybrid_alpha=$HYBRID_ALPHA
num_classes=$NUM_CLASSES
ignore_index=$IGNORE_INDEX
segmentation_ignore_value=$SEGMENTATION_IGNORE_VALUE
segmentation_label_offset=$SEGMENTATION_LABEL_OFFSET
artifact_root=$ARTIFACT_ROOT
train_manifest=$TRAIN_MANIFEST
val_manifest=$VAL_MANIFEST
sae_checkpoint=$SAE_CHECKPOINT
layer=$LAYER
EOF"

echo
echo "Full profile completed: $PROFILE"
echo "Artifact root: $ARTIFACT_ROOT"
