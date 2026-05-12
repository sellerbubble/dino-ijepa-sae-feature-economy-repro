#!/usr/bin/env bash
# Role: stage, validate, and package a public artifact bundle from an export manifest.
# Status: public release utility
# Used by: maintainers after artifact source paths have been filled and audited
# Inputs: export manifest CSV, run-plan JSON, public artifact root
# Outputs: staged artifact root, bundle check, artifact index, tables, release archive/checksum/manifest
# Safe to move/delete?: keep; this is the one-command saved-array release staging path.
# Notes: This assumes the run plan matches the manifest. Use a subset run plan for partial releases.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFEST_CSV=""
RUN_PLAN_JSON=""
ARTIFACT_ROOT=""
RELEASE_OUTPUT_DIR=""
BUNDLE_NAME=""
PAPER_VERSION="DINO/I-JEPA SAE Feature Economy paper draft"
PUBLIC_REPO_COMMIT=""
FORCE=0
DRY_RUN=0
WITH_FIGURES=0
INCLUDE_NEEDS_CONVERSION=0
SANITIZE_REPLACEMENTS=()

usage() {
  cat <<'EOF'
Usage: scripts/stage_artifact_bundle_from_manifest.sh \
  --manifest-csv PATH \
  --run-plan-json PATH \
  --artifact-root PATH \
  --release-output-dir PATH \
  --bundle-name NAME [options]

Options:
  --force                    Allow overwriting copied artifact files and release assets.
  --dry-run                  Audit and simulate copy only; skip bundle/package validation.
  --with-figures             Include generated overview figures in the release archive.
  --include-needs-conversion Copy rows marked NEEDS_CONVERSION. Use only for staged conversion inputs.
  --sanitize-replace OLD=NEW Forward a metadata replacement rule to package_artifact_bundle.sh.
                              Can be passed multiple times.
  --paper-version TEXT       Paper/version string for the generated bundle README.
  --public-repo-commit SHA   Commit string for the generated bundle README.
  -h, --help                 Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest-csv)
      MANIFEST_CSV="$2"
      shift 2
      ;;
    --run-plan-json)
      RUN_PLAN_JSON="$2"
      shift 2
      ;;
    --artifact-root)
      ARTIFACT_ROOT="$2"
      shift 2
      ;;
    --release-output-dir)
      RELEASE_OUTPUT_DIR="$2"
      shift 2
      ;;
    --bundle-name)
      BUNDLE_NAME="$2"
      shift 2
      ;;
    --paper-version)
      PAPER_VERSION="$2"
      shift 2
      ;;
    --public-repo-commit)
      PUBLIC_REPO_COMMIT="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --with-figures)
      WITH_FIGURES=1
      shift
      ;;
    --include-needs-conversion)
      INCLUDE_NEEDS_CONVERSION=1
      shift
      ;;
    --sanitize-replace)
      SANITIZE_REPLACEMENTS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${MANIFEST_CSV}" || -z "${RUN_PLAN_JSON}" || -z "${ARTIFACT_ROOT}" || -z "${RELEASE_OUTPUT_DIR}" || -z "${BUNDLE_NAME}" ]]; then
  echo "--manifest-csv, --run-plan-json, --artifact-root, --release-output-dir, and --bundle-name are required." >&2
  usage >&2
  exit 2
fi

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${PUBLIC_REPO_COMMIT}" ]]; then
  PUBLIC_REPO_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
fi

mkdir -p "${ARTIFACT_ROOT}/run_plan"
mkdir -p "${RELEASE_OUTPUT_DIR}/preflight"
cp "${RUN_PLAN_JSON}" "${ARTIFACT_ROOT}/run_plan/$(basename "${RUN_PLAN_JSON}")"

AUDIT_JSON="${RELEASE_OUTPUT_DIR}/preflight/artifact_export_manifest_audit.json"
COPY_JSON="${RELEASE_OUTPUT_DIR}/preflight/artifact_export_copy_report.json"

python scripts/audit_artifact_export_manifest.py \
  --manifest-csv "${MANIFEST_CSV}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --output-json "${AUDIT_JSON}" \
  --require-ready

COPY_ARGS=(
  --manifest-csv "${MANIFEST_CSV}"
  --artifact-root "${ARTIFACT_ROOT}"
  --output-json "${COPY_JSON}"
)
if [[ "${FORCE}" == "1" ]]; then
  COPY_ARGS+=(--force)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  COPY_ARGS+=(--dry-run)
fi
if [[ "${INCLUDE_NEEDS_CONVERSION}" == "1" ]]; then
  COPY_ARGS+=(--include-needs-conversion)
fi
python scripts/copy_artifact_export_manifest.py "${COPY_ARGS[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "Dry run complete. Skipping check-bundle, index, tables, and package steps."
  exit 0
fi

python -m feature_economy.cli.main check-bundle \
  --run-plan-json "${RUN_PLAN_JSON}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --output-json "${ARTIFACT_ROOT}/run_plan/artifact_bundle_check_final.json" \
  --require-complete

python -m feature_economy.cli.main index-artifacts \
  --input-dir "${ARTIFACT_ROOT}" \
  --output-json "${ARTIFACT_ROOT}/index/artifact_index.json" \
  --output-csv "${ARTIFACT_ROOT}/index/artifact_index.csv" \
  --require-valid

python -m feature_economy.cli.main make-tables \
  --artifact-index "${ARTIFACT_ROOT}/index/artifact_index.json" \
  --output-dir "${ARTIFACT_ROOT}/tables"

README_ARGS=(
  --run-plan-json "${RUN_PLAN_JSON}"
  --output-readme "${ARTIFACT_ROOT}/README.md"
  --bundle-name "${BUNDLE_NAME}"
  --paper-version "${PAPER_VERSION}"
  --public-repo-commit "${PUBLIC_REPO_COMMIT}"
)
if [[ "${WITH_FIGURES}" == "1" ]]; then
  README_ARGS+=(--figures-included)
fi
python scripts/create_artifact_bundle_readme.py "${README_ARGS[@]}"

PACKAGE_ARGS=(
  --artifact-root "${ARTIFACT_ROOT}"
  --run-plan-json "${RUN_PLAN_JSON}"
  --output-dir "${RELEASE_OUTPUT_DIR}"
  --bundle-name "${BUNDLE_NAME}"
)
if [[ "${FORCE}" == "1" ]]; then
  PACKAGE_ARGS+=(--force)
fi
if [[ "${WITH_FIGURES}" == "1" ]]; then
  PACKAGE_ARGS+=(--with-figures)
fi
if ((${#SANITIZE_REPLACEMENTS[@]} > 0)); then
  for rule in "${SANITIZE_REPLACEMENTS[@]}"; do
    PACKAGE_ARGS+=(--sanitize-replace "${rule}")
  done
fi
bash scripts/package_artifact_bundle.sh "${PACKAGE_ARGS[@]}"
