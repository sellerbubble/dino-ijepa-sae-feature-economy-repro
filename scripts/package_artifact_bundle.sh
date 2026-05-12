#!/usr/bin/env bash
# Role: validate, sanitize, and package a saved-array artifact bundle as a release asset.
# Status: public release utility
# Used by: maintainers before uploading public saved-array release assets
# Inputs: artifact root, run-plan JSON, optional output directory/replacement rules
# Outputs: sanitized metadata report, .tar.gz archive, .sha256 checksum, and release manifest JSON
# Safe to move/delete?: keep; this is the release gate for public artifact assets.
# Notes: This packages saved artifacts only; it does not create scientific results.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARTIFACT_ROOT=""
RUN_PLAN_JSON=""
OUTPUT_DIR=""
BUNDLE_NAME=""
WITH_FIGURES=0
FORCE=0
SKIP_SANITIZE=0
SANITIZE_REPLACEMENTS=()

usage() {
  cat <<'EOF'
Usage: scripts/package_artifact_bundle.sh --artifact-root PATH --run-plan-json PATH [options]

Validate a saved-array artifact bundle, regenerate its index and tables, then
package it as a tar.gz release asset with a SHA256 checksum and manifest.

Required:
  --artifact-root PATH   Existing artifact bundle root.
  --run-plan-json PATH   Run-plan JSON used by check-bundle.

Options:
  --output-dir PATH      Destination for archive/checksum/manifest.
                         Defaults to <artifact-root>/release_assets.
  --bundle-name NAME     Archive stem. Defaults to artifact root basename.
  --sanitize-replace OLD=NEW
                         Metadata replacement before packaging. Can be passed
                         multiple times. If omitted, metadata is still scanned
                         and packaging fails on private absolute path markers.
  --skip-sanitize        Disable metadata sanitization/scan gate. Use only for
                         local debugging, not public release assets.
  --with-figures         Also run make-figures before packaging.
  --force                Replace existing archive/checksum/manifest.
  -h, --help             Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact-root)
      ARTIFACT_ROOT="$2"
      shift 2
      ;;
    --run-plan-json)
      RUN_PLAN_JSON="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --bundle-name)
      BUNDLE_NAME="$2"
      shift 2
      ;;
    --sanitize-replace)
      SANITIZE_REPLACEMENTS+=("$2")
      shift 2
      ;;
    --skip-sanitize)
      SKIP_SANITIZE=1
      shift
      ;;
    --with-figures)
      WITH_FIGURES=1
      shift
      ;;
    --force)
      FORCE=1
      shift
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

if [[ -z "${ARTIFACT_ROOT}" || -z "${RUN_PLAN_JSON}" ]]; then
  echo "--artifact-root and --run-plan-json are required." >&2
  usage >&2
  exit 2
fi

ARTIFACT_ROOT="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${ARTIFACT_ROOT}")"
RUN_PLAN_JSON="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${RUN_PLAN_JSON}")"

if [[ ! -d "${ARTIFACT_ROOT}" ]]; then
  echo "Artifact root does not exist: ${ARTIFACT_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${RUN_PLAN_JSON}" ]]; then
  echo "Run-plan JSON does not exist: ${RUN_PLAN_JSON}" >&2
  exit 2
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${ARTIFACT_ROOT}/release_assets"
fi
OUTPUT_DIR="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${OUTPUT_DIR}")"

if [[ -z "${BUNDLE_NAME}" ]]; then
  BUNDLE_NAME="$(basename "${ARTIFACT_ROOT}")"
fi

case "${BUNDLE_NAME}" in
  *[!A-Za-z0-9._-]*|"")
    echo "--bundle-name may contain only letters, numbers, '.', '_', and '-'." >&2
    exit 2
    ;;
esac

mkdir -p "${ARTIFACT_ROOT}/run_plan" "${ARTIFACT_ROOT}/index" "${ARTIFACT_ROOT}/tables" "${OUTPUT_DIR}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m feature_economy.cli.main check-bundle \
  --run-plan-json "${RUN_PLAN_JSON}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --output-json "${ARTIFACT_ROOT}/run_plan/artifact_bundle_check_final.json" \
  --require-complete

rm -f "${ARTIFACT_ROOT}/index/artifact_index.json" "${ARTIFACT_ROOT}/index/artifact_index.csv"
rm -f "${ARTIFACT_ROOT}/tables/"*.csv

python -m feature_economy.cli.main index-artifacts \
  --input-dir "${ARTIFACT_ROOT}" \
  --output-json "${ARTIFACT_ROOT}/index/artifact_index.json" \
  --output-csv "${ARTIFACT_ROOT}/index/artifact_index.csv" \
  --require-valid

python -m feature_economy.cli.main make-tables \
  --artifact-index "${ARTIFACT_ROOT}/index/artifact_index.json" \
  --output-dir "${ARTIFACT_ROOT}/tables"

if [[ "${WITH_FIGURES}" == "1" ]]; then
  python -m feature_economy.cli.main make-figures \
    --table-dir "${ARTIFACT_ROOT}/tables" \
    --output-dir "${ARTIFACT_ROOT}/figures" \
    --formats png,pdf
fi

SANITIZE_REPORT="${ARTIFACT_ROOT}/index/artifact_metadata_sanitization_report.json"
if [[ "${SKIP_SANITIZE}" != "1" ]]; then
  SANITIZE_ARGS=(
    "${REPO_DIR}/scripts/sanitize_artifact_bundle_metadata.py"
    --artifact-root "${ARTIFACT_ROOT}"
    --output-json "${SANITIZE_REPORT}"
    --require-no-private-paths
  )
  if ((${#SANITIZE_REPLACEMENTS[@]} > 0)); then
    for rule in "${SANITIZE_REPLACEMENTS[@]}"; do
      SANITIZE_ARGS+=(--replace "${rule}")
    done
  fi
  python "${SANITIZE_ARGS[@]}"
fi

ARCHIVE_PATH="${OUTPUT_DIR}/${BUNDLE_NAME}.tar.gz"
CHECKSUM_PATH="${ARCHIVE_PATH}.sha256"
MANIFEST_PATH="${OUTPUT_DIR}/${BUNDLE_NAME}_release_manifest.json"

if [[ "${FORCE}" != "1" ]]; then
  for path in "${ARCHIVE_PATH}" "${CHECKSUM_PATH}" "${MANIFEST_PATH}"; do
    if [[ -e "${path}" ]]; then
      echo "Refusing to overwrite existing file: ${path}" >&2
      echo "Re-run with --force to replace release assets." >&2
      exit 2
    fi
  done
fi

rm -f "${ARCHIVE_PATH}" "${CHECKSUM_PATH}" "${MANIFEST_PATH}"

PARENT_DIR="$(dirname "${ARTIFACT_ROOT}")"
ROOT_BASENAME="$(basename "${ARTIFACT_ROOT}")"
tar \
  --exclude './release_assets' \
  --exclude './*.tar.gz' \
  --exclude './*.sha256' \
  --exclude './._*' \
  --exclude '*/._*' \
  -C "${PARENT_DIR}" \
  -czf "${ARCHIVE_PATH}" \
  "${ROOT_BASENAME}"

python - <<PY
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

archive = Path("${ARCHIVE_PATH}")
checksum_path = Path("${CHECKSUM_PATH}")
manifest_path = Path("${MANIFEST_PATH}")
artifact_root = Path("${ARTIFACT_ROOT}")
run_plan = Path("${RUN_PLAN_JSON}")

digest = hashlib.sha256()
with archive.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
sha256 = digest.hexdigest()
checksum_path.write_text(f"{sha256}  {archive.name}\n", encoding="utf-8")

manifest = {
    "record_type": "artifact_release_manifest",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "bundle_name": "${BUNDLE_NAME}",
    "archive": archive.name,
    "archive_bytes": archive.stat().st_size,
    "sha256": sha256,
    "artifact_root_name": artifact_root.name,
    "run_plan_json": f"{artifact_root.name}/run_plan/{run_plan.name}",
    "bundle_check_json": f"{artifact_root.name}/run_plan/artifact_bundle_check_final.json",
    "artifact_index_json": f"{artifact_root.name}/index/artifact_index.json",
    "metadata_sanitization_json": (
        f"{artifact_root.name}/index/artifact_metadata_sanitization_report.json"
        if ${SKIP_SANITIZE} == 0
        else None
    ),
    "metadata_sanitization_required": bool(${SKIP_SANITIZE} == 0),
    "tables_dir": f"{artifact_root.name}/tables",
    "figures_included": bool(${WITH_FIGURES}),
}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

echo "Packaged artifact release asset:"
echo "  archive:  ${ARCHIVE_PATH}"
echo "  checksum: ${CHECKSUM_PATH}"
echo "  manifest: ${MANIFEST_PATH}"
