#!/usr/bin/env bash
# Role: export public_repro as a clean standalone repository working tree.
# Status: public release utility
# Used by: maintainers before publishing the standalone reproduction repo
# Inputs: public_repro source tree; optional --output path
# Outputs: copied standalone tree under the requested output path
# Safe to move/delete?: keep; this is the bridge from private workbench to public repo.
# Notes: This does not initialize git or publish anything. It refuses unsafe
# deletion targets unless --force is explicit.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/feature_economy_repro_public_export}"
FORCE=0

usage() {
  cat <<'EOF'
Usage: scripts/export_public_repo.sh [--output PATH] [--force]

Export this public_repro tree as a clean standalone repository working tree.

Options:
  --output PATH   Destination directory. Defaults to
                  /tmp/feature_economy_repro_public_export
  --force         Remove an existing destination before copying.
  -h, --help      Show this help message.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      OUTPUT_DIR="$2"
      shift 2
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

OUTPUT_DIR="$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "${OUTPUT_DIR}")"

case "${OUTPUT_DIR}" in
  "/"|"${HOME}"|"${SOURCE_ROOT}"|"${SOURCE_ROOT}/"*)
    echo "Refusing unsafe output path: ${OUTPUT_DIR}" >&2
    exit 2
    ;;
esac

if [[ -e "${OUTPUT_DIR}" ]]; then
  if [[ "${FORCE}" != "1" ]]; then
    echo "Destination already exists: ${OUTPUT_DIR}" >&2
    echo "Re-run with --force to replace it." >&2
    exit 2
  fi
  rm -rf "${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"

if command -v rsync >/dev/null 2>&1; then
  rsync -a \
    --exclude '.git/' \
    --exclude '.DS_Store' \
    --exclude '._*' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.pytest_cache/' \
    --exclude '.mypy_cache/' \
    --exclude '.ruff_cache/' \
    --exclude '.venv/' \
    --exclude 'dist/' \
    --exclude 'build/' \
    --exclude '*.egg-info/' \
    --exclude '/artifacts/' \
    --exclude '/data/' \
    --exclude '/checkpoints/' \
    --exclude '/tmp/' \
    "${SOURCE_ROOT}/" "${OUTPUT_DIR}/"
else
  tar \
    --exclude='.git' \
    --exclude='.DS_Store' \
    --exclude='._*' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='.ruff_cache' \
    --exclude='.venv' \
    --exclude='dist' \
    --exclude='build' \
    --exclude='*.egg-info' \
    --exclude='./artifacts' \
    --exclude='./data' \
    --exclude='./checkpoints' \
    --exclude='./tmp' \
    -C "${SOURCE_ROOT}" -cf - . | tar -C "${OUTPUT_DIR}" -xf -
fi

python "${OUTPUT_DIR}/scripts/check_public_release.py" --root "${OUTPUT_DIR}"

echo "Exported standalone public repo tree: ${OUTPUT_DIR}"
echo "Next validation commands:"
echo "  cd ${OUTPUT_DIR}"
echo "  python -m pip install -e ."
echo "  bash scripts/reproduce_smoke.sh"
echo "  bash scripts/build_tiny_artifact_bundle.sh"
