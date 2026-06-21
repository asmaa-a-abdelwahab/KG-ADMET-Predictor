#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$1")"
shift || true

PROJECT_DIR="${PROJECT_DIR:-/home/asmaaali/KG-ADMET-Predictor}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_DIR/modeling}"
IMPL="${MODEL_IMPL:-${MODEL_IMPLEMENTATION:-improved}}"

case "$IMPL" in
  legacy|old) IMPL="legacy" ;;
  improved|new) IMPL="improved" ;;
  *)
    echo "ERROR: Unknown MODEL_IMPL='$IMPL'. Use MODEL_IMPL=legacy or MODEL_IMPL=improved." >&2
    exit 2
    ;;
esac

IMPL_DIR="$MODEL_ROOT/implementations/$IMPL"
if [ ! -d "$IMPL_DIR" ]; then
  echo "ERROR: Implementation directory not found: $IMPL_DIR" >&2
  echo "Expected one of:" >&2
  echo "  $MODEL_ROOT/implementations/legacy" >&2
  echo "  $MODEL_ROOT/implementations/improved" >&2
  exit 2
fi

bash "$MODEL_ROOT/scripts/use_implementation.sh" "$IMPL"

export MODEL_IMPL="$IMPL"
export MODEL_IMPLEMENTATION="$IMPL"
export MODELING_PACKAGE_DIR="$MODEL_ROOT"
export PYTHONPATH="$MODEL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Keep old and improved outputs separate by default. Override these env vars
# manually if you intentionally want a shared output directory.
export MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-$PROJECT_DIR/models_all_stages_${IMPL}}"
export MODEL_REPORT_DIR="${MODEL_REPORT_DIR:-$PROJECT_DIR/reports/all_stages_${IMPL}}"

mkdir -p "$PROJECT_DIR/logs" "$MODEL_OUTPUT_DIR" "$MODEL_REPORT_DIR"

echo "============================================================"
echo "Selected PRING modeling implementation: $IMPL"
echo "Running script: $SCRIPT_NAME"
echo "Package dir: $MODELING_PACKAGE_DIR"
echo "Output dir:  $MODEL_OUTPUT_DIR"
echo "Report dir:  $MODEL_REPORT_DIR"
echo "============================================================"

exec bash "$IMPL_DIR/scripts/$SCRIPT_NAME" "$@"
