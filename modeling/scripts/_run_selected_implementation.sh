#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /bin/bash "$0" "$@"
fi

set -euo pipefail

SCRIPT_NAME="$(basename "$1")"
shift || true

# When a wrapper is submitted via sbatch, SLURM may rename the copied script to
# "slurm_script". Top-level wrappers should pass a hard-coded script name, but
# keep this fallback to make old wrappers easier to diagnose.
if [ "$SCRIPT_NAME" = "slurm_script" ]; then
  case "${SLURM_JOB_NAME:-}" in
    all_models_compare) SCRIPT_NAME="run_all_models_compare_hpc.sh" ;;
    stage2_fast) SCRIPT_NAME="run_stage2_fast_no_neo4j.sh" ;;
    stage3_sampled_gpu) SCRIPT_NAME="run_stage3_sampled_gpu_no_neo4j.sh" ;;
    stage3_no_neo4j) SCRIPT_NAME="run_stage3_no_neo4j.sh" ;;
    stage3_gpu_array) SCRIPT_NAME="run_stage3_gpu_array_no_neo4j.sh" ;;
    final_validation) SCRIPT_NAME="run_final_validation_hpc.sh" ;;
    *)
      echo "ERROR: SLURM renamed the submitted script to slurm_script and the original wrapper name is unknown." >&2
      echo "Set SELECTED_SCRIPT_NAME to one of the modeling/scripts/run_*.sh wrapper names or update the top-level wrapper." >&2
      exit 2
      ;;
  esac
fi

PROJECT_DIR="${PROJECT_DIR:-/home/asmaaali/KG-ADMET-Predictor}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_DIR/modeling}"
IMPL="${MODEL_IMPL:-${MODEL_IMPLEMENTATION:-improved}}"

case "$IMPL" in
  legacy|old) IMPL="legacy" ;;
  improved|new) IMPL="improved" ;;
  improved_v2|final|finalized|v2) IMPL="improved_v2" ;;
  *)
    echo "ERROR: Unknown MODEL_IMPL='$IMPL'. Use MODEL_IMPL=legacy, MODEL_IMPL=improved, or MODEL_IMPL=improved_v2." >&2
    exit 2
    ;;
esac

IMPL_DIR="$MODEL_ROOT/implementations/$IMPL"
if [ ! -d "$IMPL_DIR" ]; then
  echo "ERROR: Implementation directory not found: $IMPL_DIR" >&2
  echo "Expected one of:" >&2
  echo "  $MODEL_ROOT/implementations/legacy" >&2
  echo "  $MODEL_ROOT/implementations/improved" >&2
  echo "  $MODEL_ROOT/implementations/improved_v2" >&2
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
