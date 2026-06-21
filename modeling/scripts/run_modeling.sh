#!/usr/bin/env bash
set -euo pipefail
SCRIPT_NAME="run_modeling.sh"
if [ -n "${MODEL_ROOT:-}" ] && [ -f "$MODEL_ROOT/scripts/_run_selected_implementation.sh" ]; then
  RUNNER="$MODEL_ROOT/scripts/_run_selected_implementation.sh"
elif [ -n "${PROJECT_DIR:-}" ] && [ -f "$PROJECT_DIR/modeling/scripts/_run_selected_implementation.sh" ]; then
  RUNNER="$PROJECT_DIR/modeling/scripts/_run_selected_implementation.sh"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/modeling/scripts/_run_selected_implementation.sh" ]; then
  RUNNER="$SLURM_SUBMIT_DIR/modeling/scripts/_run_selected_implementation.sh"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  RUNNER="$SCRIPT_DIR/_run_selected_implementation.sh"
fi
exec bash "$RUNNER" "$SCRIPT_NAME" "$@"
