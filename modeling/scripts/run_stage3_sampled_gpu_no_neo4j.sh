#!/bin/bash
#SBATCH --job-name=stage3_sampled_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/asmaaali/KG-ADMET-Predictor/logs/stage3_sampled_gpu_%j.out
#SBATCH --error=/home/asmaaali/KG-ADMET-Predictor/logs/stage3_sampled_gpu_%j.err

if [ -z "${BASH_VERSION:-}" ]; then
  exec /bin/bash "$0" "$@"
fi

set -euo pipefail
SCRIPT_NAME="run_stage3_sampled_gpu_no_neo4j.sh"

# Important for SLURM: sbatch copies the submitted script into /var/spool/slurm/...
# so dirname(${BASH_SOURCE[0]}) may point to the spool directory instead of the
# project checkout. Prefer MODEL_ROOT/PROJECT_DIR/SLURM_SUBMIT_DIR, then fall
# back to the local script directory for interactive execution.
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

if [ ! -f "$RUNNER" ]; then
  echo "ERROR: Could not find _run_selected_implementation.sh" >&2
  echo "Tried runner path: $RUNNER" >&2
  echo "Set PROJECT_DIR=/home/asmaaali/KG-ADMET-Predictor or MODEL_ROOT=/home/asmaaali/KG-ADMET-Predictor/modeling before sbatch." >&2
  exit 2
fi

exec bash "$RUNNER" "$SCRIPT_NAME" "$@"
