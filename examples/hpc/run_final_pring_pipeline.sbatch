#!/usr/bin/env bash
#SBATCH --job-name=pring-final-thesis
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00
#SBATCH --output=/home/asmaaali/PRING-APP/slurm_logs/pring-final-thesis-%j.out
#SBATCH --error=/home/asmaaali/PRING-APP/slurm_logs/pring-final-thesis-%j.err

set -Eeuo pipefail
umask 027

PROJECT_ROOT="${PROJECT_ROOT:-/home/asmaaali}"
PACKAGE_DIR="${PACKAGE_DIR:-$PROJECT_ROOT/PRING-PACKAGE}"
APP_DIR="${APP_DIR:-$PROJECT_ROOT/PRING-APP}"
VENV_DIR="${PRING_VENV:-/home/asmaaali/venvs/pring-env}"
CONFIG="${PRING_FINAL_CONFIG:-$APP_DIR/configs/final_thesis.yaml}"
RUNNER="${PRING_FINAL_RUNNER:-$APP_DIR/run_final_pring_pipeline.py}"

mkdir -p "$APP_DIR/slurm_logs"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "ERROR: Python environment not found: $VENV_DIR" >&2
  exit 2
fi
if [ ! -f "$RUNNER" ]; then
  echo "ERROR: Final pipeline runner not found: $RUNNER" >&2
  exit 2
fi
if [ ! -f "$CONFIG" ]; then
  echo "ERROR: Final thesis config not found: $CONFIG" >&2
  exit 2
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PACKAGE_DIR:$APP_DIR/modeling${PYTHONPATH:+:$PYTHONPATH}"
export PRING_CPUS="${SLURM_CPUS_PER_TASK:-24}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# NEO4J_PASSWORD must be injected by the scheduler, shell, or secret manager.
: "${NEO4J_PASSWORD:?NEO4J_PASSWORD must be exported before sbatch submission}"

exec python "$RUNNER" \
  --config "$CONFIG" \
  --resume
