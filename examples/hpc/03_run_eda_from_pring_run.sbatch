#!/bin/bash
#SBATCH --job-name=pring-eda
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$PWD}
PRING_DIR=${PRING_DIR:-$PROJECT_DIR/../pring}
RUN_DIR=${RUN_DIR:-$PROJECT_DIR/runs/current}
EDA_DIR=${EDA_DIR:-$PROJECT_DIR/artifacts/reports/eda}

module purge || true
module load python/3.11 || true

python -m venv "$PROJECT_DIR/.venv-pring-app"
source "$PROJECT_DIR/.venv-pring-app/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "$PRING_DIR"

python -m pring eda --run-path "$RUN_DIR" --output-dir "$EDA_DIR"
