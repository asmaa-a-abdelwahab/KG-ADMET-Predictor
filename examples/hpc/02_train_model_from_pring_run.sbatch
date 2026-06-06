#!/bin/bash
#SBATCH --job-name=pring-model
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$PWD}
PRING_DIR=${PRING_DIR:-$PROJECT_DIR/../pring}
RUN_DIR=${RUN_DIR:-$PROJECT_DIR/runs/current}
MODEL_DIR=${MODEL_DIR:-$PROJECT_DIR/artifacts/models}
REPORT_DIR=${REPORT_DIR:-$PROJECT_DIR/artifacts/reports/modeling}

module purge || true
module load python/3.11 || true

python -m venv "$PROJECT_DIR/.venv-kg-admet"
source "$PROJECT_DIR/.venv-kg-admet/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$PROJECT_DIR/modeling/requirements.txt"
python -m pip install -e "$PRING_DIR"

PYTHONPATH="$PROJECT_DIR/modeling" python -m pring_modeling.train \
  --run-path "$RUN_DIR" \
  --output-dir "$MODEL_DIR" \
  --report-dir "$REPORT_DIR" \
  --target-column label
