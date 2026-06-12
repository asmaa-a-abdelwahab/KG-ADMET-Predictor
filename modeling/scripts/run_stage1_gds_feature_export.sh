#!/usr/bin/env bash
#SBATCH --job-name=stage1_gds_export
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/asmaaali/KG-ADMET-Predictor/logs/stage1_gds_export_%j.out
#SBATCH --error=/home/asmaaali/KG-ADMET-Predictor/logs/stage1_gds_export_%j.err

set -euo pipefail

PROJECT_DIR="/home/asmaaali/KG-ADMET-Predictor"
DEFAULT_MODELING_DIR="/home/asmaaali/PRING/runs/cyp450_5enzymes_uncapped_raw_rematerialized/graph/ml/modeling"

cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs"
python -m pip install -e "./modeling"

RUN_DIR="${PRING_RUN_DIR:-$DEFAULT_MODELING_DIR}"
MAX_CANDIDATE_ROWS="${MODEL_MAX_CANDIDATE_ROWS:-100000}"
INCLUDE_CANDIDATES="${MODEL_INCLUDE_CANDIDATES:-true}"

ARGS=(
  python -m pring_modeling.stage1_export_gds_features
  --modeling-dir "$RUN_DIR"
  --max-candidate-rows "$MAX_CANDIDATE_ROWS"
)

if [ "$INCLUDE_CANDIDATES" = "true" ]; then
  ARGS+=(--include-candidates)
fi

printf 'Running: '; printf '%q ' "${ARGS[@]}"; printf '\n'
"${ARGS[@]}"
