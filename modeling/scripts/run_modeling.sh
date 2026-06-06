#!/usr/bin/env bash
set -euo pipefail

/opt/kg/bin/install_pring_runtime.sh

RUN_DIR="${PRING_RUN_DIR:-/runs/current}"
OUT_DIR="${MODEL_OUTPUT_DIR:-/models}"
REPORT_DIR="${MODEL_REPORT_DIR:-/reports/modeling}"
mkdir -p "$OUT_DIR" "$REPORT_DIR"

ARGS=(python -m pring_modeling.train
  --run-path "$RUN_DIR"
  --output-dir "$OUT_DIR"
  --report-dir "$REPORT_DIR"
  --target-column "${MODEL_TARGET_COLUMN:-label}"
  --threshold "${MODEL_THRESHOLD:-0.5}"
  --n-estimators "${MODEL_N_ESTIMATORS:-100}"
  --n-jobs "${MODEL_N_JOBS:-1}"
  --min-samples-leaf "${MODEL_MIN_SAMPLES_LEAF:-2}"
  --max-training-rows "${MODEL_MAX_TRAINING_ROWS:-100000}"
  --max-scoring-rows "${MODEL_MAX_SCORING_ROWS:-100000}"
  --max-predictions-file-rows "${MODEL_MAX_PREDICTIONS_FILE_ROWS:-100000}"
  --prediction-scope "${MODEL_PREDICTION_SCOPE:-candidates}"
  --max-node-feature-columns "${MODEL_MAX_NODE_FEATURE_COLUMNS:-128}"
  --max-neo4j-predictions "${MODEL_MAX_NEO4J_PREDICTIONS:-25000}"
)

if [ -n "${MODEL_MAX_DEPTH:-}" ]; then
  ARGS+=(--max-depth "${MODEL_MAX_DEPTH}")
fi

if [ "${MODEL_USE_NODE_FEATURES:-false}" = "true" ]; then
  ARGS+=(--use-node-features)
fi

if [ "${MODEL_EXPORT_TO_NEO4J:-true}" = "true" ]; then
  ARGS+=(--export-neo4j)
fi

echo "Running modeling pipeline from PRING run: $RUN_DIR"
printf 'Command: '; printf '%q ' "${ARGS[@]}"; printf '\n'
"${ARGS[@]}"
