#!/bin/bash
#SBATCH --job-name=final_validation
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=/home/asmaaali/PRING-APP/logs/final_validation_%j.out
#SBATCH --error=/home/asmaaali/PRING-APP/logs/final_validation_%j.err

if [ -z "${BASH_VERSION:-}" ]; then
  exec /bin/bash "$0" "$@"
fi

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-/home/asmaaali/PRING-APP}"
cd "$PROJECT_DIR"
OUT_ROOT="${MODEL_OUTPUT_DIR:-$PROJECT_DIR/models_all_stages_improved_v2}"
REPORT_ROOT="${MODEL_REPORT_DIR:-$PROJECT_DIR/reports/all_stages_improved_v2}"
mkdir -p "$PROJECT_DIR/logs" "$OUT_ROOT" "$REPORT_ROOT"
python -m pip install -e "${MODELING_PACKAGE_DIR:-$PROJECT_DIR/modeling}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

ARGS=(
  python -m pring_modeling.final_validation
  --outputs-root "$OUT_ROOT"
  --output-dir "$OUT_ROOT/finalized_v2"
  --meta-classifier "${MODEL_FINAL_META_CLASSIFIER:-fixed_mean}"
  --split-strategy "${MODEL_FINAL_SPLIT_STRATEGY:-registered}"
  --calibration "${MODEL_FINAL_CALIBRATION:-platt}"
  --seeds "${MODEL_FINAL_SEEDS:-1 2 3 4 5}"
  --threshold-selection "${MODEL_PRIMARY_COMPARE_METRIC:-mcc}"
  --min-specificity "${MODEL_MIN_SPECIFICITY:-0.50}"
  --min-recall "${MODEL_MIN_RECALL:-0.0}"
  --report-min-specificity "${MODEL_REPORT_MIN_SPECIFICITY:-0.50}"
  --report-high-specificity "${MODEL_REPORT_HIGH_SPECIFICITY:-0.80}"
  --report-min-recall "${MODEL_REPORT_MIN_RECALL:-0.80}"
  --balanced-eval-max-per-class "${MODEL_BALANCED_EVAL_MAX_PER_CLASS:-0}"
  --bootstrap-resamples "${MODEL_FINAL_BOOTSTRAP_RESAMPLES:-1000}"
  --top-k-per-target "${MODEL_TOP_K_PER_TARGET:-50}"
  --uncertain-top-n "${MODEL_UNCERTAIN_TOP_N:-200}"
  --per-target-min-rows "${MODEL_PER_TARGET_MIN_ROWS:-100}"
  --n-jobs 16
)
if [ "${MODEL_FINAL_STRICT_LEAKAGE_FREE:-true}" = "true" ]; then
  ARGS+=(--strict-leakage-free)
fi
if [ -n "${MODEL_EXTERNAL_LABELS:-}" ]; then
  ARGS+=(--external-labels "$MODEL_EXTERNAL_LABELS")
fi
if [ -n "${MODEL_PROVENANCE_MANIFEST:-}" ]; then
  ARGS+=(--provenance-manifest "$MODEL_PROVENANCE_MANIFEST")
fi
printf 'Running:'; printf ' %q' "${ARGS[@]}"; printf '\n'
"${ARGS[@]}"

mkdir -p "$REPORT_ROOT/finalized_v2"
cp -f "$OUT_ROOT/finalized_v2/metrics.json" "$REPORT_ROOT/finalized_v2/metrics.json" || true
