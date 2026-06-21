#!/usr/bin/env bash
#SBATCH --job-name=stage1_gds_baseline
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/asmaaali/KG-ADMET-Predictor/logs/stage1_gds_baseline_%j.out
#SBATCH --error=/home/asmaaali/KG-ADMET-Predictor/logs/stage1_gds_baseline_%j.err

set -euo pipefail

echo "============================================================"
echo "PRING / KG-ADMET Stage 1 GDS Structural Baseline"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "============================================================"

PROJECT_DIR="/home/asmaaali/KG-ADMET-Predictor"
DEFAULT_MODELING_DIR="/home/asmaaali/PRING/runs/cyp450_5enzymes_uncapped_raw_rematerialized/graph/ml/modeling"

cd "$PROJECT_DIR"

mkdir -p "$PROJECT_DIR/logs"

echo "Python executable: $(which python)"
python --version

echo "Installing/updating modeling package..."
python -m pip install -e "${MODELING_PACKAGE_DIR:-./modeling}"

RUN_DIR="${PRING_RUN_DIR:-$DEFAULT_MODELING_DIR}"

STAGE1_DIR="$RUN_DIR/stage1_neo4j_gds_baselines"

OUT_DIR="${MODEL_OUTPUT_DIR:-$PROJECT_DIR/models_v16/stage1_gds_structural}"
REPORT_DIR="${MODEL_REPORT_DIR:-$PROJECT_DIR/reports/v16/stage1}"

FEATURE_POLICY="${MODEL_FEATURE_POLICY:-leakage_safe}"
CLASSIFIER="${MODEL_CLASSIFIER:-extra_trees}"
PREDICTION_SCOPE="${MODEL_PREDICTION_SCOPE:-supervised}"
THRESHOLD_SELECTION="${MODEL_THRESHOLD_SELECTION:-mcc}"
CV_FOLDS="${MODEL_CV_FOLDS:-5}"

TRAIN_FEATURE_FILE="$STAGE1_DIR/compound_target_training_pairs_gds_features.csv"
CANDIDATE_FEATURE_FILE="$STAGE1_DIR/candidate_pairs_gds_features.csv"
EXPORT_SUMMARY_FILE="$STAGE1_DIR/stage1_gds_feature_export_summary.json"

mkdir -p "$OUT_DIR" "$REPORT_DIR"

echo "============================================================"
echo "Resolved Stage 1 configuration"
echo "============================================================"
echo "PROJECT_DIR:          $PROJECT_DIR"
echo "RUN_DIR:              $RUN_DIR"
echo "STAGE1_DIR:           $STAGE1_DIR"
echo "OUT_DIR:              $OUT_DIR"
echo "REPORT_DIR:           $REPORT_DIR"
echo "FEATURE_POLICY:       $FEATURE_POLICY"
echo "CLASSIFIER:           $CLASSIFIER"
echo "PREDICTION_SCOPE:     $PREDICTION_SCOPE"
echo "THRESHOLD_SELECTION:  $THRESHOLD_SELECTION"
echo "CV_FOLDS:             $CV_FOLDS"
echo "TRAIN_FEATURE_FILE:   $TRAIN_FEATURE_FILE"
echo "CANDIDATE_FEATURE:    $CANDIDATE_FEATURE_FILE"
echo "============================================================"

if [ ! -d "$RUN_DIR" ]; then
  echo "ERROR: RUN_DIR does not exist:"
  echo "  $RUN_DIR"
  exit 1
fi

if [ ! -d "$STAGE1_DIR" ]; then
  echo "ERROR: Stage 1 directory does not exist:"
  echo "  $STAGE1_DIR"
  exit 1
fi

if [ ! -f "$TRAIN_FEATURE_FILE" ]; then
  echo "ERROR: Generated Stage 1 GDS training feature file was not found:"
  echo "  $TRAIN_FEATURE_FILE"
  echo
  echo "You need to run stage1_export_gds_features first."
  exit 1
fi

if [ "$PREDICTION_SCOPE" = "candidates" ] && [ ! -f "$CANDIDATE_FEATURE_FILE" ]; then
  echo "ERROR: MODEL_PREDICTION_SCOPE=candidates but candidate feature file was not found:"
  echo "  $CANDIDATE_FEATURE_FILE"
  echo
  echo "Either run stage1_export_gds_features with --include-candidates,"
  echo "or set MODEL_PREDICTION_SCOPE=supervised."
  exit 1
fi

echo "Preview generated feature files:"
ls -lh "$TRAIN_FEATURE_FILE" || true
ls -lh "$CANDIDATE_FEATURE_FILE" || true
ls -lh "$EXPORT_SUMMARY_FILE" || true

echo "Training feature columns preview:"
python - <<PY
import pandas as pd
p = "$TRAIN_FEATURE_FILE"
df = pd.read_csv(p, nrows=3)
print("Rows preview:", len(df))
print("Number of columns:", len(df.columns))
print("Columns:")
for c in df.columns:
    print("  -", c)
PY

ARGS=(
  python -m pring_modeling.stage1_tabular
  --modeling-dir "$RUN_DIR"
  --output-dir "$OUT_DIR"
  --report-dir "$REPORT_DIR"
  --feature-policy "$FEATURE_POLICY"
  --prediction-scope "$PREDICTION_SCOPE"
  --classifier "$CLASSIFIER"
  --threshold-selection "$THRESHOLD_SELECTION"
  --cv-folds "$CV_FOLDS"
)

echo "============================================================"
echo "Running Stage 1 GDS structural baseline"
echo "============================================================"
printf 'Running: '
printf '%q ' "${ARGS[@]}"
printf '\n'

"${ARGS[@]}"

echo "============================================================"
echo "Stage 1 GDS structural baseline finished"
echo "End time: $(date)"
echo "Outputs:"
echo "  Models:  $OUT_DIR"
echo "  Reports: $REPORT_DIR"
echo "============================================================"