#!/usr/bin/env bash
#SBATCH --job-name=all_models_compare
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/asmaaali/KG-ADMET-Predictor/logs/all_models_compare_%j.out
#SBATCH --error=/home/asmaaali/KG-ADMET-Predictor/logs/all_models_compare_%j.err

set -euo pipefail

echo "============================================================"
echo "PRING / KG-ADMET: Run All Models + Compare"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "============================================================"

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------

PROJECT_DIR="/home/asmaaali/KG-ADMET-Predictor"
DEFAULT_MODELING_DIR="/home/asmaaali/PRING/runs/cyp450_5enzymes_uncapped_raw_rematerialized/graph/ml/modeling"

cd "$PROJECT_DIR"

RUN_DIR="${PRING_RUN_DIR:-$DEFAULT_MODELING_DIR}"
STAGE1_DIR="$RUN_DIR/stage1_neo4j_gds_baselines"

OUT_ROOT="${MODEL_OUTPUT_DIR:-$PROJECT_DIR/models_all_stages}"
REPORT_ROOT="${MODEL_REPORT_DIR:-$PROJECT_DIR/reports/all_stages}"

mkdir -p "$PROJECT_DIR/logs" "$OUT_ROOT" "$REPORT_ROOT"

# ------------------------------------------------------------
# 2. Runtime setup
# ------------------------------------------------------------

echo "Python executable: $(which python)"
python --version

echo "Installing/updating modeling package..."
python -m pip install -e "${MODELING_PACKAGE_DIR:-./modeling}"

REQ_PYG="${MODELING_PACKAGE_DIR:-$PROJECT_DIR/modeling}/requirements-pyg-cu124.txt"
if [ -f "$REQ_PYG" ]; then
  echo "Installing PyG CUDA 12.4 sampling dependencies if needed..."
  python -m pip install -r "$REQ_PYG" || true
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "CUDA check:"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

# ------------------------------------------------------------
# 3. Run switches
# ------------------------------------------------------------

RUN_STAGE1="${RUN_STAGE1:-true}"
RUN_STAGE2="${RUN_STAGE2:-true}"
RUN_STAGE3_RGCN="${RUN_STAGE3_RGCN:-true}"
RUN_STAGE3_HGT="${RUN_STAGE3_HGT:-true}"
RUN_COMPARE="${RUN_COMPARE:-true}"
RUN_ENSEMBLE="${RUN_ENSEMBLE:-true}"
RUN_FINAL_VALIDATION="${RUN_FINAL_VALIDATION:-true}"
MODEL_SEED="${MODEL_SEED:-42}"

DEVICE="${MODEL_DEVICE:-auto}"
if [ "$DEVICE" = "auto" ] || [ -z "$DEVICE" ]; then
  DEVICE="$(python - <<'PYDEVICE'
import torch
print('cuda' if torch.cuda.is_available() else 'cpu')
PYDEVICE
)"
fi

# Stage 1
STAGE1_CLASSIFIER="${MODEL_STAGE1_CLASSIFIER:-extra_trees}"
STAGE1_CV_FOLDS="${MODEL_STAGE1_CV_FOLDS:-5}"
STAGE1_THRESHOLD_SELECTION="${MODEL_STAGE1_THRESHOLD_SELECTION:-mcc}"
STAGE1_PREDICTION_SCOPE="${MODEL_STAGE1_PREDICTION_SCOPE:-supervised}"
STAGE1_N_ESTIMATORS="${MODEL_STAGE1_N_ESTIMATORS:-1000}"
STAGE1_MIN_SAMPLES_LEAF="${MODEL_STAGE1_MIN_SAMPLES_LEAF:-5}"
STAGE1_GROUP_COLUMN="${MODEL_STAGE1_GROUP_COLUMN:-compound_node_id}"
STAGE1_RDKIT_FEATURES="${MODEL_STAGE1_RDKIT_FEATURES:-false}"
STAGE1_RDKIT_BITS="${MODEL_STAGE1_RDKIT_BITS:-0}"
MODEL_MIN_SPECIFICITY="${MODEL_MIN_SPECIFICITY:-0.50}"
MODEL_MIN_RECALL="${MODEL_MIN_RECALL:-0.0}"
MODEL_REPORT_MIN_SPECIFICITY="${MODEL_REPORT_MIN_SPECIFICITY:-0.50}"
MODEL_REPORT_HIGH_SPECIFICITY="${MODEL_REPORT_HIGH_SPECIFICITY:-0.80}"
MODEL_REPORT_MIN_RECALL="${MODEL_REPORT_MIN_RECALL:-0.80}"

# Stage 2
STAGE2_MODELS="${MODEL_STAGE2_MODELS:-complex distmult rotate}"
STAGE2_EPOCHS="${MODEL_STAGE2_EPOCHS:-150}"
KGE_DIM="${MODEL_KGE_DIM:-128}"
STAGE2_BATCH_SIZE="${MODEL_STAGE2_BATCH_SIZE:-16384}"
STAGE2_SCORE_BATCH_SIZE="${MODEL_STAGE2_SCORE_BATCH_SIZE:-65536}"
STAGE2_MAX_GRAPH_TRAIN_TRIPLES="${MODEL_MAX_GRAPH_TRAIN_TRIPLES:-0}"
STAGE2_TARGET_TRAIN_REPEAT="${MODEL_STAGE2_TARGET_TRAIN_REPEAT:-20}"
STAGE2_NEGATIVES_PER_POSITIVE="${MODEL_STAGE2_NEGATIVES_PER_POSITIVE:-5}"
STAGE2_EVAL_NEGATIVES_PER_POSITIVE="${MODEL_STAGE2_EVAL_NEGATIVES_PER_POSITIVE:-10}"
STAGE2_SUPERVISED_DECODER="${MODEL_STAGE2_SUPERVISED_DECODER:-extra_trees}"
STAGE2_CHECKPOINT_METRIC="${MODEL_STAGE2_CHECKPOINT_METRIC:-roc_auc}"

# Stage 3 R-GCN
RGCN_EPOCHS="${MODEL_RGCN_EPOCHS:-100}"
RGCN_HIDDEN_DIM="${MODEL_RGCN_HIDDEN_DIM:-128}"
RGCN_NUM_LAYERS="${MODEL_RGCN_NUM_LAYERS:-2}"
RGCN_NUM_NEIGHBORS="${MODEL_RGCN_NUM_NEIGHBORS:-15,10}"
RGCN_BATCH_SIZE="${MODEL_RGCN_BATCH_SIZE:-128}"

# Stage 3 HGT
HGT_EPOCHS="${MODEL_HGT_EPOCHS:-100}"
HGT_HIDDEN_DIM="${MODEL_HGT_HIDDEN_DIM:-64}"
HGT_NUM_LAYERS="${MODEL_HGT_NUM_LAYERS:-2}"
HGT_NUM_NEIGHBORS="${MODEL_HGT_NUM_NEIGHBORS:-10,5}"
HGT_BATCH_SIZE="${MODEL_HGT_BATCH_SIZE:-64}"
HGT_HEADS="${MODEL_HGT_HEADS:-2}"

# Shared Stage 3 imbalance/ranking settings
STAGE3_LOSS="${MODEL_STAGE3_LOSS:-weighted_bce_bpr}"
STAGE3_BPR_WEIGHT="${MODEL_BPR_WEIGHT:-0.5}"
STAGE3_CLASS_WEIGHTING="${MODEL_CLASS_WEIGHTING:-negative_ratio}"
STAGE3_THRESHOLD_SELECTION="${MODEL_THRESHOLD_SELECTION:-mcc}"
STAGE3_EARLY_STOPPING_METRIC="${MODEL_EARLY_STOPPING_METRIC:-mcc}"
STAGE3_PATIENCE="${MODEL_PATIENCE:-12}"
STAGE3_DROPOUT="${MODEL_DROPOUT:-0.2}"
STAGE3_LR="${MODEL_LR:-0.001}"

PRIMARY_COMPARE_METRIC="${MODEL_PRIMARY_COMPARE_METRIC:-mcc}"

echo "============================================================"
echo "Resolved configuration"
echo "============================================================"
echo "RUN_DIR:        $RUN_DIR"
echo "OUT_ROOT:       $OUT_ROOT"
echo "REPORT_ROOT:    $REPORT_ROOT"
echo "DEVICE:         $DEVICE"
echo "RUN_STAGE1:     $RUN_STAGE1"
echo "RUN_STAGE2:     $RUN_STAGE2"
echo "RUN_STAGE3_RGCN:$RUN_STAGE3_RGCN"
echo "RUN_STAGE3_HGT: $RUN_STAGE3_HGT"
echo "RUN_COMPARE:    $RUN_COMPARE"
echo "RUN_FINAL_VALIDATION:$RUN_FINAL_VALIDATION"
echo "MODEL_SEED:     $MODEL_SEED"
echo "STAGE2_MODELS:  $STAGE2_MODELS"
echo "============================================================"

# ------------------------------------------------------------
# 4. Helpers
# ------------------------------------------------------------

run_cmd() {
  echo "------------------------------------------------------------"
  echo "Running:"
  printf '%q ' "$@"
  printf '\n'
  echo "------------------------------------------------------------"
  "$@"
}

module_help() {
  local module="$1"
  python -m "$module" --help 2>&1 || true
}

has_flag() {
  local help_text="$1"
  local flag="$2"
  echo "$help_text" | grep -q -- "$flag"
}

# ------------------------------------------------------------
# 5. Stage 1 � leakage-safe GDS structural baseline
# ------------------------------------------------------------

run_stage1() {
  echo "============================================================"
  echo "Stage 1: leakage-safe GDS structural baseline"
  echo "============================================================"

  local train_features="$STAGE1_DIR/compound_target_training_pairs_gds_features.csv"
  local candidate_features="$STAGE1_DIR/candidate_pairs_gds_features.csv"

  if [ ! -f "$train_features" ]; then
    echo "ERROR: Missing Stage 1 generated GDS feature file:"
    echo "  $train_features"
    echo "Run stage1_export_gds_features first."
    exit 1
  fi

  echo "Found Stage 1 GDS feature files:"
  ls -lh "$train_features"
  ls -lh "$candidate_features" || true

  local out_dir="$OUT_ROOT/stage1_gds_${STAGE1_CLASSIFIER}"
  local report_dir="$REPORT_ROOT/stage1"
  mkdir -p "$out_dir" "$report_dir"

  local args=(
    python -m pring_modeling.stage1_tabular
    --modeling-dir "$RUN_DIR"
    --output-dir "$out_dir"
    --report-dir "$report_dir"
    --feature-policy leakage_safe
    --prediction-scope "$STAGE1_PREDICTION_SCOPE"
    --classifier "$STAGE1_CLASSIFIER"
    --threshold-selection "$STAGE1_THRESHOLD_SELECTION"
    --min-specificity "$MODEL_MIN_SPECIFICITY"
    --min-recall "$MODEL_MIN_RECALL"
    --report-min-specificity "$MODEL_REPORT_MIN_SPECIFICITY"
    --report-high-specificity "$MODEL_REPORT_HIGH_SPECIFICITY"
    --report-min-recall "$MODEL_REPORT_MIN_RECALL"
    --n-estimators "$STAGE1_N_ESTIMATORS"
    --min-samples-leaf "$STAGE1_MIN_SAMPLES_LEAF"
    --cv-folds "$STAGE1_CV_FOLDS"
    --group-column "$STAGE1_GROUP_COLUMN"
    --balanced-eval-max-per-class "${MODEL_BALANCED_EVAL_MAX_PER_CLASS:-0}"
    --seed "$MODEL_SEED"
  )

  if [ "$STAGE1_RDKIT_FEATURES" = "true" ]; then
    args+=(--rdkit-features --rdkit-fingerprint-bits "$STAGE1_RDKIT_BITS")
  fi

  run_cmd "${args[@]}"
}

# ------------------------------------------------------------
# 6. Stage 2 � KGE models + supervised decoder
# ------------------------------------------------------------

run_stage2_model() {
  local model="$1"

  echo "============================================================"
  echo "Stage 2: $model"
  echo "============================================================"

  local out_dir="$OUT_ROOT/stage2_${model}_supervised"
  mkdir -p "$out_dir"

  local help_text
  help_text="$(module_help pring_modeling.stage2_kge)"

  local args=(
    python -m pring_modeling.stage2_kge
    --modeling-dir "$RUN_DIR"
    --output-dir "$out_dir"
    --model "$model"
    --epochs "$STAGE2_EPOCHS"
    --dim "$KGE_DIM"
    --batch-size "$STAGE2_BATCH_SIZE"
    --max-graph-train-triples "$STAGE2_MAX_GRAPH_TRAIN_TRIPLES"
    --target-train-repeat "$STAGE2_TARGET_TRAIN_REPEAT"
    --loss softplus
    --optimizer auto
    --checkpoint-metric "$STAGE2_CHECKPOINT_METRIC"
    --negatives-per-positive "$STAGE2_NEGATIVES_PER_POSITIVE"
    --eval-negatives-per-positive "$STAGE2_EVAL_NEGATIVES_PER_POSITIVE"
    --supervised-min-specificity "$MODEL_MIN_SPECIFICITY"
    --supervised-min-recall "$MODEL_MIN_RECALL"
    --report-min-specificity "$MODEL_REPORT_MIN_SPECIFICITY"
    --report-high-specificity "$MODEL_REPORT_HIGH_SPECIFICITY"
    --report-min-recall "$MODEL_REPORT_MIN_RECALL"
    --seed "$MODEL_SEED"
    --device "$DEVICE"
  )

  if has_flag "$help_text" "--score-batch-size"; then
    args+=(--score-batch-size "$STAGE2_SCORE_BATCH_SIZE")
  fi

  if has_flag "$help_text" "--sparse-embeddings"; then
    args+=(--sparse-embeddings)
  fi

  if has_flag "$help_text" "--no-score-candidates"; then
    args+=(--no-score-candidates)
  fi

  if has_flag "$help_text" "--no-save-mappings"; then
    args+=(--no-save-mappings)
  fi

  if has_flag "$help_text" "--export-eval-predictions"; then
    args+=(--export-eval-predictions)
  fi

  if has_flag "$help_text" "--train-supervised-decoder"; then
    args+=(--train-supervised-decoder)
  fi

  if has_flag "$help_text" "--supervised-decoder"; then
    args+=(--supervised-decoder "$STAGE2_SUPERVISED_DECODER")
  fi

  run_cmd "${args[@]}"
}

run_stage2_all() {
  for model in $STAGE2_MODELS; do
    run_stage2_model "$model"
  done
}

# ------------------------------------------------------------
# 7. Stage 3 � sampled R-GCN
# ------------------------------------------------------------

run_stage3_rgcn() {
  echo "============================================================"
  echo "Stage 3: sampled R-GCN"
  echo "============================================================"

  local out_dir="$OUT_ROOT/stage3_rgcn_sampled"
  mkdir -p "$out_dir"

  local help_text
  help_text="$(module_help pring_modeling.stage3_rgcn)"

  local args=(
    python -m pring_modeling.stage3_rgcn
    --modeling-dir "$RUN_DIR"
    --output-dir "$out_dir"
    --epochs "$RGCN_EPOCHS"
    --hidden-dim "$RGCN_HIDDEN_DIM"
    --num-layers "$RGCN_NUM_LAYERS"
    --batch-size "$RGCN_BATCH_SIZE"
    --dropout "$STAGE3_DROPOUT"
    --lr "$STAGE3_LR"
    --seed "$MODEL_SEED"
    --device "$DEVICE"
  )

  if has_flag "$help_text" "--num-neighbors"; then
    args+=(--num-neighbors "$RGCN_NUM_NEIGHBORS")
  fi

  if has_flag "$help_text" "--featureless-mode"; then
    args+=(--featureless-mode type)
  fi

  if has_flag "$help_text" "--loss"; then
    args+=(--loss "$STAGE3_LOSS")
  fi

  if has_flag "$help_text" "--bpr-weight"; then
    args+=(--bpr-weight "$STAGE3_BPR_WEIGHT")
  fi

  if has_flag "$help_text" "--class-weighting"; then
    args+=(--class-weighting "$STAGE3_CLASS_WEIGHTING")
  fi

  if has_flag "$help_text" "--threshold-selection"; then
    args+=(--threshold-selection "$STAGE3_THRESHOLD_SELECTION")
  fi

  if has_flag "$help_text" "--min-specificity"; then
    args+=(--min-specificity "$MODEL_MIN_SPECIFICITY" --min-recall "$MODEL_MIN_RECALL")
    args+=(--report-min-specificity "$MODEL_REPORT_MIN_SPECIFICITY" --report-high-specificity "$MODEL_REPORT_HIGH_SPECIFICITY" --report-min-recall "$MODEL_REPORT_MIN_RECALL")
  fi

  if has_flag "$help_text" "--balanced-batches"; then
    args+=(--balanced-batches --balance-ratio "${MODEL_BALANCE_RATIO:-1.0}")
  fi

  if has_flag "$help_text" "--early-stopping-metric"; then
    args+=(--early-stopping-metric "$STAGE3_EARLY_STOPPING_METRIC")
  fi

  if has_flag "$help_text" "--patience"; then
    args+=(--patience "$STAGE3_PATIENCE")
  fi

  if has_flag "$help_text" "--no-score-candidates"; then
    args+=(--no-score-candidates)
  fi

  run_cmd "${args[@]}"
}

# ------------------------------------------------------------
# 8. Stage 3 � sampled HGT
# ------------------------------------------------------------

run_stage3_hgt() {
  echo "============================================================"
  echo "Stage 3: sampled HGT"
  echo "============================================================"

  local out_dir="$OUT_ROOT/stage3_hgt_sampled"
  mkdir -p "$out_dir"

  local help_text
  help_text="$(module_help pring_modeling.stage3_hgt)"

  local args=(
    python -m pring_modeling.stage3_hgt
    --modeling-dir "$RUN_DIR"
    --output-dir "$out_dir"
    --epochs "$HGT_EPOCHS"
    --hidden-dim "$HGT_HIDDEN_DIM"
    --num-layers "$HGT_NUM_LAYERS"
    --heads "$HGT_HEADS"
    --batch-size "$HGT_BATCH_SIZE"
    --dropout "$STAGE3_DROPOUT"
    --lr "$STAGE3_LR"
    --seed "$MODEL_SEED"
    --device "$DEVICE"
  )

  if has_flag "$help_text" "--num-neighbors"; then
    args+=(--num-neighbors "$HGT_NUM_NEIGHBORS")
  fi

  if has_flag "$help_text" "--featureless-mode"; then
    args+=(--featureless-mode type)
  fi

  if has_flag "$help_text" "--loss"; then
    args+=(--loss "$STAGE3_LOSS")
  fi

  if has_flag "$help_text" "--bpr-weight"; then
    args+=(--bpr-weight "$STAGE3_BPR_WEIGHT")
  fi

  if has_flag "$help_text" "--class-weighting"; then
    args+=(--class-weighting "$STAGE3_CLASS_WEIGHTING")
  fi

  if has_flag "$help_text" "--threshold-selection"; then
    args+=(--threshold-selection "$STAGE3_THRESHOLD_SELECTION")
  fi

  if has_flag "$help_text" "--min-specificity"; then
    args+=(--min-specificity "$MODEL_MIN_SPECIFICITY" --min-recall "$MODEL_MIN_RECALL")
    args+=(--report-min-specificity "$MODEL_REPORT_MIN_SPECIFICITY" --report-high-specificity "$MODEL_REPORT_HIGH_SPECIFICITY" --report-min-recall "$MODEL_REPORT_MIN_RECALL")
  fi

  if has_flag "$help_text" "--balanced-batches"; then
    args+=(--balanced-batches --balance-ratio "${MODEL_BALANCE_RATIO:-1.0}")
  fi

  if has_flag "$help_text" "--early-stopping-metric"; then
    args+=(--early-stopping-metric "$STAGE3_EARLY_STOPPING_METRIC")
  fi

  if has_flag "$help_text" "--patience"; then
    args+=(--patience 10)
  fi

  # HGT + pyg-lib can fail with AMP Half/Float mismatch, so disable AMP if the CLI supports it.
  if has_flag "$help_text" "--no-amp"; then
    args+=(--no-amp)
  fi

  if has_flag "$help_text" "--no-score-candidates"; then
    args+=(--no-score-candidates)
  fi

  run_cmd "${args[@]}"
}

# ------------------------------------------------------------
# 9. Stacked ensemble from labelled stage outputs
# ------------------------------------------------------------

run_ensemble() {
  echo "============================================================"
  echo "Stacked ensemble"
  echo "============================================================"

  local out_dir="$OUT_ROOT/ensemble_stacked"
  mkdir -p "$out_dir"

  run_cmd \
    python -m pring_modeling.ensemble \
    --outputs-root "$OUT_ROOT" \
    --output-dir "$out_dir" \
    --meta-classifier "${MODEL_ENSEMBLE_META_CLASSIFIER:-extra_trees}" \
    --threshold-selection "$PRIMARY_COMPARE_METRIC" \
    --min-specificity "$MODEL_MIN_SPECIFICITY" \
    --report-min-specificity "$MODEL_REPORT_MIN_SPECIFICITY" \
    --report-high-specificity "$MODEL_REPORT_HIGH_SPECIFICITY" \
    --report-min-recall "$MODEL_REPORT_MIN_RECALL" \
    --n-jobs 16
}


# ------------------------------------------------------------
# 10. Finalized V2 validation: common test, calibration, uncertainty, seeds
# ------------------------------------------------------------

run_final_validation() {
  echo "============================================================"
  echo "Finalized V2 validation"
  echo "============================================================"

  local out_dir="$OUT_ROOT/finalized_v2"
  mkdir -p "$out_dir"

  local args=(
    python -m pring_modeling.final_validation
    --outputs-root "$OUT_ROOT"
    --output-dir "$out_dir"
    --meta-classifier "${MODEL_FINAL_META_CLASSIFIER:-extra_trees}"
    --split-strategy "${MODEL_FINAL_SPLIT_STRATEGY:-compound}"
    --calibration "${MODEL_FINAL_CALIBRATION:-platt}"
    --seeds "${MODEL_FINAL_SEEDS:-$MODEL_SEED}"
    --threshold-selection "$PRIMARY_COMPARE_METRIC"
    --min-specificity "$MODEL_MIN_SPECIFICITY"
    --min-recall "$MODEL_MIN_RECALL"
    --report-min-specificity "$MODEL_REPORT_MIN_SPECIFICITY"
    --report-high-specificity "$MODEL_REPORT_HIGH_SPECIFICITY"
    --report-min-recall "$MODEL_REPORT_MIN_RECALL"
    --balanced-eval-max-per-class "${MODEL_BALANCED_EVAL_MAX_PER_CLASS:-0}"
    --top-k-per-target "${MODEL_TOP_K_PER_TARGET:-50}"
    --uncertain-top-n "${MODEL_UNCERTAIN_TOP_N:-200}"
    --per-target-min-rows "${MODEL_PER_TARGET_MIN_ROWS:-100}"
    --n-jobs 16
  )

  if [ "${MODEL_FINAL_STRICT_LEAKAGE_FREE:-false}" = "true" ]; then
    args+=(--strict-leakage-free)
  fi

  if [ -n "${MODEL_EXTERNAL_LABELS:-}" ]; then
    args+=(--external-labels "$MODEL_EXTERNAL_LABELS")
  fi

  run_cmd "${args[@]}"
}

# ------------------------------------------------------------
# 11. Compare all results
# ------------------------------------------------------------

run_compare() {
  echo "============================================================"
  echo "Compare all models"
  echo "============================================================"

  local compare_dir="$REPORT_ROOT/comparison"
  mkdir -p "$compare_dir"

  local help_text
  help_text="$(module_help pring_modeling.compare)"

  local args=(
    python -m pring_modeling.compare
    metrics
    --outputs-root "$OUT_ROOT"
    --output-dir "$compare_dir"
  )

  if has_flag "$help_text" "--primary-metric"; then
    args+=(--primary-metric "$PRIMARY_COMPARE_METRIC")
  fi

  run_cmd "${args[@]}"

  echo "Comparison outputs:"
  find "$compare_dir" -maxdepth 2 -type f -print | sort || true
}

# ------------------------------------------------------------
# 10. Execute selected stages
# ------------------------------------------------------------

if [ "$RUN_STAGE1" = "true" ]; then
  run_stage1
fi

if [ "$RUN_STAGE2" = "true" ]; then
  run_stage2_all
fi

if [ "$RUN_STAGE3_RGCN" = "true" ]; then
  run_stage3_rgcn
fi

if [ "$RUN_STAGE3_HGT" = "true" ]; then
  run_stage3_hgt
fi

if [ "$RUN_ENSEMBLE" = "true" ]; then
  run_ensemble || echo "WARNING: ensemble skipped or failed; continuing to comparison"
fi

if [ "$RUN_FINAL_VALIDATION" = "true" ]; then
  run_final_validation || echo "WARNING: finalized V2 validation skipped or failed; continuing to comparison"
fi

if [ "$RUN_COMPARE" = "true" ]; then
  run_compare
fi

echo "============================================================"
echo "All requested models finished"
echo "End time: $(date)"
echo "Outputs:"
echo "  Models:  $OUT_ROOT"
echo "  Reports: $REPORT_ROOT"
echo "============================================================"