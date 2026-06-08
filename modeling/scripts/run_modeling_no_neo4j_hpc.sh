#!/usr/bin/env bash
#SBATCH --job-name=modeling_no_neo4j
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=12:00:00
#SBATCH --output=/home/asmaaali/KG-ADMET-Predictor/logs/no_neo4j_%j.out
#SBATCH --error=/home/asmaaali/KG-ADMET-Predictor/logs/no_neo4j_%j.err

set -euo pipefail

echo "============================================================"
echo "PRING / KG-ADMET Modeling Slurm Job - NO NEO4J VERSION"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "============================================================"

# ------------------------------------------------------------
# 1. Project paths on HPC
# ------------------------------------------------------------

PROJECT_DIR="/home/asmaaali/KG-ADMET-Predictor"

DEFAULT_MODELING_DIR="/home/asmaaali/PRING/runs/cyp450_5enzymes_uncapped_raw_rematerialized/graph/ml/modeling"
DEFAULT_OUTPUT_DIR="${PROJECT_DIR}/models_no_neo4j"
DEFAULT_REPORT_DIR="${PROJECT_DIR}/reports/modeling_no_neo4j"

cd "$PROJECT_DIR"

mkdir -p \
  "$PROJECT_DIR/logs" \
  "$DEFAULT_OUTPUT_DIR" \
  "$DEFAULT_REPORT_DIR"

# ------------------------------------------------------------
# 2. Install/update modeling package
# ------------------------------------------------------------

echo "Python executable: $(which python)"
python --version

echo "Installing/updating modeling package..."
python -m pip install -e "./modeling"

# ------------------------------------------------------------
# 3. Runtime/threading settings
# ------------------------------------------------------------

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"

echo "Thread settings:"
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "  MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "  OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
echo "  NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS}"
echo "  TORCH_NUM_THREADS=${TORCH_NUM_THREADS}"
echo "  TORCH_NUM_INTEROP_THREADS=${TORCH_NUM_INTEROP_THREADS}"

# ------------------------------------------------------------
# 4. Main configurable variables
# ------------------------------------------------------------

RUN_DIR="${PRING_RUN_DIR:-$DEFAULT_MODELING_DIR}"
OUT_DIR="${MODEL_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
REPORT_DIR="${MODEL_REPORT_DIR:-$DEFAULT_REPORT_DIR}"

# Options:
#   run_all
#   stage1
#   stage2
#   stage3
#   hgt
#   compare
#
# Stage4 is intentionally disabled here because the current stage4_explain
# workflow needs Neo4j for evidence path extraction.
STAGE="${MODEL_STAGE:-run_all}"

mkdir -p "$OUT_DIR" "$REPORT_DIR"

echo "============================================================"
echo "Resolved configuration"
echo "============================================================"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "RUN_DIR:     $RUN_DIR"
echo "OUT_DIR:     $OUT_DIR"
echo "REPORT_DIR:  $REPORT_DIR"
echo "STAGE:       $STAGE"
echo "Neo4j:       disabled"
echo "============================================================"

if [ ! -d "$RUN_DIR" ]; then
  echo "ERROR: RUN_DIR does not exist:"
  echo "  $RUN_DIR"
  exit 1
fi

# ------------------------------------------------------------
# 5. Helper function
# ------------------------------------------------------------

run_cmd() {
  echo "------------------------------------------------------------"
  echo "Running command:"
  printf '%q ' "$@"
  printf '\n'
  echo "------------------------------------------------------------"
  "$@"
}

# ------------------------------------------------------------
# 6. Stage commands
# ------------------------------------------------------------

run_stage1() {
  run_cmd \
    python -m pring_modeling.stage1_tabular \
      --modeling-dir "$RUN_DIR" \
      --output-dir "$OUT_DIR/stage1_tabular" \
      --report-dir "$REPORT_DIR" \
      --target-column "${MODEL_TARGET_COLUMN:-label}" \
      --threshold "${MODEL_THRESHOLD:-0.5}" \
      --n-estimators "${MODEL_N_ESTIMATORS:-300}" \
      --n-jobs "${MODEL_N_JOBS:-1}" \
      --max-training-rows "${MODEL_MAX_TRAINING_ROWS:-1000000}" \
      --max-scoring-rows "${MODEL_MAX_SCORING_ROWS:-3755472}" \
      --max-predictions-file-rows "${MODEL_MAX_PREDICTIONS_FILE_ROWS:-3755472}"
}

run_stage2_one_model() {
  local model_name="$1"

  run_cmd \
    python -m pring_modeling.stage2_kge \
      --modeling-dir "$RUN_DIR" \
      --output-dir "$OUT_DIR/stage2_${model_name}" \
      --model "$model_name" \
      --epochs "${MODEL_STAGE2_EPOCHS:-30}" \
      --dim "${MODEL_KGE_DIM:-128}" \
      --batch-size "${MODEL_BATCH_SIZE:-4096}" \
      --score-batch-size "${MODEL_SCORE_BATCH_SIZE:-4096}" \
      --max-graph-train-triples "${MODEL_MAX_GRAPH_TRAIN_TRIPLES:-5716825}" \
      --max-candidate-triples "${MODEL_MAX_CANDIDATE_TRIPLES:-3755472}" \
      --device "${MODEL_DEVICE:-cpu}"
}

run_stage2_all_models() {
  IFS=' ' read -r -a KGE_MODELS <<< "${MODEL_STAGE2_MODELS:-distmult complex rotate}"

  for model_name in "${KGE_MODELS[@]}"; do
    echo "Running Stage 2 KGE model: ${model_name}"
    run_stage2_one_model "$model_name"
  done
}

run_stage3_rgcn() {
  ARGS=(
    python -m pring_modeling.stage3_rgcn
    --modeling-dir "$RUN_DIR"
    --output-dir "$OUT_DIR/stage3_rgcn"
    --epochs "${MODEL_STAGE3_EPOCHS:-50}"
    --hidden-dim "${MODEL_HIDDEN_DIM:-128}"
    --num-layers "${MODEL_NUM_LAYERS:-2}"
    --batch-size "${MODEL_BATCH_SIZE:-4096}"
    --device "${MODEL_DEVICE:-cpu}"
  )

  if [ "${MODEL_SCORE_CANDIDATES:-true}" = "true" ]; then
    ARGS+=(
      --score-candidates
      --max-candidate-pairs "${MODEL_MAX_CANDIDATE_PAIRS:-3794225}"
    )
  fi

  run_cmd "${ARGS[@]}"
}

run_stage3_hgt() {
  ARGS=(
    python -m pring_modeling.stage3_hgt
    --modeling-dir "$RUN_DIR"
    --output-dir "$OUT_DIR/stage3_hgt"
    --epochs "${MODEL_STAGE3_EPOCHS:-50}"
    --hidden-dim "${MODEL_HIDDEN_DIM:-128}"
    --num-layers "${MODEL_NUM_LAYERS:-2}"
    --batch-size "${MODEL_BATCH_SIZE:-4096}"
    --device "${MODEL_DEVICE:-cpu}"
  )

  if [ "${MODEL_SCORE_CANDIDATES:-true}" = "true" ]; then
    ARGS+=(
      --score-candidates
      --max-candidate-pairs "${MODEL_MAX_CANDIDATE_PAIRS:-3794225}"
    )
  fi

  run_cmd "${ARGS[@]}"
}

run_compare() {
  run_cmd \
    python -m pring_modeling.compare \
      metrics \
      --outputs-root "$OUT_DIR" \
      --output-dir "$REPORT_DIR/comparison"
}

# ------------------------------------------------------------
# 7. Run selected workflow
# ------------------------------------------------------------

case "$STAGE" in

  run_all|all)
    echo "Running all modeling stages without Neo4j export..."

    run_stage1
    run_stage2_all_models
    run_stage3_rgcn
    run_stage3_hgt
    run_compare
    ;;

  stage1|stage1_tabular|tabular)
    run_stage1
    ;;

  stage2|kge)
    # If MODEL_KGE_MODEL is set, run one model.
    # Otherwise run all Stage 2 models.
    if [ -n "${MODEL_KGE_MODEL:-}" ]; then
      run_stage2_one_model "$MODEL_KGE_MODEL"
    else
      run_stage2_all_models
    fi
    ;;

  stage3|rgcn)
    run_stage3_rgcn
    ;;

  hgt|stage3_hgt)
    run_stage3_hgt
    ;;

  compare)
    run_compare
    ;;

  stage4|explain)
    echo "ERROR: Stage 4 is disabled in this no-Neo4j script."
    echo "Reason: current stage4_explain requires Neo4j for evidence path extraction."
    echo "Use the normal Neo4j-enabled script for Stage 4."
    exit 2
    ;;

  *)
    echo "ERROR: Unknown MODEL_STAGE=$STAGE"
    echo "Allowed values for no-Neo4j script:"
    echo "  run_all"
    echo "  stage1"
    echo "  stage2"
    echo "  stage3"
    echo "  hgt"
    echo "  compare"
    exit 2
    ;;

esac

echo "============================================================"
echo "Modeling job finished successfully - NO NEO4J VERSION"
echo "End time: $(date)"
echo "Outputs:"
echo "  Models:  $OUT_DIR"
echo "  Reports: $REPORT_DIR"
echo "============================================================"