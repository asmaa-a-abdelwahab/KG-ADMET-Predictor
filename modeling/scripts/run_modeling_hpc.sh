#!/usr/bin/env bash
#SBATCH --job-name=modeling
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=12:00:00
#SBATCH --output=/home/asmaaali/KG-ADMET-Predictor/logs/all_stages_%j.out
#SBATCH --error=/home/asmaaali/KG-ADMET-Predictor/logs/all_stages_%j.err

set -euo pipefail

echo "============================================================"
echo "PRING / KG-ADMET Modeling Slurm Job"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "============================================================"

# ------------------------------------------------------------
# 1. Project paths on HPC
# ------------------------------------------------------------

PROJECT_DIR="/home/asmaaali/KG-ADMET-Predictor"

DEFAULT_MODELING_DIR="/home/asmaaali/PRING/runs/cyp450_5enzymes_uncapped_raw_rematerialized/graph/ml/modeling"
DEFAULT_OUTPUT_DIR="${PROJECT_DIR}/models"
DEFAULT_REPORT_DIR="${PROJECT_DIR}/reports/modeling"

cd "$PROJECT_DIR"

mkdir -p \
  "$PROJECT_DIR/logs" \
  "$DEFAULT_OUTPUT_DIR" \
  "$DEFAULT_REPORT_DIR"

# ------------------------------------------------------------
# 2. Activate Python environment
# ------------------------------------------------------------
# IMPORTANT:
# Use Python >= 3.10.
# Replace this path if your Python 3.10/3.11 environment has another name.

if [ -f "${PROJECT_DIR}/pring-py310-env/bin/activate" ]; then
  source "${PROJECT_DIR}/pring-py310-env/bin/activate"
elif [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
  source "${PROJECT_DIR}/.venv/bin/activate"
else
  echo "ERROR: No Python environment found."
  echo "Expected one of:"
  echo "  ${PROJECT_DIR}/pring-py310-env/bin/activate"
  echo "  ${PROJECT_DIR}/.venv/bin/activate"
  echo
  echo "Create one with:"
  echo "  cd ${PROJECT_DIR}"
  echo "  module load python/3.10"
  echo "  python -m venv pring-py310-env"
  echo "  source pring-py310-env/bin/activate"
  echo "  python -m pip install --upgrade pip setuptools wheel"
  echo "  python -m pip install -e './modeling[notebooks]'"
  exit 1
fi

echo "Python executable: $(which python)"
python --version

PYTHON_CHECK=$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
if sys.version_info < (3, 10):
    raise SystemExit(1)
PY
) || {
  echo "ERROR: Python >= 3.10 is required."
  echo "Current Python:"
  python --version
  exit 1
}

echo "Python version check passed: ${PYTHON_CHECK}"

# ------------------------------------------------------------
# 3. Install package in editable mode if available
# ------------------------------------------------------------

echo "Installing/updating modeling package..."
python -m pip install -e "./modeling"

# ------------------------------------------------------------
# 4. Runtime/threading settings
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
# 5. Main configurable variables
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
#   stage4
#   compare
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
echo "============================================================"

if [ ! -d "$RUN_DIR" ]; then
  echo "ERROR: RUN_DIR does not exist:"
  echo "  $RUN_DIR"
  exit 1
fi

# ------------------------------------------------------------
# 6. Neo4j configuration
# ------------------------------------------------------------
# These are only used if Neo4j export or path-based explanation is enabled.
# For HPC, this will work only if the compute node can reach Neo4j.

export NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-cyp450kg}"
export NEO4J_DATABASE="${NEO4J_DATABASE:-neo4j}"

echo "Neo4j URI: ${NEO4J_URI}"
echo "Neo4j database: ${NEO4J_DATABASE}"

# ------------------------------------------------------------
# 7. Build command by stage
# ------------------------------------------------------------

case "$STAGE" in

  run_all|all)
    ARGS=(
      python -m pring_modeling.run_all
      --modeling-dir "$RUN_DIR"
      --output-dir "$OUT_DIR"
      --report-dir "$REPORT_DIR"
    )
    ;;

  stage1|stage1_tabular|tabular)
    ARGS=(
      python -m pring_modeling.stage1_tabular
      --modeling-dir "$RUN_DIR"
      --output-dir "$OUT_DIR/stage1_tabular"
      --report-dir "$REPORT_DIR"
      --target-column "${MODEL_TARGET_COLUMN:-label}"
      --threshold "${MODEL_THRESHOLD:-0.5}"
      --n-estimators "${MODEL_N_ESTIMATORS:-300}"
      --n-jobs "${MODEL_N_JOBS:-1}"
      --max-training-rows "${MODEL_MAX_TRAINING_ROWS:-1000000}"
      --max-scoring-rows "${MODEL_MAX_SCORING_ROWS:-3755472}"
      --max-predictions-file-rows "${MODEL_MAX_PREDICTIONS_FILE_ROWS:-3755472}"
    )
    ;;

  stage2|kge)
    ARGS=(
      python -m pring_modeling.stage2_kge
      --modeling-dir "$RUN_DIR"
      --output-dir "$OUT_DIR/stage2_${MODEL_KGE_MODEL:-rotate}"
      --model "${MODEL_KGE_MODEL:-rotate}"
      --epochs "${MODEL_STAGE2_EPOCHS:-30}"
      --dim "${MODEL_KGE_DIM:-128}"
      --batch-size "${MODEL_BATCH_SIZE:-4096}"
      --score-batch-size "${MODEL_SCORE_BATCH_SIZE:-4096}"
      --max-graph-train-triples "${MODEL_MAX_GRAPH_TRAIN_TRIPLES:-5716825}"
      --max-candidate-triples "${MODEL_MAX_CANDIDATE_TRIPLES:-3755472}"
      --device "${MODEL_DEVICE:-cpu}"
    )
    ;;

  stage3|rgcn)
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
    ;;

  hgt|stage3_hgt)
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
    ;;

  stage4|explain)
    if [ -z "${MODEL_PREDICTIONS_CSV:-}" ]; then
      echo "ERROR: MODEL_PREDICTIONS_CSV is required for MODEL_STAGE=stage4"
      echo "Example:"
      echo "  MODEL_STAGE=stage4 MODEL_PREDICTIONS_CSV=/path/to/predictions.csv sbatch modeling/scripts/run_modeling.sh"
      exit 2
    fi

    ARGS=(
      python -m pring_modeling.stage4_explain
      --predictions "$MODEL_PREDICTIONS_CSV"
      --neo4j-uri "$NEO4J_URI"
      --neo4j-user "$NEO4J_USER"
      --neo4j-password "$NEO4J_PASSWORD"
      --database "$NEO4J_DATABASE"
      --output-dir "$REPORT_DIR/stage4_explanations"
    )
    ;;

  compare)
    ARGS=(
      python -m pring_modeling.compare
      metrics
      --outputs-root "$OUT_DIR"
      --output-dir "$REPORT_DIR/comparison"
    )
    ;;

  *)
    echo "ERROR: Unknown MODEL_STAGE=$STAGE"
    echo "Allowed values:"
    echo "  run_all"
    echo "  stage1"
    echo "  stage2"
    echo "  stage3"
    echo "  hgt"
    echo "  stage4"
    echo "  compare"
    exit 2
    ;;

esac

# ------------------------------------------------------------
# 8. Optional flags
# ------------------------------------------------------------

# Export predictions back to Neo4j for individual stages.
# For run_all, the run_all module should control best-model export internally.
if [ "${MODEL_EXPORT_TO_NEO4J:-false}" = "true" ] && \
   [[ "$STAGE" != "run_all" && "$STAGE" != "all" && "$STAGE" != "stage4" && "$STAGE" != "explain" && "$STAGE" != "compare" ]]; then
  ARGS+=(
    --export-neo4j
    --max-neo4j-predictions "${MODEL_MAX_NEO4J_PREDICTIONS:-25000}"
  )
fi

# Score Stage 3 candidate pairs.
if [ "${MODEL_SCORE_CANDIDATES:-true}" = "true" ] && \
   [[ "$STAGE" == "stage3" || "$STAGE" == "rgcn" || "$STAGE" == "hgt" || "$STAGE" == "stage3_hgt" ]]; then
  ARGS+=(
    --score-candidates
    --max-candidate-pairs "${MODEL_MAX_CANDIDATE_PAIRS:-3794225}"
  )
fi

# Optional evidence path depth for Stage 1 if supported by installed code.
if [ -n "${MODEL_MAX_DEPTH:-}" ] && \
   [[ "$STAGE" == "stage1" || "$STAGE" == "stage1_tabular" || "$STAGE" == "tabular" ]]; then
  ARGS+=(
    --max-depth "$MODEL_MAX_DEPTH"
  )
fi

# ------------------------------------------------------------
# 9. Run
# ------------------------------------------------------------

echo "============================================================"
echo "Running modeling stage: $STAGE"
echo "Command:"
printf '%q ' "${ARGS[@]}"
printf '\n'
echo "============================================================"

"${ARGS[@]}"

echo "============================================================"
echo "Modeling job finished successfully"
echo "End time: $(date)"
echo "============================================================"