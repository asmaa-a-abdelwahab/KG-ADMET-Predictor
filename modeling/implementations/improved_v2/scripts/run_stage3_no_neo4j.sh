#!/bin/bash
#SBATCH --job-name=stage3_sampled
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/asmaaali/PRING-APP/logs/stage3_sampled_%j.out
#SBATCH --error=/home/asmaaali/PRING-APP/logs/stage3_sampled_%j.err

if [ -z "${BASH_VERSION:-}" ]; then
  exec /bin/bash "$0" "$@"
fi

set -euo pipefail

echo "============================================================"
echo "PRING-APP Stage 3 Sampled GNN Job - NO NEO4J"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "============================================================"

# ------------------------------------------------------------
# 1. Project paths on HPC
# ------------------------------------------------------------

PROJECT_DIR="/home/asmaaali/PRING-APP"

DEFAULT_MODELING_DIR="/home/asmaaali/PRING-PACKAGE/runs/cyp450_5enzymes_uncapped_raw_rematerialized/graph/ml/modeling"
DEFAULT_OUTPUT_DIR="${PROJECT_DIR}/models_stage3_sampled"
DEFAULT_REPORT_DIR="${PROJECT_DIR}/reports/stage3_sampled"

cd "$PROJECT_DIR"

mkdir -p \
  "$PROJECT_DIR/logs" \
  "$DEFAULT_OUTPUT_DIR" \
  "$DEFAULT_REPORT_DIR"

# ------------------------------------------------------------
# 2. Python environment and package installation
# ------------------------------------------------------------

echo "Python executable before install: $(which python)"
python --version

echo "Installing/updating modeling package..."
python -m pip install -e "${MODELING_PACKAGE_DIR:-./modeling}"

echo "Testing Stage 3 imports..."
python - <<'PY'
import sys
print("Python:", sys.version)

import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

import pring_modeling
print("pring_modeling import OK")

try:
    import torch_geometric
    print("PyG:", torch_geometric.__version__)
except Exception as e:
    print("WARNING: Could not import torch_geometric:", e)

try:
    import pyg_lib
    print("pyg-lib import OK")
except Exception as e:
    print("WARNING: pyg-lib not available:", e)

try:
    import torch_sparse
    print("torch-sparse import OK")
except Exception as e:
    print("WARNING: torch-sparse not available:", e)
PY

# ------------------------------------------------------------
# 3. Runtime/threading settings
# ------------------------------------------------------------

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"

# Helps reduce CUDA memory fragmentation on small GPUs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Thread settings:"
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "  MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "  OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
echo "  NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS}"
echo "  TORCH_NUM_THREADS=${TORCH_NUM_THREADS}"
echo "  TORCH_NUM_INTEROP_THREADS=${TORCH_NUM_INTEROP_THREADS}"
echo "  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

# ------------------------------------------------------------
# 4. Stage 3 configurable variables
# ------------------------------------------------------------

RUN_DIR="${PRING_RUN_DIR:-$DEFAULT_MODELING_DIR}"
OUT_DIR="${MODEL_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
REPORT_DIR="${MODEL_REPORT_DIR:-$DEFAULT_REPORT_DIR}"

# Options:
#   rgcn
#   hgt
#   both
STAGE3_MODEL="${MODEL_STAGE3_MODEL:-rgcn}"

# Safer defaults for 8 GB GPUs.
EPOCHS="${MODEL_STAGE3_EPOCHS:-20}"
HIDDEN_DIM="${MODEL_HIDDEN_DIM:-32}"
NUM_LAYERS="${MODEL_NUM_LAYERS:-1}"
BATCH_SIZE="${MODEL_BATCH_SIZE:-128}"
DEVICE="${MODEL_STAGE3_DEVICE:-${MODEL_DEVICE:-auto}}"

# New sampled-training settings.
NUM_NEIGHBORS="${MODEL_NUM_NEIGHBORS:-5}"
FEATURELESS_MODE="${MODEL_FEATURELESS_MODE:-type}"
HGT_HEADS="${MODEL_HGT_HEADS:-1}"

# Candidate scoring is intentionally disabled by default.
# Enable later after training is stable.
SCORE_CANDIDATES="${MODEL_SCORE_CANDIDATES:-false}"
MAX_CANDIDATE_PAIRS="${MODEL_MAX_CANDIDATE_PAIRS:-100000}"

if [ "$DEVICE" = "auto" ]; then
  if python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    DEVICE="cuda"
  else
    DEVICE="cpu"
  fi
fi

mkdir -p \
  "$OUT_DIR/stage3_rgcn_sampled" \
  "$OUT_DIR/stage3_hgt_sampled" \
  "$REPORT_DIR"

echo "============================================================"
echo "Resolved Stage 3 sampled configuration"
echo "============================================================"
echo "PROJECT_DIR:         $PROJECT_DIR"
echo "RUN_DIR:             $RUN_DIR"
echo "OUT_DIR:             $OUT_DIR"
echo "REPORT_DIR:          $REPORT_DIR"
echo "STAGE3_MODEL:        $STAGE3_MODEL"
echo "EPOCHS:              $EPOCHS"
echo "HIDDEN_DIM:          $HIDDEN_DIM"
echo "NUM_LAYERS:          $NUM_LAYERS"
echo "BATCH_SIZE:          $BATCH_SIZE"
echo "DEVICE:              $DEVICE"
echo "NUM_NEIGHBORS:       $NUM_NEIGHBORS"
echo "FEATURELESS_MODE:    $FEATURELESS_MODE"
echo "HGT_HEADS:           $HGT_HEADS"
echo "SCORE_CANDIDATES:    $SCORE_CANDIDATES"
echo "MAX_CANDIDATE_PAIRS: $MAX_CANDIDATE_PAIRS"
echo "Neo4j:               disabled"
echo "============================================================"

if [ ! -d "$RUN_DIR" ]; then
  echo "ERROR: RUN_DIR does not exist:"
  echo "  $RUN_DIR"
  exit 1
fi

if [ ! -f "$RUN_DIR/stage3_heterogeneous_gnn/pyg_export/heterodata.pt" ] && \
   [ ! -f "$RUN_DIR/stage3_heterogeneous_gnn/pyg_export/heterodata_payload.pt" ]; then
  echo "WARNING: Could not find expected Stage 3 PyG export files:"
  echo "  $RUN_DIR/stage3_heterogeneous_gnn/pyg_export/heterodata.pt"
  echo "  $RUN_DIR/stage3_heterogeneous_gnn/pyg_export/heterodata_payload.pt"
  echo "The script will continue, but Stage 3 may fail if the loader requires these files."
fi

echo "============================================================"
echo "GPU status before training"
echo "============================================================"
nvidia-smi || true

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
# 6. Run sampled Stage 3 R-GCN
# ------------------------------------------------------------

run_stage3_rgcn() {
  local RUN_NAME="stage3_rgcn_sampled_e${EPOCHS}_h${HIDDEN_DIM}_l${NUM_LAYERS}_n${NUM_NEIGHBORS}_b${BATCH_SIZE}"
  local MODEL_OUT_DIR="$OUT_DIR/$RUN_NAME"

  mkdir -p "$MODEL_OUT_DIR"

  ARGS=(
    python -m pring_modeling.stage3_rgcn
    --modeling-dir "$RUN_DIR"
    --output-dir "$MODEL_OUT_DIR"
    --epochs "$EPOCHS"
    --hidden-dim "$HIDDEN_DIM"
    --num-layers "$NUM_LAYERS"
    --batch-size "$BATCH_SIZE"
    --device "$DEVICE"
    --num-neighbors "$NUM_NEIGHBORS"
    --featureless-mode "$FEATURELESS_MODE"
  )

  if [ "$SCORE_CANDIDATES" = "true" ]; then
    ARGS+=(
      --score-candidates
      --max-candidate-pairs "$MAX_CANDIDATE_PAIRS"
    )
  fi

  run_cmd "${ARGS[@]}"
}

# ------------------------------------------------------------
# 7. Run sampled Stage 3 HGT
# ------------------------------------------------------------

run_stage3_hgt() {
  local RUN_NAME="stage3_hgt_sampled_e${EPOCHS}_h${HIDDEN_DIM}_l${NUM_LAYERS}_heads${HGT_HEADS}_n${NUM_NEIGHBORS}_b${BATCH_SIZE}"
  local MODEL_OUT_DIR="$OUT_DIR/$RUN_NAME"

  mkdir -p "$MODEL_OUT_DIR"

  ARGS=(
    python -m pring_modeling.stage3_hgt
    --modeling-dir "$RUN_DIR"
    --output-dir "$MODEL_OUT_DIR"
    --epochs "$EPOCHS"
    --hidden-dim "$HIDDEN_DIM"
    --num-layers "$NUM_LAYERS"
    --batch-size "$BATCH_SIZE"
    --device "$DEVICE"
    --num-neighbors "$NUM_NEIGHBORS"
    --featureless-mode "$FEATURELESS_MODE"
    --heads "$HGT_HEADS"
  )

  if [ "$SCORE_CANDIDATES" = "true" ]; then
    ARGS+=(
      --score-candidates
      --max-candidate-pairs "$MAX_CANDIDATE_PAIRS"
    )
  fi

  run_cmd "${ARGS[@]}"
}

# ------------------------------------------------------------
# 8. Run selected Stage 3 model(s)
# ------------------------------------------------------------

case "$STAGE3_MODEL" in

  rgcn)
    echo "Running sampled Stage 3 R-GCN only..."
    run_stage3_rgcn
    ;;

  hgt)
    echo "Running sampled Stage 3 HGT only..."
    run_stage3_hgt
    ;;

  both|all)
    echo "Running sampled Stage 3 R-GCN and HGT..."
    run_stage3_rgcn
    run_stage3_hgt
    ;;

  *)
    echo "ERROR: Unknown MODEL_STAGE3_MODEL=$STAGE3_MODEL"
    echo "Allowed values:"
    echo "  rgcn"
    echo "  hgt"
    echo "  both"
    exit 2
    ;;

esac

echo "============================================================"
echo "Stage 3 sampled modeling finished successfully - NO NEO4J"
echo "End time: $(date)"
echo "Outputs:"
echo "  Models:  $OUT_DIR"
echo "  Reports: $REPORT_DIR"
echo "============================================================"
