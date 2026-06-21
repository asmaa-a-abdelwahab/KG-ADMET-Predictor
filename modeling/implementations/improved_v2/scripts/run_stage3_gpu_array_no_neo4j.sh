#!/usr/bin/env bash
#SBATCH --job-name=stage3_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7%12
#SBATCH --output=/home/asmaaali/KG-ADMET-Predictor/logs/stage3_gpu_%A_%a.out
#SBATCH --error=/home/asmaaali/KG-ADMET-Predictor/logs/stage3_gpu_%A_%a.err

set -euo pipefail

echo "============================================================"
echo "PRING / KG-ADMET Stage 3 GPU Array Job - NO NEO4J"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID:-unknown}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID:-unknown}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "============================================================"

PROJECT_DIR="/home/asmaaali/KG-ADMET-Predictor"
DEFAULT_MODELING_DIR="/home/asmaaali/PRING/runs/cyp450_5enzymes_uncapped_raw_rematerialized/graph/ml/modeling"
DEFAULT_OUTPUT_DIR="${PROJECT_DIR}/models_stage3_gpu"
DEFAULT_REPORT_DIR="${PROJECT_DIR}/reports/stage3_gpu"

cd "$PROJECT_DIR"

mkdir -p \
  "$PROJECT_DIR/logs" \
  "$DEFAULT_OUTPUT_DIR" \
  "$DEFAULT_REPORT_DIR"

echo "Python executable: $(which python)"
python --version

echo "Installing/updating modeling package..."
python -m pip install -e "${MODELING_PACKAGE_DIR:-./modeling}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_DIR="${PRING_RUN_DIR:-$DEFAULT_MODELING_DIR}"
OUT_DIR="${MODEL_OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
REPORT_DIR="${MODEL_REPORT_DIR:-$DEFAULT_REPORT_DIR}"

mkdir -p "$OUT_DIR" "$REPORT_DIR"

if [ ! -d "$RUN_DIR" ]; then
  echo "ERROR: RUN_DIR does not exist:"
  echo "  $RUN_DIR"
  exit 1
fi

echo "============================================================"
echo "Checking GPU"
echo "============================================================"

nvidia-smi || true

python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
PY

# ------------------------------------------------------------
# Stage 3 experiment grid
# Each array task runs one model/configuration on one GPU.
# Keep configs small because each GPU appears to have around 8 GB VRAM.
# ------------------------------------------------------------

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

case "$TASK_ID" in
  0)
    MODEL_NAME="rgcn"
    EPOCHS=20
    HIDDEN_DIM=32
    NUM_LAYERS=1
    BATCH_SIZE=256
    ;;
  1)
    MODEL_NAME="rgcn"
    EPOCHS=20
    HIDDEN_DIM=64
    NUM_LAYERS=1
    BATCH_SIZE=256
    ;;
  2)
    MODEL_NAME="rgcn"
    EPOCHS=30
    HIDDEN_DIM=32
    NUM_LAYERS=2
    BATCH_SIZE=256
    ;;
  3)
    MODEL_NAME="rgcn"
    EPOCHS=30
    HIDDEN_DIM=64
    NUM_LAYERS=2
    BATCH_SIZE=256
    ;;
  4)
    MODEL_NAME="hgt"
    EPOCHS=20
    HIDDEN_DIM=32
    NUM_LAYERS=1
    BATCH_SIZE=128
    ;;
  5)
    MODEL_NAME="hgt"
    EPOCHS=20
    HIDDEN_DIM=64
    NUM_LAYERS=1
    BATCH_SIZE=128
    ;;
  6)
    MODEL_NAME="hgt"
    EPOCHS=30
    HIDDEN_DIM=32
    NUM_LAYERS=2
    BATCH_SIZE=128
    ;;
  7)
    MODEL_NAME="rgcn"
    EPOCHS=50
    HIDDEN_DIM=64
    NUM_LAYERS=1
    BATCH_SIZE=512
    ;;
  *)
    echo "ERROR: Unsupported SLURM_ARRAY_TASK_ID=$TASK_ID"
    exit 2
    ;;
esac

RUN_NAME="stage3_${MODEL_NAME}_e${EPOCHS}_h${HIDDEN_DIM}_l${NUM_LAYERS}_b${BATCH_SIZE}_task${TASK_ID}"
MODEL_OUT_DIR="${OUT_DIR}/${RUN_NAME}"

mkdir -p "$MODEL_OUT_DIR"

echo "============================================================"
echo "Resolved configuration"
echo "============================================================"
echo "RUN_DIR:       $RUN_DIR"
echo "OUT_DIR:       $OUT_DIR"
echo "REPORT_DIR:    $REPORT_DIR"
echo "MODEL_NAME:    $MODEL_NAME"
echo "EPOCHS:        $EPOCHS"
echo "HIDDEN_DIM:    $HIDDEN_DIM"
echo "NUM_LAYERS:    $NUM_LAYERS"
echo "BATCH_SIZE:    $BATCH_SIZE"
echo "MODEL_OUT_DIR: $MODEL_OUT_DIR"
echo "Device:        cuda"
echo "Neo4j:         disabled"
echo "Candidate scoring: disabled during training"
echo "============================================================"

if [ "$MODEL_NAME" = "rgcn" ]; then
  CMD=(
    python -m pring_modeling.stage3_rgcn
    --modeling-dir "$RUN_DIR"
    --output-dir "$MODEL_OUT_DIR"
    --epochs "$EPOCHS"
    --hidden-dim "$HIDDEN_DIM"
    --num-layers "$NUM_LAYERS"
    --batch-size "$BATCH_SIZE"
    --device cuda
  )
elif [ "$MODEL_NAME" = "hgt" ]; then
  CMD=(
    python -m pring_modeling.stage3_hgt
    --modeling-dir "$RUN_DIR"
    --output-dir "$MODEL_OUT_DIR"
    --epochs "$EPOCHS"
    --hidden-dim "$HIDDEN_DIM"
    --num-layers "$NUM_LAYERS"
    --batch-size "$BATCH_SIZE"
    --device cuda
  )
else
  echo "ERROR: Unknown MODEL_NAME=$MODEL_NAME"
  exit 2
fi

echo "Running command:"
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "============================================================"
echo "Stage 3 GPU task finished successfully"
echo "End time: $(date)"
echo "Output: $MODEL_OUT_DIR"
echo "============================================================"
