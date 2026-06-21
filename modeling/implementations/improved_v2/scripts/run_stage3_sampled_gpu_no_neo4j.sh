#!/bin/bash
#SBATCH --job-name=stage3_sampled_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/asmaaali/KG-ADMET-Predictor/logs/stage3_sampled_gpu_%j.out
#SBATCH --error=/home/asmaaali/KG-ADMET-Predictor/logs/stage3_sampled_gpu_%j.err

if [ -z "${BASH_VERSION:-}" ]; then
  exec /bin/bash "$0" "$@"
fi

set -euo pipefail

echo "============================================================"
echo "PRING / KG-ADMET Stage 3 Sampled GPU Job - NO NEO4J"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "============================================================"

PROJECT_DIR="/home/asmaaali/KG-ADMET-Predictor"
DEFAULT_MODELING_DIR="/home/asmaaali/PRING/runs/cyp450_5enzymes_uncapped_raw_rematerialized/graph/ml/modeling"
DEFAULT_OUTPUT_DIR="${PROJECT_DIR}/models_stage3_sampled_gpu"
DEFAULT_REPORT_DIR="${PROJECT_DIR}/reports/stage3_sampled_gpu"

cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs" "$DEFAULT_OUTPUT_DIR" "$DEFAULT_REPORT_DIR"

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

# Options: rgcn, hgt
MODEL_NAME="${MODEL_STAGE3_MODEL:-rgcn}"
EPOCHS="${MODEL_STAGE3_EPOCHS:-60}"
HIDDEN_DIM="${MODEL_HIDDEN_DIM:-64}"
NUM_LAYERS="${MODEL_NUM_LAYERS:-1}"
NUM_NEIGHBORS="${MODEL_NUM_NEIGHBORS:-15}"
BATCH_SIZE="${MODEL_BATCH_SIZE:-128}"
DEVICE="${MODEL_STAGE3_DEVICE:-${MODEL_DEVICE:-cuda}}"
FEATURELESS_MODE="${MODEL_FEATURELESS_MODE:-type}"
HEADS="${MODEL_HGT_HEADS:-1}"
# R-GCN can use AMP, but HGTConv/pyg-lib grouped_matmul may fail with
# Float/Half mismatch. Default HGT to FP32 unless explicitly overridden.
DEFAULT_AMP="true"
if [ "$MODEL_NAME" = "hgt" ]; then
  DEFAULT_AMP="false"
fi
AMP_FLAG="${MODEL_AMP:-$DEFAULT_AMP}"
SCORE_CANDIDATES="${MODEL_SCORE_CANDIDATES:-false}"
MAX_CANDIDATE_PAIRS="${MODEL_MAX_CANDIDATE_PAIRS:-100000}"

# Imbalance-aware training controls.
LOSS="${MODEL_LOSS:-weighted_bce}"                       # bce, weighted_bce, focal
CLASS_WEIGHTING="${MODEL_CLASS_WEIGHTING:-balanced}"     # none, balanced, negative_ratio
BALANCED_BATCHES="${MODEL_BALANCED_BATCHES:-true}"       # true/false
BALANCE_RATIO="${MODEL_BALANCE_RATIO:-1.0}"              # target negative:positive ratio when oversampling
NEGATIVE_CLASS_WEIGHT="${MODEL_NEGATIVE_CLASS_WEIGHT:-}" # optional manual label-0 weight
POSITIVE_CLASS_WEIGHT="${MODEL_POSITIVE_CLASS_WEIGHT:-}" # optional manual label-1 weight
FOCAL_GAMMA="${MODEL_FOCAL_GAMMA:-2.0}"
FOCAL_ALPHA="${MODEL_FOCAL_ALPHA:--1.0}"
BPR_WEIGHT="${MODEL_BPR_WEIGHT:-0.5}"
THRESHOLD="${MODEL_THRESHOLD:-0.5}"
THRESHOLD_SELECTION="${MODEL_THRESHOLD_SELECTION:-mcc}"  # fixed, mcc, balanced_accuracy, f1
EARLY_STOPPING_METRIC="${MODEL_EARLY_STOPPING_METRIC:-mcc}"
PATIENCE="${MODEL_PATIENCE:-10}"
MIN_DELTA="${MODEL_MIN_DELTA:-0.0001}"
LR="${MODEL_LR:-0.001}"
DROPOUT="${MODEL_DROPOUT:-0.2}"
WEIGHT_DECAY="${MODEL_WEIGHT_DECAY:-0.0001}"
GRAD_CLIP="${MODEL_GRAD_CLIP:-1.0}"

mkdir -p "$OUT_DIR" "$REPORT_DIR"

if [ ! -d "$RUN_DIR" ]; then
  echo "ERROR: RUN_DIR does not exist: $RUN_DIR" >&2
  exit 1
fi

nvidia-smi || true
python - <<'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
try:
    import pyg_lib
    print('pyg_lib: OK')
except Exception as e:
    print('pyg_lib: missing', e)
try:
    import torch_sparse
    print('torch_sparse: OK')
except Exception as e:
    print('torch_sparse: missing', e)
PY

RUN_NAME="stage3_${MODEL_NAME}_sampled_e${EPOCHS}_h${HIDDEN_DIM}_l${NUM_LAYERS}_n${NUM_NEIGHBORS}_b${BATCH_SIZE}_${LOSS}_${CLASS_WEIGHTING}_${THRESHOLD_SELECTION}"
MODEL_OUT_DIR="${OUT_DIR}/${RUN_NAME}"
mkdir -p "$MODEL_OUT_DIR"

echo "============================================================"
echo "Resolved Stage 3 sampled configuration"
echo "============================================================"
echo "RUN_DIR:                $RUN_DIR"
echo "MODEL_OUT_DIR:          $MODEL_OUT_DIR"
echo "REPORT_DIR:             $REPORT_DIR"
echo "MODEL_NAME:             $MODEL_NAME"
echo "EPOCHS:                 $EPOCHS"
echo "HIDDEN_DIM:             $HIDDEN_DIM"
echo "NUM_LAYERS:             $NUM_LAYERS"
echo "NUM_NEIGHBORS:          $NUM_NEIGHBORS"
echo "BATCH_SIZE:             $BATCH_SIZE"
echo "DEVICE:                 $DEVICE"
echo "FEATURELESS_MODE:       $FEATURELESS_MODE"
echo "AMP_FLAG:               $AMP_FLAG"
echo "LOSS:                   $LOSS"
echo "BPR_WEIGHT:             $BPR_WEIGHT"
echo "CLASS_WEIGHTING:        $CLASS_WEIGHTING"
echo "BALANCED_BATCHES:       $BALANCED_BATCHES"
echo "BALANCE_RATIO:          $BALANCE_RATIO"
echo "THRESHOLD_SELECTION:    $THRESHOLD_SELECTION"
echo "EARLY_STOPPING_METRIC:  $EARLY_STOPPING_METRIC"
echo "SCORE_CANDIDATES:       $SCORE_CANDIDATES"
echo "Neo4j:                  disabled"
echo "============================================================"

COMMON_ARGS=(
  --modeling-dir "$RUN_DIR"
  --output-dir "$MODEL_OUT_DIR"
  --epochs "$EPOCHS"
  --hidden-dim "$HIDDEN_DIM"
  --num-layers "$NUM_LAYERS"
  --num-neighbors "$NUM_NEIGHBORS"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
  --featureless-mode "$FEATURELESS_MODE"
  --loss "$LOSS"
  --class-weighting "$CLASS_WEIGHTING"
  --balance-ratio "$BALANCE_RATIO"
  --focal-gamma "$FOCAL_GAMMA"
  --focal-alpha "$FOCAL_ALPHA"
  --bpr-weight "$BPR_WEIGHT"
  --threshold "$THRESHOLD"
  --threshold-selection "$THRESHOLD_SELECTION"
  --early-stopping-metric "$EARLY_STOPPING_METRIC"
  --patience "$PATIENCE"
  --min-delta "$MIN_DELTA"
  --lr "$LR"
  --dropout "$DROPOUT"
  --weight-decay "$WEIGHT_DECAY"
  --grad-clip "$GRAD_CLIP"
)

if [ "$BALANCED_BATCHES" = "true" ]; then
  COMMON_ARGS+=(--balanced-batches)
fi

if [ -n "$NEGATIVE_CLASS_WEIGHT" ]; then
  COMMON_ARGS+=(--negative-class-weight "$NEGATIVE_CLASS_WEIGHT")
fi

if [ -n "$POSITIVE_CLASS_WEIGHT" ]; then
  COMMON_ARGS+=(--positive-class-weight "$POSITIVE_CLASS_WEIGHT")
fi

if [ "$AMP_FLAG" = "true" ]; then
  COMMON_ARGS+=(--amp)
fi

if [ "$SCORE_CANDIDATES" = "true" ]; then
  COMMON_ARGS+=(--score-candidates --max-candidate-pairs "$MAX_CANDIDATE_PAIRS")
fi

case "$MODEL_NAME" in
  rgcn)
    CMD=(python -m pring_modeling.stage3_rgcn "${COMMON_ARGS[@]}")
    ;;
  hgt)
    CMD=(python -m pring_modeling.stage3_hgt "${COMMON_ARGS[@]}" --heads "$HEADS")
    ;;
  *)
    echo "ERROR: MODEL_STAGE3_MODEL must be rgcn or hgt, got: $MODEL_NAME" >&2
    exit 2
    ;;
esac

echo "Running command:"
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "============================================================"
echo "Stage 3 sampled GPU job finished successfully"
echo "End time: $(date)"
echo "Output: $MODEL_OUT_DIR"
echo "============================================================"
