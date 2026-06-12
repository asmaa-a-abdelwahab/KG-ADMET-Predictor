#!/usr/bin/env bash
#SBATCH --job-name=stage2_fast
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/asmaaali/KG-ADMET-Predictor/logs/stage2_fast_%j.out
#SBATCH --error=/home/asmaaali/KG-ADMET-Predictor/logs/stage2_fast_%j.err

set -euo pipefail

echo "============================================================"
echo "PRING / KG-ADMET Stage 2 Efficient KGE Job - NO NEO4J"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "============================================================"

PROJECT_DIR="/home/asmaaali/KG-ADMET-Predictor"
DEFAULT_MODELING_DIR="/home/asmaaali/PRING/runs/cyp450_5enzymes_uncapped_raw_rematerialized/graph/ml/modeling"
DEFAULT_OUTPUT_DIR="${PROJECT_DIR}/models_stage2_fast"
DEFAULT_REPORT_DIR="${PROJECT_DIR}/reports/stage2_fast"

cd "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs" "$DEFAULT_OUTPUT_DIR" "$DEFAULT_REPORT_DIR"

echo "Python executable: $(which python)"
python --version

echo "Installing/updating modeling package..."
python -m pip install -e "./modeling"

echo "Torch/GPU status:"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
nvidia-smi || true

# Keep CPU math libraries from over-subscribing the node.
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

KGE_MODEL="${MODEL_KGE_MODEL:-rotate}"              # distmult | complex | rotate
EPOCHS="${MODEL_STAGE2_EPOCHS:-40}"
DIM="${MODEL_KGE_DIM:-64}"                         # 64 is safer on 8 GB GPU; use 128 on CPU or larger GPU
BATCH_SIZE="${MODEL_BATCH_SIZE:-16384}"
SCORE_BATCH_SIZE="${MODEL_SCORE_BATCH_SIZE:-262144}"
MAX_GRAPH_TRAIN_TRIPLES="${MODEL_MAX_GRAPH_TRAIN_TRIPLES:-1000000}"
TARGET_REPEAT="${MODEL_STAGE2_TARGET_TRAIN_REPEAT:-5}"
LOSS="${MODEL_STAGE2_LOSS:-softplus}"
OPTIMIZER="${MODEL_STAGE2_OPTIMIZER:-auto}"
NEG_PER_POS="${MODEL_STAGE2_NEGATIVES_PER_POSITIVE:-1}"
EVAL_NEG_PER_POS="${MODEL_STAGE2_EVAL_NEGATIVES_PER_POSITIVE:-1}"
EVAL_EVERY="${MODEL_STAGE2_EVAL_EVERY:-1}"
PATIENCE="${MODEL_STAGE2_PATIENCE:-5}"
CHECKPOINT_METRIC="${MODEL_STAGE2_CHECKPOINT_METRIC:-average_precision}"
DEVICE="${MODEL_DEVICE:-auto}"
NUM_WORKERS="${MODEL_NUM_WORKERS:-0}"
SPARSE_EMBEDDINGS="${MODEL_STAGE2_SPARSE_EMBEDDINGS:-true}"

# Candidate scoring is disabled by default because full scoring of 3.75M pairs can be run separately
# after the best Stage 2 model is selected.
SCORE_CANDIDATES="${MODEL_STAGE2_SCORE_CANDIDATES:-false}"
SCORE_ONLY="${MODEL_STAGE2_SCORE_ONLY:-false}"
LOAD_MODEL="${MODEL_STAGE2_LOAD_MODEL:-}"
MAX_CANDIDATE_TRIPLES="${MODEL_MAX_CANDIDATE_TRIPLES:-100000}"
PREDICTION_TOP_K="${MODEL_PREDICTION_TOP_K:-0}"
SAVE_MAPPINGS="${MODEL_STAGE2_SAVE_MAPPINGS:-false}"
ATTACH_ENTITY_REFS="${MODEL_STAGE2_ATTACH_ENTITY_REFS:-false}"

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

RUN_NAME="stage2_${KGE_MODEL}_fast_e${EPOCHS}_d${DIM}_g${MAX_GRAPH_TRAIN_TRIPLES}_b${BATCH_SIZE}_${DEVICE}"
MODEL_OUT_DIR="$OUT_DIR/$RUN_NAME"
mkdir -p "$MODEL_OUT_DIR" "$REPORT_DIR"

echo "============================================================"
echo "Resolved Stage 2 efficient configuration"
echo "============================================================"
echo "RUN_DIR:                 $RUN_DIR"
echo "MODEL_OUT_DIR:           $MODEL_OUT_DIR"
echo "KGE_MODEL:               $KGE_MODEL"
echo "EPOCHS:                  $EPOCHS"
echo "DIM:                     $DIM"
echo "BATCH_SIZE:              $BATCH_SIZE"
echo "SCORE_BATCH_SIZE:        $SCORE_BATCH_SIZE"
echo "MAX_GRAPH_TRAIN_TRIPLES: $MAX_GRAPH_TRAIN_TRIPLES"
echo "TARGET_REPEAT:           $TARGET_REPEAT"
echo "LOSS:                    $LOSS"
echo "OPTIMIZER:               $OPTIMIZER"
echo "NEG_PER_POS:             $NEG_PER_POS"
echo "EVAL_NEG_PER_POS:        $EVAL_NEG_PER_POS"
echo "EVAL_EVERY:              $EVAL_EVERY"
echo "PATIENCE:                $PATIENCE"
echo "CHECKPOINT_METRIC:       $CHECKPOINT_METRIC"
echo "DEVICE:                  $DEVICE"
echo "NUM_WORKERS:             $NUM_WORKERS"
echo "SPARSE_EMBEDDINGS:       $SPARSE_EMBEDDINGS"
echo "SCORE_CANDIDATES:        $SCORE_CANDIDATES"
echo "SCORE_ONLY:              $SCORE_ONLY"
echo "LOAD_MODEL:              ${LOAD_MODEL:-<none>}"
echo "MAX_CANDIDATE_TRIPLES:   $MAX_CANDIDATE_TRIPLES"
echo "PREDICTION_TOP_K:        $PREDICTION_TOP_K"
echo "SAVE_MAPPINGS:           $SAVE_MAPPINGS"
echo "ATTACH_ENTITY_REFS:      $ATTACH_ENTITY_REFS"
echo "Neo4j:                   disabled"
echo "============================================================"

if [ ! -d "$RUN_DIR" ]; then
  echo "ERROR: RUN_DIR does not exist: $RUN_DIR" >&2
  exit 1
fi

ARGS=(
  python -m pring_modeling.stage2_kge
  --modeling-dir "$RUN_DIR"
  --output-dir "$MODEL_OUT_DIR"
  --model "$KGE_MODEL"
  --epochs "$EPOCHS"
  --dim "$DIM"
  --batch-size "$BATCH_SIZE"
  --score-batch-size "$SCORE_BATCH_SIZE"
  --max-graph-train-triples "$MAX_GRAPH_TRAIN_TRIPLES"
  --target-train-repeat "$TARGET_REPEAT"
  --loss "$LOSS"
  --optimizer "$OPTIMIZER"
  --negatives-per-positive "$NEG_PER_POS"
  --eval-negatives-per-positive "$EVAL_NEG_PER_POS"
  --eval-every "$EVAL_EVERY"
  --patience "$PATIENCE"
  --checkpoint-metric "$CHECKPOINT_METRIC"
  --num-workers "$NUM_WORKERS"
  --max-candidate-triples "$MAX_CANDIDATE_TRIPLES"
  --prediction-top-k "$PREDICTION_TOP_K"
  --device "$DEVICE"
)

if [ "$SPARSE_EMBEDDINGS" = "true" ]; then ARGS+=(--sparse-embeddings); else ARGS+=(--no-sparse-embeddings); fi
if [ "$SCORE_CANDIDATES" = "true" ]; then ARGS+=(--score-candidates); else ARGS+=(--no-score-candidates); fi
if [ "$SCORE_ONLY" = "true" ]; then
  ARGS+=(--score-only)
  if [ -z "$LOAD_MODEL" ]; then
    echo "ERROR: MODEL_STAGE2_SCORE_ONLY=true requires MODEL_STAGE2_LOAD_MODEL=/path/to/best_model.pt" >&2
    exit 2
  fi
  ARGS+=(--load-model "$LOAD_MODEL")
else
  ARGS+=(--no-score-only)
fi
if [ "$SAVE_MAPPINGS" = "true" ]; then ARGS+=(--save-mappings); else ARGS+=(--no-save-mappings); fi
if [ "$ATTACH_ENTITY_REFS" = "true" ]; then ARGS+=(--attach-entity-refs); else ARGS+=(--no-attach-entity-refs); fi

echo "Running command:"
printf '%q ' "${ARGS[@]}"
printf '\n'
"${ARGS[@]}"

echo "============================================================"
echo "Stage 2 efficient KGE finished"
echo "End time: $(date)"
echo "Output: $MODEL_OUT_DIR"
echo "============================================================"
