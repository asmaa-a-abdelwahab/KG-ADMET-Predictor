#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /bin/bash "$0" "$@"
fi

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"

/opt/kg/bin/install_pring_runtime.sh || true

RUN_DIR="${PRING_RUN_DIR:-/runs/current}"
OUT_DIR="${MODEL_OUTPUT_DIR:-/models}"
REPORT_DIR="${MODEL_REPORT_DIR:-/reports/modeling}"
RUN_MODE="${MODEL_STAGE:-run_all}"
AUTO_TRAIN="${MODEL_AUTO_TRAIN:-false}"
KEEP_ALIVE="${MODEL_KEEP_ALIVE:-true}"
mkdir -p "$OUT_DIR" "$REPORT_DIR"

print_usage() {
  cat <<EOF
PRING modeling container is ready.

Set MODEL_AUTO_TRAIN=true to train immediately when the container starts.
Current settings:
  PRING_RUN_DIR=$RUN_DIR
  MODEL_OUTPUT_DIR=$OUT_DIR
  MODEL_REPORT_DIR=$REPORT_DIR
  MODEL_STAGE=$RUN_MODE

Manual examples inside the container:
  python -m pring_modeling.run_all --modeling-dir "$RUN_DIR" --output-dir "$OUT_DIR" --report-dir "$REPORT_DIR"
  python -m pring_modeling.stage2_kge --modeling-dir "$RUN_DIR" --output-dir "$OUT_DIR/stage2_rotate" --model rotate --epochs 1 --dim 32
EOF
}

if [[ "$AUTO_TRAIN" != "true" && "$AUTO_TRAIN" != "1" && "$AUTO_TRAIN" != "yes" ]]; then
  print_usage
  if [[ "$KEEP_ALIVE" == "true" || "$KEEP_ALIVE" == "1" || "$KEEP_ALIVE" == "yes" ]]; then
    echo "MODEL_AUTO_TRAIN is false. Keeping container alive for manual commands."
    tail -f /dev/null
  else
    echo "MODEL_AUTO_TRAIN is false and MODEL_KEEP_ALIVE is false. Exiting."
  fi
  exit 0
fi

case "$RUN_MODE" in
  run_all|all)
    ARGS=(python -m pring_modeling.run_all
      --modeling-dir "$RUN_DIR"
      --output-dir "$OUT_DIR"
      --report-dir "$REPORT_DIR")
    ;;
  stage1|stage1_tabular|tabular)
    ARGS=(python -m pring_modeling.stage1_tabular
      --modeling-dir "$RUN_DIR"
      --output-dir "$OUT_DIR/stage1_${MODEL_STAGE1_CLASSIFIER:-random_forest}"
      --report-dir "$REPORT_DIR"
      --target-column "${MODEL_TARGET_COLUMN:-label}"
      --threshold "${MODEL_THRESHOLD:-0.5}"
      --classifier "${MODEL_STAGE1_CLASSIFIER:-random_forest}"
      --n-estimators "${MODEL_N_ESTIMATORS:-300}"
      --n-jobs "${MODEL_N_JOBS:-1}"
      --min-samples-leaf "${MODEL_MIN_SAMPLES_LEAF:-2}"
      --max-training-rows "${MODEL_MAX_TRAINING_ROWS:-0}"
      --max-scoring-rows "${MODEL_MAX_SCORING_ROWS:-0}"
      --max-predictions-file-rows "${MODEL_MAX_PREDICTIONS_FILE_ROWS:-0}"
      --prediction-scope "${MODEL_PREDICTION_SCOPE:-candidates}")
    ;;
  stage2|kge)
    ARGS=(python -m pring_modeling.stage2_kge
      --modeling-dir "$RUN_DIR"
      --output-dir "$OUT_DIR/stage2_${MODEL_KGE_MODEL:-rotate}"
      --model "${MODEL_KGE_MODEL:-rotate}"
      --epochs "${MODEL_STAGE2_EPOCHS:-30}"
      --dim "${MODEL_KGE_DIM:-128}"
      --batch-size "${MODEL_BATCH_SIZE:-16384}"
      --score-batch-size "${MODEL_SCORE_BATCH_SIZE:-262144}"
      --max-graph-train-triples "${MODEL_MAX_GRAPH_TRAIN_TRIPLES:-1000000}"
      --max-candidate-triples "${MODEL_MAX_CANDIDATE_TRIPLES:-100000}"
      --target-train-repeat "${MODEL_STAGE2_TARGET_TRAIN_REPEAT:-5}"
      --loss "${MODEL_STAGE2_LOSS:-softplus}"
      --optimizer "${MODEL_STAGE2_OPTIMIZER:-auto}"
      --negatives-per-positive "${MODEL_STAGE2_NEGATIVES_PER_POSITIVE:-1}"
      --eval-negatives-per-positive "${MODEL_STAGE2_EVAL_NEGATIVES_PER_POSITIVE:-1}"
      --eval-every "${MODEL_STAGE2_EVAL_EVERY:-1}"
      --patience "${MODEL_STAGE2_PATIENCE:-5}"
      --checkpoint-metric "${MODEL_STAGE2_CHECKPOINT_METRIC:-average_precision}"
      --num-workers "${MODEL_NUM_WORKERS:-0}"
      --device "${MODEL_STAGE2_DEVICE:-${MODEL_DEVICE:-auto}}")
    if [ "${MODEL_STAGE2_SPARSE_EMBEDDINGS:-true}" = "true" ]; then ARGS+=(--sparse-embeddings); else ARGS+=(--no-sparse-embeddings); fi
    if [ "${MODEL_STAGE2_SCORE_CANDIDATES:-false}" = "true" ]; then ARGS+=(--score-candidates); else ARGS+=(--no-score-candidates); fi
    if [ "${MODEL_STAGE2_SAVE_MAPPINGS:-false}" = "true" ]; then ARGS+=(--save-mappings); else ARGS+=(--no-save-mappings); fi
    if [ "${MODEL_STAGE2_ATTACH_ENTITY_REFS:-false}" = "true" ]; then ARGS+=(--attach-entity-refs); else ARGS+=(--no-attach-entity-refs); fi
    ;;
  stage3|rgcn)
    ARGS=(python -m pring_modeling.stage3_rgcn
      --modeling-dir "$RUN_DIR"
      --output-dir "$OUT_DIR/stage3_rgcn"
      --epochs "${MODEL_STAGE3_EPOCHS:-50}"
      --hidden-dim "${MODEL_HIDDEN_DIM:-128}"
      --num-layers "${MODEL_NUM_LAYERS:-2}"
      --batch-size "${MODEL_BATCH_SIZE:-4096}"
      --score-batch-size "${MODEL_SCORE_BATCH_SIZE:-65536}"
      --device "${MODEL_STAGE3_DEVICE:-${MODEL_DEVICE:-cpu}}")
    ;;
  hgt)
    ARGS=(python -m pring_modeling.stage3_hgt
      --modeling-dir "$RUN_DIR"
      --output-dir "$OUT_DIR/stage3_hgt"
      --epochs "${MODEL_STAGE3_EPOCHS:-50}"
      --hidden-dim "${MODEL_HIDDEN_DIM:-128}"
      --num-layers "${MODEL_NUM_LAYERS:-2}"
      --heads "${MODEL_HGT_HEADS:-2}"
      --batch-size "${MODEL_BATCH_SIZE:-4096}"
      --score-batch-size "${MODEL_SCORE_BATCH_SIZE:-65536}"
      --device "${MODEL_STAGE3_DEVICE:-${MODEL_DEVICE:-cpu}}")
    ;;
  stage4|explain)
    if [ -z "${MODEL_PREDICTIONS_CSV:-}" ]; then echo "MODEL_PREDICTIONS_CSV is required for stage4/explain" >&2; exit 2; fi
    ARGS=(python -m pring_modeling.stage4_explain
      --predictions "$MODEL_PREDICTIONS_CSV"
      --neo4j-uri "${NEO4J_URI:-bolt://neo4j:7687}"
      --neo4j-user "${NEO4J_USER:-neo4j}"
      --neo4j-password "${NEO4J_PASSWORD:-cyp450kg}"
      --database "${NEO4J_DATABASE:-neo4j}"
      --output-dir "$REPORT_DIR/stage4_explanations")
    ;;
  compare)
    ARGS=(python -m pring_modeling.compare metrics --outputs-root "$OUT_DIR" --output-dir "$REPORT_DIR/comparison")
    ;;
  *)
    echo "Unknown MODEL_STAGE=$RUN_MODE" >&2; exit 2;;
esac

if [ "${MODEL_EXPORT_TO_NEO4J:-true}" = "true" ] && [[ "$RUN_MODE" != "run_all" && "$RUN_MODE" != "all" && "$RUN_MODE" != "stage4" && "$RUN_MODE" != "explain" && "$RUN_MODE" != "compare" ]]; then
  ARGS+=(--export-neo4j --max-neo4j-predictions "${MODEL_MAX_NEO4J_PREDICTIONS:-25000}")
fi
if [ "${MODEL_SCORE_CANDIDATES:-true}" = "true" ] && [[ "$RUN_MODE" == "stage3" || "$RUN_MODE" == "rgcn" || "$RUN_MODE" == "hgt" ]]; then
  ARGS+=(--score-candidates --max-candidate-pairs "${MODEL_MAX_CANDIDATE_PAIRS:-0}")
fi
if [ -n "${MODEL_MAX_DEPTH:-}" ] && [[ "$RUN_MODE" == "stage1" || "$RUN_MODE" == "stage1_tabular" || "$RUN_MODE" == "tabular" ]]; then
  ARGS+=(--max-depth "$MODEL_MAX_DEPTH")
fi

echo "Running PRING modeling mode: $RUN_MODE"
printf 'Command: '; printf '%q ' "${ARGS[@]}"; printf '\n'
"${ARGS[@]}"
