#!/usr/bin/env bash
#SBATCH --job-name=modeling
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=12:00:00
#SBATCH --output=logs/all_stages_%j.out
#SBATCH --error=logs/all_stages_%j.err


set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"

/opt/kg/bin/install_pring_runtime.sh || true

RUN_DIR="${PRING_RUN_DIR:-/runs/current}"
OUT_DIR="${MODEL_OUTPUT_DIR:-/models}"
REPORT_DIR="${MODEL_REPORT_DIR:-/reports/modeling}"
STAGE="${MODEL_STAGE:-run_all}"
mkdir -p "$OUT_DIR" "$REPORT_DIR"

case "$STAGE" in
  run_all|all)
    ARGS=(python -m pring_modeling.run_all
      --modeling-dir "$RUN_DIR"
      --output-dir "$OUT_DIR"
      --report-dir "$REPORT_DIR")
    ;;
  stage1|stage1_tabular|tabular)
    ARGS=(python -m pring_modeling.stage1_tabular
      --modeling-dir "$RUN_DIR"
      --output-dir "$OUT_DIR/stage1_tabular"
      --report-dir "$REPORT_DIR"
      --target-column "${MODEL_TARGET_COLUMN:-label}"
      --threshold "${MODEL_THRESHOLD:-0.5}"
      --n-estimators "${MODEL_N_ESTIMATORS:-200}"
      --n-jobs "${MODEL_N_JOBS:-1}"
      --max-training-rows "${MODEL_MAX_TRAINING_ROWS:-100000}"
      --max-scoring-rows "${MODEL_MAX_SCORING_ROWS:-100000}"
      --max-predictions-file-rows "${MODEL_MAX_PREDICTIONS_FILE_ROWS:-100000}")
    ;;
  stage1_mlp|gds_mlp)
    if [ -z "${MODEL_EMBEDDING_CSV:-}" ]; then echo "MODEL_EMBEDDING_CSV is required for stage1_mlp" >&2; exit 2; fi
    ARGS=(python -m pring_modeling.stage1_mlp_gds
      --modeling-dir "$RUN_DIR"
      --embedding-csv "$MODEL_EMBEDDING_CSV"
      --output-dir "$OUT_DIR/stage1_mlp")
    ;;
  stage2|kge)
    ARGS=(python -m pring_modeling.stage2_kge
      --modeling-dir "$RUN_DIR"
      --output-dir "$OUT_DIR/stage2_${MODEL_KGE_MODEL:-rotate}"
      --model "${MODEL_KGE_MODEL:-rotate}"
      --epochs "${MODEL_STAGE2_EPOCHS:-20}"
      --dim "${MODEL_KGE_DIM:-64}"
      --batch-size "${MODEL_BATCH_SIZE:-4096}"
      --max-graph-train-triples "${MODEL_MAX_GRAPH_TRAIN_TRIPLES:-500000}"
      --max-candidate-triples "${MODEL_MAX_CANDIDATE_TRIPLES:-100000}")
    ;;
  stage3|rgcn)
    ARGS=(python -m pring_modeling.stage3_rgcn
      --modeling-dir "$RUN_DIR"
      --output-dir "$OUT_DIR/stage3_rgcn"
      --epochs "${MODEL_STAGE3_EPOCHS:-50}"
      --hidden-dim "${MODEL_HIDDEN_DIM:-128}"
      --num-layers "${MODEL_NUM_LAYERS:-2}"
      --batch-size "${MODEL_BATCH_SIZE:-4096}")
    ;;
  hgt)
    ARGS=(python -m pring_modeling.stage3_hgt
      --modeling-dir "$RUN_DIR"
      --output-dir "$OUT_DIR/stage3_hgt"
      --epochs "${MODEL_STAGE3_EPOCHS:-50}"
      --hidden-dim "${MODEL_HIDDEN_DIM:-128}"
      --num-layers "${MODEL_NUM_LAYERS:-2}"
      --batch-size "${MODEL_BATCH_SIZE:-4096}")
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
    echo "Unknown MODEL_STAGE=$STAGE" >&2; exit 2;;
esac

if [ "${MODEL_EXPORT_TO_NEO4J:-true}" = "true" ] && [[ "$STAGE" != "run_all" && "$STAGE" != "all" && "$STAGE" != "stage4" && "$STAGE" != "explain" && "$STAGE" != "compare" ]]; then
  ARGS+=(--export-neo4j --max-neo4j-predictions "${MODEL_MAX_NEO4J_PREDICTIONS:-25000}")
fi
if [ "${MODEL_SCORE_CANDIDATES:-false}" = "true" ] && [[ "$STAGE" == "stage3" || "$STAGE" == "rgcn" || "$STAGE" == "hgt" ]]; then
  ARGS+=(--score-candidates --max-candidate-pairs "${MODEL_MAX_CANDIDATE_PAIRS:-100000}")
fi
if [ -n "${MODEL_MAX_DEPTH:-}" ] && [[ "$STAGE" == "stage1" || "$STAGE" == "stage1_tabular" || "$STAGE" == "tabular" ]]; then
  ARGS+=(--max-depth "$MODEL_MAX_DEPTH")
fi

echo "Running modeling stage: $STAGE"
printf 'Command: '; printf '%q ' "${ARGS[@]}"; printf '\n'
"${ARGS[@]}"
