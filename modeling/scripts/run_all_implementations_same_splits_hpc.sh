#!/bin/bash
#SBATCH --job-name=all_impl_same_splits
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/asmaaali/KG-ADMET-Predictor/logs/all_impl_same_splits_%j.out
#SBATCH --error=/home/asmaaali/KG-ADMET-Predictor/logs/all_impl_same_splits_%j.err

if [ -z "${BASH_VERSION:-}" ]; then
  exec /bin/bash "$0" "$@"
fi

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/asmaaali/KG-ADMET-Predictor}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_DIR/modeling}"
BASE_MODELING_DIR="${PRING_BASE_MODELING_DIR:-${PRING_RUN_DIR:-/home/asmaaali/PRING/runs/cyp450_5enzymes_uncapped_raw_rematerialized/graph/ml/modeling}}"
RUN_ID="${MODEL_SHARED_SPLIT_RUN_ID:-shared_seed_${MODEL_SHARED_SPLIT_SEED:-42}_${MODEL_SHARED_SPLIT_STRATEGY:-compound}}"
SHARED_ROOT="${MODEL_SHARED_SPLIT_ROOT:-$PROJECT_DIR/shared_splits/$RUN_ID}"
PREPARED_MODELING_DIR="${MODEL_SHARED_PREPARED_DIR:-$SHARED_ROOT/modeling_prepared}"
OUTPUT_PARENT="${MODEL_SHARED_OUTPUT_PARENT:-$PROJECT_DIR/models_all_stages_same_splits/$RUN_ID}"
REPORT_PARENT="${MODEL_SHARED_REPORT_PARENT:-$PROJECT_DIR/reports/all_stages_same_splits/$RUN_ID}"
IMPLEMENTATIONS="${MODEL_IMPLS:-legacy improved improved_v2}"

mkdir -p "$PROJECT_DIR/logs" "$SHARED_ROOT" "$OUTPUT_PARENT" "$REPORT_PARENT"
cd "$PROJECT_DIR"

printf '============================================================\n'
printf 'PRING same-split comparison for all implementations\n'
printf 'Job ID: %s\n' "${SLURM_JOB_ID:-unknown}"
printf 'Node: %s\n' "$(hostname)"
printf 'Start time: %s\n' "$(date)"
printf 'Base modeling dir: %s\n' "$BASE_MODELING_DIR"
printf 'Prepared modeling dir: %s\n' "$PREPARED_MODELING_DIR"
printf 'Output parent: %s\n' "$OUTPUT_PARENT"
printf 'Report parent: %s\n' "$REPORT_PARENT"
printf 'Implementations: %s\n' "$IMPLEMENTATIONS"
printf '============================================================\n'

# Use improved_v2 for shared split preparation because it contains the newest neutral helper.
bash "$MODEL_ROOT/scripts/use_implementation.sh" improved_v2
export MODELING_PACKAGE_DIR="$MODEL_ROOT"
export PYTHONPATH="$MODEL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -m pip install -e "$MODEL_ROOT"

if [ "${MODEL_RECREATE_SHARED_SPLIT:-true}" = "true" ] || [ ! -f "$SHARED_ROOT/split_manifest.csv" ]; then
  echo "Creating/materializing shared split manifest..."
  python -m pring_modeling.shared_splits prepare \
    --source-modeling-dir "$BASE_MODELING_DIR" \
    --output-dir "$SHARED_ROOT" \
    --prepared-modeling-dir "$PREPARED_MODELING_DIR" \
    --strategy "${MODEL_SHARED_SPLIT_STRATEGY:-compound}" \
    --seed "${MODEL_SHARED_SPLIT_SEED:-42}" \
    --test-size "${MODEL_SHARED_TEST_SIZE:-0.15}" \
    --valid-size "${MODEL_SHARED_VALID_SIZE:-0.15}" \
    --force
else
  echo "Reusing existing shared split manifest: $SHARED_ROOT/split_manifest.csv"
fi

printf '\nShared split summary:\n'
cat "$SHARED_ROOT/split_summary.json" || true
printf '\n'

for impl in $IMPLEMENTATIONS; do
  printf '\n============================================================\n'
  printf 'Running implementation on shared split: %s\n' "$impl"
  printf '============================================================\n'

  bash "$MODEL_ROOT/scripts/use_implementation.sh" "$impl"
  export MODEL_IMPL="$impl"
  export MODEL_IMPLEMENTATION="$impl"
  export MODELING_PACKAGE_DIR="$MODEL_ROOT"
  export PYTHONPATH="$MODEL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  export PRING_RUN_DIR="$PREPARED_MODELING_DIR"
  export MODEL_OUTPUT_DIR="$OUTPUT_PARENT/$impl"
  export MODEL_REPORT_DIR="$REPORT_PARENT/$impl"

  mkdir -p "$MODEL_OUTPUT_DIR" "$MODEL_REPORT_DIR"

  # Do not let one implementation overwrite another. Each writes under its own subfolder.
  bash "$MODEL_ROOT/implementations/$impl/scripts/run_all_models_compare_hpc.sh"
done

printf '\n============================================================\n'
printf 'Cross-implementation comparison\n'
printf '============================================================\n'

bash "$MODEL_ROOT/scripts/use_implementation.sh" improved_v2
export MODEL_IMPL="improved_v2"
export MODEL_IMPLEMENTATION="improved_v2"
export MODELING_PACKAGE_DIR="$MODEL_ROOT"
export PYTHONPATH="$MODEL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -m pring_modeling.cross_implementation_compare \
  --outputs-root "$OUTPUT_PARENT" \
  --output-dir "$REPORT_PARENT/cross_implementation" \
  --implementations $IMPLEMENTATIONS \
  --primary-metric "${MODEL_PRIMARY_COMPARE_METRIC:-mcc}"

printf '\nDone. Main report:\n'
printf '%s\n' "$REPORT_PARENT/cross_implementation/cross_implementation_comparison.md"
printf 'End time: %s\n' "$(date)"
