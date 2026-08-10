#!/usr/bin/env bash
#SBATCH --job-name=pring-cyp-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --output=pring-cyp-full-%j.out
#SBATCH --error=pring-cyp-full-%j.err

# End-to-end PRING five-CYP workflow. This file is the Slurm entry point and
# the canonical implementation used by the local launcher:
#   1. collect or reuse source data;
#   2. create a new, non-destructive, modeling-ready PRING run;
#   3. create the EDA HTML/Markdown/JSON report;
#   4. load a dedicated Neo4j database and write train-compatible FastRP data;
#   5. export Stage 1 pair features;
#   6. train Stage 1, Stage 2, R-GCN and HGT;
#   7. run strict, multi-seed, calibrated final validation and comparison;
#   8. create machine-readable and Markdown readiness reports.
#
# This script does not start Neo4j.  Supply a dedicated, reachable Neo4j + GDS
# instance and inject NEO4J_PASSWORD through the scheduler or a secret manager.
# See README_FULL_PIPELINE.md and full_pipeline.env.example.

set -Eeuo pipefail
umask 027

CURRENT_STAGE="initialization"
PIPELINE_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

die() {
  log "ERROR: $*"
  exit 2
}

is_true() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_file() {
  [ -s "$1" ] || die "Required file is missing or empty: $1"
}

require_dir() {
  [ -d "$1" ] || die "Required directory does not exist: $1"
}

mark_stage() {
  local name="$1"
  mkdir -p "$STATUS_DIR"
  printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_DIR/${name}.complete"
}

on_exit() {
  local exit_code=$?
  if [ -n "${STATUS_DIR:-}" ]; then
    mkdir -p "$STATUS_DIR" || true
    if [ "$exit_code" -eq 0 ]; then
      printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$STATUS_DIR/pipeline.complete"
    else
      {
        printf 'status=failed\n'
        printf 'stage=%s\n' "$CURRENT_STAGE"
        printf 'exit_code=%s\n' "$exit_code"
        printf 'failed_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      } > "$STATUS_DIR/pipeline.failed"
    fi
  fi
  exit "$exit_code"
}
trap on_exit EXIT

# ---------------------------------------------------------------------------
# 1. Resolve repositories and immutable run/output identifiers
# ---------------------------------------------------------------------------

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -n "${FRAMEWORK_ROOT:-}" ]; then
  :
elif [ -d "$SUBMIT_DIR/PRING-PACKAGE" ] && [ -d "$SUBMIT_DIR/PRING-APP" ]; then
  FRAMEWORK_ROOT="$SUBMIT_DIR"
elif [ -d "$SUBMIT_DIR/../PRING-PACKAGE" ] && [ -d "$SUBMIT_DIR/modeling" ]; then
  FRAMEWORK_ROOT="$(cd "$SUBMIT_DIR/.." && pwd)"
else
  die "Set FRAMEWORK_ROOT to a directory containing PRING-PACKAGE and PRING-APP."
fi

PACKAGE_DIR="${PACKAGE_DIR:-$FRAMEWORK_ROOT/PRING-PACKAGE}"
APP_DIR="${APP_DIR:-$FRAMEWORK_ROOT/PRING-APP}"
RUN_ROOT="${PRING_RUN_ROOT:-$PACKAGE_DIR/runs}"
CACHE_DIR="${PRING_CACHE_DIR:-${SCRATCH:-$FRAMEWORK_ROOT}/pring_cache}"
VENV_DIR="${PRING_VENV:-${PRING_HPC_VENV:-$FRAMEWORK_ROOT/.venv-pring-hpc}}"
PIPELINE_CONTEXT="${PRING_EXECUTION_CONTEXT:-hpc}"
PIPELINE_CPUS="${PRING_CPUS:-${SLURM_CPUS_PER_TASK:-16}}"

case "$PIPELINE_CPUS" in
  ''|*[!0-9]*) die "PRING_CPUS/SLURM_CPUS_PER_TASK must be a positive integer." ;;
esac
[ "$PIPELINE_CPUS" -gt 0 ] || die "At least one pipeline CPU is required."

require_dir "$PACKAGE_DIR"
require_dir "$APP_DIR"
require_file "$PACKAGE_DIR/pyproject.toml"
require_file "$APP_DIR/modeling/pyproject.toml"

JOB_TOKEN="${SLURM_JOB_ID:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_ID="${PRING_RUN_ID:-cyp450_5enzymes_final_${JOB_TOKEN}}"
READY_RUN_ID="${PRING_READY_RUN_ID:-${RUN_ID}_ready}"

case "$RUN_ID:$READY_RUN_ID" in
  *[!A-Za-z0-9_.:-]*) die "Run identifiers may contain only letters, numbers, dot, underscore, colon, or hyphen." ;;
esac
[ "$RUN_ID" != "$READY_RUN_ID" ] || die "PRING_RUN_ID and PRING_READY_RUN_ID must be different."

READY_RUN_DIR="$RUN_ROOT/$READY_RUN_ID"
PIPELINE_OUTPUT_ROOT="${PIPELINE_OUTPUT_ROOT:-$APP_DIR/artifacts/$PIPELINE_CONTEXT/$READY_RUN_ID}"
EDA_DIR="${EDA_DIR:-$PIPELINE_OUTPUT_ROOT/reports/eda}"
MODEL_OUTPUT_DIR="${MODEL_OUTPUT_DIR:-$PIPELINE_OUTPUT_ROOT/models}"
MODEL_REPORT_DIR="${MODEL_REPORT_DIR:-$PIPELINE_OUTPUT_ROOT/reports/modeling}"
READINESS_DIR="${READINESS_DIR:-$PIPELINE_OUTPUT_ROOT/readiness}"
STATUS_DIR="${STATUS_DIR:-$PIPELINE_OUTPUT_ROOT/status}"
LOG_DIR="${PIPELINE_LOG_DIR:-$PIPELINE_OUTPUT_ROOT/logs}"

if [ -e "$PIPELINE_OUTPUT_ROOT" ]; then
  die "Output directory already exists: $PIPELINE_OUTPUT_ROOT. Use new run/output identifiers; this workflow never overwrites or resumes scientific results."
fi

mkdir -p \
  "$RUN_ROOT" "$CACHE_DIR" "$PIPELINE_OUTPUT_ROOT" "$EDA_DIR" \
  "$MODEL_OUTPUT_DIR" "$MODEL_REPORT_DIR" "$READINESS_DIR" \
  "$STATUS_DIR" "$LOG_DIR"

exec > >(tee -a "$LOG_DIR/pipeline.log") 2>&1

log "PRING full five-CYP pipeline ($PIPELINE_CONTEXT)"
log "Framework root: $FRAMEWORK_ROOT"
log "Package: $PACKAGE_DIR"
log "Application: $APP_DIR"
log "Ready run: $READY_RUN_DIR"
log "Pipeline outputs: $PIPELINE_OUTPUT_ROOT"
log "Job ID: ${SLURM_JOB_ID:-not-running-under-slurm}"
log "Host: $(hostname)"

# ---------------------------------------------------------------------------
# 2. Load modules and a prebuilt scientific Python environment
# ---------------------------------------------------------------------------

CURRENT_STAGE="runtime_setup"
if is_true "${PURGE_MODULES:-false}" && type module >/dev/null 2>&1; then
  module purge
fi
if [ -n "${HPC_MODULES:-}" ]; then
  type module >/dev/null 2>&1 || die "HPC_MODULES was set but the module command is unavailable."
  read -r -a MODULE_LIST <<< "$HPC_MODULES"
  module load "${MODULE_LIST[@]}"
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  if ! is_true "${BOOTSTRAP_ENV:-false}"; then
    die "Missing $VENV_DIR/bin/python. Prepare the environment once as documented, or submit with BOOTSTRAP_ENV=true."
  fi
  PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
  command -v "$PYTHON_BOOTSTRAP" >/dev/null 2>&1 || die "Python executable not found: $PYTHON_BOOTSTRAP"
  "$PYTHON_BOOTSTRAP" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
PYTHON_BIN="$VENV_DIR/bin/python"

if is_true "${BOOTSTRAP_ENV:-false}"; then
  "$PYTHON_BIN" -m pip install --upgrade pip
  case "${BOOTSTRAP_TORCH_PROFILE:-none}" in
    none)
      log "Using a preinstalled PyTorch/PyG runtime."
      ;;
    cpu)
      "$PYTHON_BIN" -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION:-2.5.1}"
      ;;
    cu124)
      "$PYTHON_BIN" -m pip install \
        --index-url https://download.pytorch.org/whl/cu124 \
        "torch==${TORCH_VERSION:-2.5.1}"
      ;;
    *)
      die "BOOTSTRAP_TORCH_PROFILE must be one of: none, cpu, cu124."
      ;;
  esac
  "$PYTHON_BIN" -m pip install -e "${PACKAGE_DIR}[analysis]"
  "$PYTHON_BIN" -m pip install -r "$APP_DIR/modeling/requirements.txt"
  "$PYTHON_BIN" -m pip install -e "$APP_DIR/modeling"
  if [ "${BOOTSTRAP_TORCH_PROFILE:-none}" != "none" ]; then
    "$PYTHON_BIN" -m pip install \
      "torch-geometric==${TORCH_GEOMETRIC_VERSION:-2.6.1}"
    "$PYTHON_BIN" -m pip install -r \
      "$APP_DIR/modeling/requirements-pyg-${BOOTSTRAP_TORCH_PROFILE}.txt"
  fi
fi

export PYTHONPATH="$PACKAGE_DIR:$APP_DIR/modeling${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PYTHON_BIN" - <<'PY'
import importlib
import sys

required = (
    "pring",
    "pring_modeling",
    "pandas",
    "numpy",
    "sklearn",
    "neo4j",
    "matplotlib",
    "torch",
    "torch_geometric",
)
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("Missing runtime dependencies:\n  " + "\n  ".join(missing))
print("Python:", sys.version)
PY

MODEL_DEVICE="${MODEL_DEVICE:-cuda}"
if [ "$MODEL_DEVICE" = "auto" ]; then
  MODEL_DEVICE="$("$PYTHON_BIN" - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)"
fi
export MODEL_DEVICE
if [ "$MODEL_DEVICE" = "cuda" ]; then
  "$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(
        "MODEL_DEVICE=cuda but PyTorch cannot access a GPU. "
        "Load the correct CUDA/PyTorch modules or set MODEL_DEVICE=cpu for a smoke run."
    )
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY
fi

export MODEL_STAGE1_CLASSIFIER="${MODEL_STAGE1_CLASSIFIER:-extra_trees}"
export MODEL_STAGE2_MODELS="${MODEL_STAGE2_MODELS:-complex distmult rotate}"
[ "$MODEL_STAGE1_CLASSIFIER" = "extra_trees" ] \
  || die "The complete improved_v2/deployable contract requires MODEL_STAGE1_CLASSIFIER=extra_trees."
for required_kge in complex distmult rotate; do
  case " $MODEL_STAGE2_MODELS " in
    *" $required_kge "*) ;;
    *) die "MODEL_STAGE2_MODELS must include complex, distmult, and rotate; missing $required_kge." ;;
  esac
done
case "${MODEL_FINAL_MIN_SEEDS:-5}" in
  ''|*[!0-9]*) die "MODEL_FINAL_MIN_SEEDS must be a positive integer." ;;
esac
[ "${MODEL_FINAL_MIN_SEEDS:-5}" -gt 0 ] \
  || die "MODEL_FINAL_MIN_SEEDS must be at least 1."
read -r -a CONFIGURED_FINAL_SEEDS <<< "${MODEL_FINAL_SEEDS:-1 2 3 4 5}"
[ "${#CONFIGURED_FINAL_SEEDS[@]}" -ge "${MODEL_FINAL_MIN_SEEDS:-5}" ] \
  || die "MODEL_FINAL_SEEDS contains fewer values than MODEL_FINAL_MIN_SEEDS."

PACKAGE_COMMIT="$(git -C "$PACKAGE_DIR" rev-parse HEAD 2>/dev/null || printf unknown)"
APP_COMMIT="$(git -C "$APP_DIR" rev-parse HEAD 2>/dev/null || printf unknown)"
PACKAGE_DIRTY_COUNT="$(git -C "$PACKAGE_DIR" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
APP_DIRTY_COUNT="$(git -C "$APP_DIR" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
if is_true "${REQUIRE_CLEAN_GIT:-true}" \
  && { [ "$PACKAGE_COMMIT" = "unknown" ] || [ "$APP_COMMIT" = "unknown" ] \
       || [ "$PACKAGE_DIRTY_COUNT" -ne 0 ] || [ "$APP_DIRTY_COUNT" -ne 0 ]; }; then
  die "Final runs require clean Git checkouts with resolvable commits. Package dirty files=$PACKAGE_DIRTY_COUNT; app dirty files=$APP_DIRTY_COUNT."
fi
{
  printf 'pipeline_started_at=%s\n' "$PIPELINE_STARTED_AT"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-none}"
  printf 'host=%s\n' "$(hostname)"
  printf 'package_commit=%s\n' "$PACKAGE_COMMIT"
  printf 'app_commit=%s\n' "$APP_COMMIT"
  printf 'package_dirty_files=%s\n' "$PACKAGE_DIRTY_COUNT"
  printf 'app_dirty_files=%s\n' "$APP_DIRTY_COUNT"
  printf 'run_id=%s\n' "$RUN_ID"
  printf 'ready_run_id=%s\n' "$READY_RUN_ID"
  printf 'execution_context=%s\n' "$PIPELINE_CONTEXT"
  printf 'pipeline_cpus=%s\n' "$PIPELINE_CPUS"
  printf 'model_device=%s\n' "$MODEL_DEVICE"
  printf 'activity_threshold_um=%s\n' "${PRING_ACTIVITY_THRESHOLD_UM:-10}"
  printf 'weak_activity_as_negative=%s\n' "${PRING_WEAK_ACTIVITY_AS_NEGATIVE:-false}"
  printf 'stage2_models=%s\n' "${MODEL_STAGE2_MODELS:-complex distmult rotate}"
  printf 'final_seeds=%s\n' "${MODEL_FINAL_SEEDS:-1 2 3 4 5}"
  printf 'python=%s\n' "$("$PYTHON_BIN" --version 2>&1)"
} > "$STATUS_DIR/run_environment.txt"
mark_stage runtime_setup

# ---------------------------------------------------------------------------
# 3. Collect a new five-CYP run, or reuse a supplied source run
# ---------------------------------------------------------------------------

CURRENT_STAGE="data_collection"
SOURCE_RUN_DIR="${PRING_SOURCE_RUN_DIR:-}"
if [ -z "$SOURCE_RUN_DIR" ]; then
  RAW_RUN_DIR="$RUN_ROOT/$RUN_ID"
  [ ! -e "$RAW_RUN_DIR" ] || die "Raw run already exists: $RAW_RUN_DIR"

  TARGET_FILE="${PRING_TARGET_FILE:-$PACKAGE_DIR/examples/inputs/target_ids_cyp450_5.txt}"
  require_file "$TARGET_FILE"
  "$PYTHON_BIN" - "$TARGET_FILE" <<'PY'
from pathlib import Path
import sys

expected = {"P08684", "P05177", "P33261", "P11712", "P10635"}
values = [
    line.strip()
    for line in Path(sys.argv[1]).read_text(encoding="utf-8-sig").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
if len(values) != 5 or set(values) != expected:
    raise SystemExit(
        "The final five-CYP target file must contain exactly these UniProt "
        f"accessions once each: {', '.join(sorted(expected))}; found: {values}"
    )
PY
  read -r -a PLUGIN_LIST <<< "${PRING_PLUGINS:-uniprot go reactome interpro}"

  BUILD_ARGS=(
    "$PYTHON_BIN" -m pring build
    --mode "${PRING_MODE:-sparql}"
    --scope expand-from-targets
    --target-ids "$TARGET_FILE"
    --taxid 9606
    --case-study-mode final-cyp450
    --resource-profile high
    --max-workers "$PIPELINE_CPUS"
    --max-memory-mb "${PRING_MAX_MEMORY_MB:-225280}"
    --memory-safety-margin-mb "${PRING_MEMORY_MARGIN_MB:-8192}"
    --reserve-system-memory-mb "${PRING_SYSTEM_RESERVE_MB:-8192}"
    --sparql-page-size "${PRING_SPARQL_PAGE_SIZE:-10}"
    --sparql-timeout-s "${PRING_SPARQL_TIMEOUT_S:-300}"
    --sparql-evidence-timeout-s "${PRING_SPARQL_EVIDENCE_TIMEOUT_S:-300}"
    --sparql-max-retries "${PRING_SPARQL_MAX_RETRIES:-5}"
    --sparql-evidence-max-retries "${PRING_SPARQL_EVIDENCE_MAX_RETRIES:-2}"
    --sparql-adaptive-chunking true
    --sparql-min-page-size 1
    --sparql-skip-failed-chunks false
    --max-compounds-per-target none
    --max-targets-per-compound none
    --max-substances-per-compound none
    --max-measuregroups-per-target none
    --max-measuregroups-per-compound none
    --max-endpoints-per-pair none
    --max-similar-compounds-per-compound none
    --max-textmine-records none
    --max-textmine-records-per-target none
    --max-textmine-references-per-pair none
    --max-enrichment-records-per-entity none
    --activity-threshold-um "${PRING_ACTIVITY_THRESHOLD_UM:-10}"
    --weak-activity-as-negative "${PRING_WEAK_ACTIVITY_AS_NEGATIVE:-false}"
    --candidate-pair-mode all
    --max-candidate-missing-pairs none
    --include-compound-similarity "${PRING_INCLUDE_SIMILARITY:-true}"
    --compound-similarity-threshold "${PRING_SIMILARITY_THRESHOLD:-90}"
    --include-textmining "${PRING_INCLUDE_TEXTMINING:-false}"
    --include-optional-context true
    --include-endpoint-metadata true
    --include-endpoint-references "${PRING_INCLUDE_ENDPOINT_REFERENCES:-false}"
    --write-csv-mirrors true
    --save-raw true
    --save-extracted true
    --cache-dir "$CACHE_DIR"
    --load-neo4j false
    --overwrite-run false
    --out-dir "$RUN_ROOT"
    --run-id "$RUN_ID"
  )
  if [ "${#PLUGIN_LIST[@]}" -gt 0 ]; then
    BUILD_ARGS+=(--plugins "${PLUGIN_LIST[@]}")
  fi

  log "Starting uncapped five-CYP data collection."
  printf 'Command:'; printf ' %q' "${BUILD_ARGS[@]}"; printf '\n'
  "${BUILD_ARGS[@]}"
  SOURCE_RUN_DIR="$RAW_RUN_DIR"
else
  require_dir "$SOURCE_RUN_DIR"
  SOURCE_RUN_DIR="$(cd "$SOURCE_RUN_DIR" && pwd)"
  log "Reusing source run without modifying it: $SOURCE_RUN_DIR"
fi
require_file "$SOURCE_RUN_DIR/manifest.json"
SOURCE_QUALITY_REPORT="$SOURCE_RUN_DIR/graph/run_quality_report.json"
require_file "$SOURCE_QUALITY_REPORT"
"$PYTHON_BIN" - "$SOURCE_QUALITY_REPORT" <<'PY'
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
quality = json.loads(path.read_text(encoding="utf-8-sig"))
cap_status = (quality.get("cap_completeness_report") or {}).get(
    "data_completeness_status"
)
readiness = quality.get("cyp450_gcn_readiness_report") or {}
candidate_mode = (readiness.get("ml_pair_summary") or {}).get(
    "candidate_missing_pair_mode"
)
blockers = readiness.get("blockers") or []
problems = []
if cap_status != "uncapped_or_no_internal_caps_detected":
    problems.append(f"data completeness is {cap_status!r}")
if readiness.get("pipeline_validation_ready") is not True:
    problems.append("pipeline_validation_ready is not true")
if blockers:
    problems.append(f"readiness blockers={blockers}")
if str(candidate_mode).lower() != "all":
    problems.append(f"candidate pair mode is {candidate_mode!r}, not 'all'")
if problems:
    raise SystemExit(
        f"Source quality gate failed for {path}: " + "; ".join(problems)
    )
print(
    "Source quality gate passed: uncapped, all candidates, "
    "and ready for pipeline validation."
)
PY
mark_stage data_collection

# ---------------------------------------------------------------------------
# 4. Create a fresh modeling-ready copy and regenerate all derived artifacts
# ---------------------------------------------------------------------------

CURRENT_STAGE="modeling_data_preparation"
[ ! -e "$READY_RUN_DIR" ] || die "Ready run already exists: $READY_RUN_DIR"

MATERIALIZE_ARGS=(
  "$PYTHON_BIN" -m pring load-run
  --run-dir "$SOURCE_RUN_DIR"
  --out-dir "$RUN_ROOT"
  --run-id "$READY_RUN_ID"
  --schema-dot "$PACKAGE_DIR/schema/pring-implementation-ready-schema.dot"
  --rematerialize-schema true
  --rematerialize-csv true
  --ensure-neo4j-schema false
  --validate-dot-schema true
  --complete-similar-compound-nodes false
  --allow-network false
  --load-neo4j false
  --case-study-mode final-cyp450
  --activity-threshold-um "${PRING_ACTIVITY_THRESHOLD_UM:-10}"
  --weak-activity-as-negative "${PRING_WEAK_ACTIVITY_AS_NEGATIVE:-false}"
  --candidate-pair-mode all
  --max-candidate-missing-pairs none
  --write-csv-mirrors true
)
printf 'Command:'; printf ' %q' "${MATERIALIZE_ARGS[@]}"; printf '\n'
"${MATERIALIZE_ARGS[@]}"

MODELING_DIR="$READY_RUN_DIR/graph/ml/modeling"
MODEL_PROVENANCE_MANIFEST="$MODELING_DIR/modeling_stage_manifest.json"
require_file "$MODEL_PROVENANCE_MANIFEST"
require_file "$MODELING_DIR/stage3_heterogeneous_gnn/edge_index_train_only.csv"
mark_stage modeling_data_preparation

# ---------------------------------------------------------------------------
# 5. EDA report from the frozen, modeling-ready run
# ---------------------------------------------------------------------------

CURRENT_STAGE="eda"
"$PYTHON_BIN" -m pring eda \
  --run-path "$READY_RUN_DIR" \
  --output-dir "$EDA_DIR" \
  --top-n "${EDA_TOP_N:-30}"
require_file "$EDA_DIR/eda_report.html"
require_file "$EDA_DIR/eda_report.md"
require_file "$EDA_DIR/eda_summary.json"

"$PYTHON_BIN" "$APP_DIR/examples/hpc/validate_pipeline_artifacts.py" \
  --run-dir "$READY_RUN_DIR" \
  --phase prepared \
  --eda-dir "$EDA_DIR" \
  --output-json "$READINESS_DIR/prepared_readiness.json" \
  --output-markdown "$READINESS_DIR/prepared_readiness.md" \
  --strict
mark_stage eda

# ---------------------------------------------------------------------------
# 6. Dedicated Neo4j load, leakage-safe GDS projection, and Stage 1 features
# ---------------------------------------------------------------------------

CURRENT_STAGE="neo4j_gds_features"
if [ -z "${NEO4J_PASSWORD:-}" ]; then
  die "NEO4J_PASSWORD is required. Inject it through Slurm or a secret manager; do not commit it."
fi
export NEO4J_URI="${NEO4J_URI:-bolt://127.0.0.1:7687}"
export NEO4J_USER="${NEO4J_USER:-neo4j}"
export NEO4J_DATABASE="${NEO4J_DATABASE:-neo4j}"

"$PYTHON_BIN" - <<'PY'
import os
from neo4j import GraphDatabase

uri = os.environ["NEO4J_URI"]
user = os.environ["NEO4J_USER"]
password = os.environ["NEO4J_PASSWORD"]
database = os.environ["NEO4J_DATABASE"]
allow_nonempty = os.environ.get("ALLOW_NONEMPTY_NEO4J", "false").lower() == "true"
driver = GraphDatabase.driver(uri, auth=(user, password))
try:
    driver.verify_connectivity()
    with driver.session(database=database) as session:
        node_count = int(session.run("MATCH (n) RETURN count(n) AS n").single()["n"])
        gds_version = session.run("RETURN gds.version() AS version").single()["version"]
    print(f"Neo4j connectivity OK; nodes={node_count}; GDS={gds_version}")
    if node_count and not allow_nonempty:
        raise SystemExit(
            "Neo4j is not empty. A stale graph would contaminate FastRP features. "
            "Use a fresh database or explicitly set ALLOW_NONEMPTY_NEO4J=true."
        )
finally:
    driver.close()
PY

"$PYTHON_BIN" -m pring load-run \
  --run-dir "$READY_RUN_DIR" \
  --schema-dot "$PACKAGE_DIR/schema/pring-implementation-ready-schema.dot" \
  --rematerialize-schema false \
  --rematerialize-csv false \
  --ensure-neo4j-schema true \
  --validate-dot-schema true \
  --complete-similar-compound-nodes false \
  --allow-network false \
  --load-neo4j true \
  --batch-size "${NEO4J_LOAD_BATCH_SIZE:-1000}"

STAGE1_DIR="$MODELING_DIR/stage1_neo4j_gds_baselines"
STAGE1_PAIR_FILE="$STAGE1_DIR/compound_target_training_pairs_for_gds.csv"
STAGE1_GDS_SUMMARY="$STAGE1_DIR/stage1_outcome_safe_gds_summary.json"
require_file "$STAGE1_PAIR_FILE"
"$PYTHON_BIN" "$APP_DIR/examples/hpc/prepare_leakage_safe_gds.py" \
  --pair-file "$STAGE1_PAIR_FILE" \
  --provenance-manifest "$MODEL_PROVENANCE_MANIFEST" \
  --output-json "$STAGE1_GDS_SUMMARY" \
  --graph-name "${STAGE1_GDS_GRAPH_NAME:-pring_stage1_outcome_safe}" \
  --write-property "${STAGE1_EMBEDDING_PROPERTY:-pringFastRP}" \
  --embedding-dimension "${STAGE1_EMBEDDING_DIMENSION:-128}" \
  --seed "${MODEL_SEED:-42}" \
  --batch-size "${STAGE1_NEO4J_BATCH_SIZE:-5000}"

"$PYTHON_BIN" -m pring_modeling.stage1_export_gds_features \
  --modeling-dir "$MODELING_DIR" \
  --include-candidates \
  --max-training-rows 0 \
  --max-candidate-rows "${STAGE1_MAX_CANDIDATE_ROWS:-0}" \
  --chunk-size "${STAGE1_EXPORT_CHUNK_SIZE:-50000}" \
  --neo4j-batch-size "${STAGE1_NEO4J_BATCH_SIZE:-5000}"

require_file "$STAGE1_DIR/compound_target_training_pairs_gds_features.csv"
require_file "$STAGE1_DIR/candidate_pairs_gds_features.csv"
require_file "$STAGE1_DIR/stage1_gds_feature_export_summary.json"
require_file "$STAGE1_GDS_SUMMARY"
mark_stage neo4j_gds_features

# ---------------------------------------------------------------------------
# 7. Train all component models through the canonical improved_v2 implementation
# ---------------------------------------------------------------------------

CURRENT_STAGE="component_modeling"
export PROJECT_DIR="$APP_DIR"
export MODEL_ROOT="$APP_DIR/modeling"
export MODELING_PACKAGE_DIR="$APP_DIR/modeling"
export MODEL_IMPL="improved_v2"
export MODEL_INSTALL_RUNTIME="false"
export PRING_RUN_DIR="$MODELING_DIR"
export MODEL_OUTPUT_DIR
export MODEL_REPORT_DIR
export MODEL_DEVICE
export MODEL_STAGE1_DEVICE="${MODEL_STAGE1_DEVICE:-$MODEL_DEVICE}"
export MODEL_STAGE2_DEVICE="${MODEL_STAGE2_DEVICE:-$MODEL_DEVICE}"
export MODEL_STAGE3_DEVICE="${MODEL_STAGE3_DEVICE:-$MODEL_DEVICE}"
export MODEL_PRIMARY_COMPARE_METRIC="${MODEL_PRIMARY_COMPARE_METRIC:-mcc}"
export MODEL_MIN_SPECIFICITY="${MODEL_MIN_SPECIFICITY:-0.50}"
export MODEL_MIN_RECALL="${MODEL_MIN_RECALL:-0.0}"
export MODEL_STAGE2_SCORE_BATCH_SIZE="${MODEL_STAGE2_SCORE_BATCH_SIZE:-65536}"
export MODEL_SEED="${MODEL_SEED:-42}"
export MODEL_N_JOBS="$PIPELINE_CPUS"
export RUN_STAGE1="true"
export RUN_STAGE2="true"
export RUN_STAGE3_RGCN="true"
export RUN_STAGE3_HGT="true"
export RUN_ENSEMBLE="false"
export RUN_FINAL_VALIDATION="false"
export RUN_COMPARE="false"

bash "$APP_DIR/modeling/implementations/improved_v2/scripts/run_all_models_compare_hpc.sh"
require_file "$MODEL_OUTPUT_DIR/stage1_gds_${MODEL_STAGE1_CLASSIFIER}/metrics.json"
for kge_model in complex distmult rotate; do
  require_file "$MODEL_OUTPUT_DIR/stage2_${kge_model}_supervised/metrics.json"
done
require_file "$MODEL_OUTPUT_DIR/stage3_rgcn_sampled/metrics.json"
require_file "$MODEL_OUTPUT_DIR/stage3_hgt_sampled/metrics.json"
mark_stage component_modeling

# ---------------------------------------------------------------------------
# 8. Strict calibration, multi-seed locked validation, and comparison
# ---------------------------------------------------------------------------

CURRENT_STAGE="final_validation"
FINAL_ARGS=(
  "$PYTHON_BIN" -m pring_modeling.final_validation
  --outputs-root "$MODEL_OUTPUT_DIR"
  --output-dir "$MODEL_OUTPUT_DIR/finalized_v2"
  --meta-classifier fixed_mean
  --split-strategy registered
  --calibration "${MODEL_FINAL_CALIBRATION:-platt}"
  --seeds "${MODEL_FINAL_SEEDS:-1 2 3 4 5}"
  --threshold-selection "${MODEL_PRIMARY_COMPARE_METRIC:-mcc}"
  --min-specificity "${MODEL_MIN_SPECIFICITY:-0.50}"
  --min-recall "${MODEL_MIN_RECALL:-0.0}"
  --report-min-specificity "${MODEL_REPORT_MIN_SPECIFICITY:-0.50}"
  --report-high-specificity "${MODEL_REPORT_HIGH_SPECIFICITY:-0.80}"
  --report-min-recall "${MODEL_REPORT_MIN_RECALL:-0.80}"
  --balanced-eval-max-per-class "${MODEL_BALANCED_EVAL_MAX_PER_CLASS:-0}"
  --bootstrap-resamples "${MODEL_FINAL_BOOTSTRAP_RESAMPLES:-1000}"
  --top-k-per-target "${MODEL_TOP_K_PER_TARGET:-50}"
  --uncertain-top-n "${MODEL_UNCERTAIN_TOP_N:-200}"
  --per-target-min-rows "${MODEL_PER_TARGET_MIN_ROWS:-100}"
  --n-jobs "$PIPELINE_CPUS"
  --strict-leakage-free
  --provenance-manifest "$MODEL_PROVENANCE_MANIFEST"
)
if [ -n "${MODEL_EXTERNAL_LABELS:-}" ]; then
  require_file "$MODEL_EXTERNAL_LABELS"
  FINAL_ARGS+=(--external-labels "$MODEL_EXTERNAL_LABELS")
fi
printf 'Command:'; printf ' %q' "${FINAL_ARGS[@]}"; printf '\n'
"${FINAL_ARGS[@]}"

"$PYTHON_BIN" -m pring_modeling.compare metrics \
  --outputs-root "$MODEL_OUTPUT_DIR" \
  --output-dir "$MODEL_REPORT_DIR/comparison" \
  --primary-metric "${MODEL_PRIMARY_COMPARE_METRIC:-mcc}"

require_file "$MODEL_OUTPUT_DIR/finalized_v2/metrics.json"
require_file "$MODEL_OUTPUT_DIR/finalized_v2/seed_metrics.csv"
require_file "$MODEL_OUTPUT_DIR/finalized_v2/seed_metric_summary.csv"
mark_stage final_validation

# ---------------------------------------------------------------------------
# 9. Final readiness report and human-readable hand-off
# ---------------------------------------------------------------------------

CURRENT_STAGE="readiness_report"
"$PYTHON_BIN" "$APP_DIR/examples/hpc/validate_pipeline_artifacts.py" \
  --run-dir "$READY_RUN_DIR" \
  --phase final \
  --eda-dir "$EDA_DIR" \
  --model-output-dir "$MODEL_OUTPUT_DIR" \
  --model-report-dir "$MODEL_REPORT_DIR" \
  --min-final-seeds "${MODEL_FINAL_MIN_SEEDS:-5}" \
  --output-json "$READINESS_DIR/final_readiness.json" \
  --output-markdown "$READINESS_DIR/final_readiness.md" \
  --strict

{
  printf '# PRING five-CYP full run\n\n'
  printf -- '- Execution context: `%s`\n' "$PIPELINE_CONTEXT"
  printf -- '- Ready run: `%s`\n' "$READY_RUN_DIR"
  printf -- '- Modeling data: `%s`\n' "$MODELING_DIR"
  printf -- '- EDA report: `%s`\n' "$EDA_DIR/eda_report.html"
  printf -- '- Model outputs: `%s`\n' "$MODEL_OUTPUT_DIR"
  printf -- '- Model comparison: `%s`\n' "$MODEL_REPORT_DIR/comparison"
  printf -- '- Final validation: `%s`\n' "$MODEL_OUTPUT_DIR/finalized_v2/metrics.json"
  printf -- '- Readiness report: `%s`\n' "$READINESS_DIR/final_readiness.md"
  printf -- '- PRING-PACKAGE commit: `%s`\n' "$PACKAGE_COMMIT"
  printf -- '- PRING-APP commit: `%s`\n' "$APP_COMMIT"
  printf -- '- Started: `%s`\n' "$PIPELINE_STARTED_AT"
  printf -- '- Finished: `%s`\n\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'Predictions are computational hypotheses for research use. '
  printf 'A passing readiness gate does not constitute clinical validation.\n'
} > "$PIPELINE_OUTPUT_ROOT/PIPELINE_SUMMARY.md"

mark_stage readiness_report
CURRENT_STAGE="complete"
log "Pipeline completed successfully."
log "EDA: $EDA_DIR/eda_report.html"
log "Final validation: $MODEL_OUTPUT_DIR/finalized_v2/metrics.json"
log "Readiness: $READINESS_DIR/final_readiness.md"
