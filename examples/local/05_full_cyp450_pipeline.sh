#!/usr/bin/env bash

# Complete local entry point for the same five-CYP scientific workflow used on
# Slurm. It may start a private, ephemeral Neo4j+GDS container, then delegates
# to the canonical pipeline implementation in the HPC script (SBATCH lines are
# comments when that file is executed with Bash).

set -Eeuo pipefail
umask 027

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_APP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ "$#" -gt 1 ]; then
  die "Usage: $0 [path/to/local-full-pipeline.env]"
fi
CONFIG_FILE="${1:-${PRING_PIPELINE_ENV:-}}"
if [ -n "$CONFIG_FILE" ]; then
  [ -s "$CONFIG_FILE" ] || die "Configuration file is missing or empty: $CONFIG_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
  set +a
fi

export APP_DIR="${APP_DIR:-$DEFAULT_APP_DIR}"
export FRAMEWORK_ROOT="${FRAMEWORK_ROOT:-$(cd "$APP_DIR/.." && pwd)}"
export PACKAGE_DIR="${PACKAGE_DIR:-$FRAMEWORK_ROOT/PRING-PACKAGE}"
[ -s "$PACKAGE_DIR/pyproject.toml" ] \
  || die "PRING-PACKAGE was not found at $PACKAGE_DIR. Set FRAMEWORK_ROOT or PACKAGE_DIR."

LOCAL_TOKEN="$(date -u +%Y%m%d_%H%M%S)"
export PRING_RUN_ID="${PRING_RUN_ID:-cyp450_5enzymes_local_${LOCAL_TOKEN}}"
export PRING_READY_RUN_ID="${PRING_READY_RUN_ID:-${PRING_RUN_ID}_ready}"
case "$PRING_RUN_ID:$PRING_READY_RUN_ID" in
  *[!A-Za-z0-9_.:-]*) die "Run identifiers contain unsupported characters." ;;
esac
[ "$PRING_RUN_ID" != "$PRING_READY_RUN_ID" ] \
  || die "PRING_RUN_ID and PRING_READY_RUN_ID must be different."
export PRING_RUN_ROOT="${PRING_RUN_ROOT:-$PACKAGE_DIR/runs}"
export PRING_CACHE_DIR="${PRING_CACHE_DIR:-$FRAMEWORK_ROOT/pring_cache}"
export PIPELINE_OUTPUT_ROOT="${PIPELINE_OUTPUT_ROOT:-$APP_DIR/artifacts/local/$PRING_READY_RUN_ID}"
export PRING_EXECUTION_CONTEXT="local"
export PRING_VENV="${PRING_VENV:-$FRAMEWORK_ROOT/.venv-pring-local}"
export BOOTSTRAP_ENV="${BOOTSTRAP_ENV:-true}"
export BOOTSTRAP_TORCH_PROFILE="${BOOTSTRAP_TORCH_PROFILE:-cpu}"
export MODEL_DEVICE="${MODEL_DEVICE:-auto}"
export REQUIRE_CLEAN_GIT="${REQUIRE_CLEAN_GIT:-true}"

if [ -z "${PRING_CPUS:-}" ]; then
  PRING_CPUS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf 4)"
fi
export PRING_CPUS

if [ -z "${PRING_MAX_MEMORY_MB:-}" ]; then
  if [ -r /proc/meminfo ]; then
    PRING_MAX_MEMORY_MB="$(awk '/MemTotal:/ {print int(($2 / 1024) * 0.85)}' /proc/meminfo)"
  elif command -v sysctl >/dev/null 2>&1; then
    LOCAL_MEMORY_BYTES="$(sysctl -n hw.memsize 2>/dev/null || printf 0)"
    PRING_MAX_MEMORY_MB="$((LOCAL_MEMORY_BYTES * 85 / 100 / 1024 / 1024))"
  else
    PRING_MAX_MEMORY_MB=16384
  fi
fi
case "$PRING_MAX_MEMORY_MB" in
  ''|*[!0-9]*) die "PRING_MAX_MEMORY_MB must be a positive integer in MB." ;;
esac
[ "$PRING_MAX_MEMORY_MB" -gt 0 ] \
  || die "PRING_MAX_MEMORY_MB must be greater than zero."
export PRING_MAX_MEMORY_MB
export PRING_MEMORY_MARGIN_MB="${PRING_MEMORY_MARGIN_MB:-1024}"
export PRING_SYSTEM_RESERVE_MB="${PRING_SYSTEM_RESERVE_MB:-2048}"

# The final experiment contract. These may be changed only deliberately; any
# change creates a distinct experiment and is captured in run_environment.txt.
export PRING_ACTIVITY_THRESHOLD_UM="${PRING_ACTIVITY_THRESHOLD_UM:-10}"
export PRING_WEAK_ACTIVITY_AS_NEGATIVE="${PRING_WEAK_ACTIVITY_AS_NEGATIVE:-false}"
export PRING_INCLUDE_SIMILARITY="${PRING_INCLUDE_SIMILARITY:-true}"
export PRING_SIMILARITY_THRESHOLD="${PRING_SIMILARITY_THRESHOLD:-90}"
export PRING_INCLUDE_TEXTMINING="${PRING_INCLUDE_TEXTMINING:-false}"
export PRING_INCLUDE_ENDPOINT_REFERENCES="${PRING_INCLUDE_ENDPOINT_REFERENCES:-false}"
export PRING_PLUGINS="${PRING_PLUGINS:-uniprot go reactome interpro}"
export MODEL_STAGE1_CLASSIFIER="${MODEL_STAGE1_CLASSIFIER:-extra_trees}"
export MODEL_STAGE2_MODELS="${MODEL_STAGE2_MODELS:-complex distmult rotate}"
export MODEL_FINAL_SEEDS="${MODEL_FINAL_SEEDS:-1 2 3 4 5}"
export MODEL_FINAL_MIN_SEEDS="${MODEL_FINAL_MIN_SEEDS:-5}"
export MODEL_FINAL_CALIBRATION="${MODEL_FINAL_CALIBRATION:-platt}"
export MODEL_FINAL_BOOTSTRAP_RESAMPLES="${MODEL_FINAL_BOOTSTRAP_RESAMPLES:-1000}"

LOCAL_NEO4J_MODE="${LOCAL_NEO4J_MODE:-managed}"
LOCAL_KEEP_NEO4J="${LOCAL_KEEP_NEO4J:-false}"
MANAGED_CONTAINER=""
MANAGED_VOLUME=""

cleanup() {
  local exit_code=$?
  if [ -n "$MANAGED_CONTAINER" ] && ! is_true "$LOCAL_KEEP_NEO4J"; then
    log "Removing pipeline-managed Neo4j container $MANAGED_CONTAINER."
    docker rm --force "$MANAGED_CONTAINER" >/dev/null 2>&1 || true
  fi
  if [ -n "$MANAGED_VOLUME" ] && ! is_true "$LOCAL_KEEP_NEO4J"; then
    log "Removing pipeline-managed Neo4j volume $MANAGED_VOLUME."
    docker volume rm "$MANAGED_VOLUME" >/dev/null 2>&1 || true
  fi
  exit "$exit_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

case "$LOCAL_NEO4J_MODE" in
  external)
    [ -n "${NEO4J_PASSWORD:-}" ] \
      || die "NEO4J_PASSWORD is required for LOCAL_NEO4J_MODE=external."
    export NEO4J_PASSWORD
    export NEO4J_URI="${NEO4J_URI:-bolt://127.0.0.1:7687}"
    export NEO4J_USER="${NEO4J_USER:-neo4j}"
    export NEO4J_DATABASE="${NEO4J_DATABASE:-neo4j}"
    log "Using external Neo4j+GDS at $NEO4J_URI."
    ;;
  managed)
    command -v docker >/dev/null 2>&1 \
      || die "Docker is required for LOCAL_NEO4J_MODE=managed."
    docker info >/dev/null 2>&1 \
      || die "Docker is installed but its daemon is not reachable."
    [ -n "${NEO4J_PASSWORD:-}" ] \
      || die "Set a private NEO4J_PASSWORD (at least 8 characters)."
    [ "${#NEO4J_PASSWORD}" -ge 8 ] \
      || die "NEO4J_PASSWORD must contain at least 8 characters."

    export NEO4J_PASSWORD
    export NEO4J_USER="${NEO4J_USER:-neo4j}"
    export NEO4J_DATABASE="${NEO4J_DATABASE:-neo4j}"
    LOCAL_NEO4J_PORT="${LOCAL_NEO4J_PORT:-17687}"
    case "$LOCAL_NEO4J_PORT" in
      ''|*[!0-9]*) die "LOCAL_NEO4J_PORT must be numeric." ;;
    esac
    [ "$LOCAL_NEO4J_PORT" -ge 1024 ] && [ "$LOCAL_NEO4J_PORT" -le 65535 ] \
      || die "LOCAL_NEO4J_PORT must be between 1024 and 65535."
    export NEO4J_URI="bolt://127.0.0.1:$LOCAL_NEO4J_PORT"

    if [ -z "${LOCAL_NEO4J_HEAP_MAX:-}" ]; then
      if [ "$PRING_MAX_MEMORY_MB" -ge 64000 ]; then
        LOCAL_NEO4J_HEAP_MAX=8G
        LOCAL_NEO4J_PAGECACHE="${LOCAL_NEO4J_PAGECACHE:-4G}"
      elif [ "$PRING_MAX_MEMORY_MB" -ge 32000 ]; then
        LOCAL_NEO4J_HEAP_MAX=4G
        LOCAL_NEO4J_PAGECACHE="${LOCAL_NEO4J_PAGECACHE:-2G}"
      else
        LOCAL_NEO4J_HEAP_MAX=2G
        LOCAL_NEO4J_PAGECACHE="${LOCAL_NEO4J_PAGECACHE:-1G}"
      fi
    fi
    LOCAL_NEO4J_PAGECACHE="${LOCAL_NEO4J_PAGECACHE:-1G}"

    SAFE_READY_ID="${PRING_READY_RUN_ID//:/-}"
    MANAGED_CONTAINER="${LOCAL_NEO4J_CONTAINER:-pring-full-${SAFE_READY_ID}}"
    MANAGED_VOLUME="${LOCAL_NEO4J_VOLUME:-${MANAGED_CONTAINER}-data}"
    if docker container inspect "$MANAGED_CONTAINER" >/dev/null 2>&1; then
      die "Managed Neo4j container already exists: $MANAGED_CONTAINER"
    fi
    if docker volume inspect "$MANAGED_VOLUME" >/dev/null 2>&1; then
      die "Managed Neo4j volume already exists: $MANAGED_VOLUME"
    fi

    docker volume create "$MANAGED_VOLUME" >/dev/null
    export NEO4J_AUTH="$NEO4J_USER/$NEO4J_PASSWORD"
    log "Starting isolated Neo4j+GDS container $MANAGED_CONTAINER."
    docker run --detach --rm \
      --name "$MANAGED_CONTAINER" \
      --publish "127.0.0.1:$LOCAL_NEO4J_PORT:7687" \
      --volume "$MANAGED_VOLUME:/data" \
      --env NEO4J_AUTH \
      --env 'NEO4J_PLUGINS=["graph-data-science"]' \
      --env 'NEO4J_dbms_security_procedures_unrestricted=gds.*' \
      --env 'NEO4J_dbms_security_procedures_allowlist=gds.*' \
      --env "NEO4J_server_memory_heap_initial__size=${LOCAL_NEO4J_HEAP_INITIAL:-2G}" \
      --env "NEO4J_server_memory_heap_max__size=$LOCAL_NEO4J_HEAP_MAX" \
      --env "NEO4J_server_memory_pagecache_size=$LOCAL_NEO4J_PAGECACHE" \
      "${LOCAL_NEO4J_IMAGE:-neo4j:5.26-community}" >/dev/null

    log "Waiting for Neo4j and GDS to become ready."
    ready=false
    for ((attempt = 1; attempt <= 180; attempt++)); do
      if docker exec "$MANAGED_CONTAINER" cypher-shell \
        -a bolt://127.0.0.1:7687 \
        -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" \
        'RETURN gds.version() AS version;' >/dev/null 2>&1; then
        ready=true
        break
      fi
      sleep 2
    done
    [ "$ready" = true ] \
      || die "Neo4j+GDS did not become ready; inspect: docker logs $MANAGED_CONTAINER"
    ;;
  *)
    die "LOCAL_NEO4J_MODE must be managed or external."
    ;;
esac

log "Starting complete local five-CYP run $PRING_READY_RUN_ID."
bash "$APP_DIR/examples/hpc/04_full_cyp450_pipeline.sbatch"
