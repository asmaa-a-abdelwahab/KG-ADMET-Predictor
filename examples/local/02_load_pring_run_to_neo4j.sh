#!/usr/bin/env bash
set -euo pipefail
# Example:
#   PRING_RUNS_DIR=/absolute/path/to/pring/runs PRING_RUN_DIR=/runs/cyp450_5enzymes ./examples/local/02_load_pring_run_to_neo4j.sh

docker compose --profile load up --build pring-loader
