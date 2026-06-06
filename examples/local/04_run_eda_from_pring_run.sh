#!/usr/bin/env bash
set -euo pipefail
# Runs PRING's native EDA pipeline from the mounted local PRING package/environment.
RUN_DIR="${PRING_RUN_DIR_HOST:-./runs/current}"
OUT_DIR="${EDA_OUTPUT_DIR:-./artifacts/reports/eda}"
python -m pring eda --run-path "$RUN_DIR" --output-dir "$OUT_DIR"
