#!/usr/bin/env bash
set -euo pipefail
cp -n .env.example .env || true
# Edit .env so PRING_RUNS_DIR points to the directory containing your PRING run folders
# and PRING_RUN_DIR points to one mounted run, e.g. /runs/cyp450_5enzymes.
docker compose up -d neo4j streamlit
