#!/usr/bin/env bash
set -euo pipefail
# Trains the lightweight tabular baseline from graph/ml/*.csv and exports predictions to Neo4j.
docker compose --profile train up --build modeling
