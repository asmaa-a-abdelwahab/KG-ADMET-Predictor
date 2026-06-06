#!/bin/bash
#SBATCH --job-name=pring-load-neo4j
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# Adjust these paths for your cluster.
PROJECT_DIR=${PROJECT_DIR:-$PWD}
PRING_DIR=${PRING_DIR:-$PROJECT_DIR/../pring}
RUN_DIR=${RUN_DIR:-$PROJECT_DIR/runs/current}

module purge || true
module load python/3.11 || true

python -m venv "$PROJECT_DIR/.venv-kg-admet"
source "$PROJECT_DIR/.venv-kg-admet/bin/activate"
python -m pip install --upgrade pip
python -m pip install neo4j
python -m pip install -e "$PRING_DIR"

python -m pring load-run \
  --run-dir "$RUN_DIR" \
  --neo4j-uri "${NEO4J_URI:-bolt://localhost:7687}" \
  --neo4j-user "${NEO4J_USER:-neo4j}" \
  --neo4j-password "${NEO4J_PASSWORD:-cyp450kg}" \
  --neo4j-db "${NEO4J_DATABASE:-neo4j}" \
  --load-neo4j true \
  --ensure-neo4j-schema true \
  --rematerialize-schema true \
  --rematerialize-csv true \
  --validate-dot-schema true \
  --candidate-pair-mode all \
  --activity-threshold-um 10 \
  --weak-activity-as-negative true \
  --complete-similar-compound-nodes false \
  --allow-network false \
  --batch-size 1000
