#!/usr/bin/env bash
set -euo pipefail

/opt/kg/bin/install_pring_runtime.sh
python /opt/kg/bin/check_neo4j.py

RUN_DIR="${PRING_RUN_DIR:-/runs/current}"
if [ ! -d "$RUN_DIR" ]; then
  echo "ERROR: PRING_RUN_DIR does not exist: $RUN_DIR" >&2
  echo "Mount your PRING run folders with PRING_RUNS_DIR and set PRING_RUN_DIR=/runs/<run_id>." >&2
  exit 2
fi

CMD=(python -m pring load-run
  --run-dir "$RUN_DIR"
  --neo4j-uri "${NEO4J_URI:-bolt://neo4j:7687}"
  --neo4j-user "${NEO4J_USER:-neo4j}"
  --neo4j-password "${NEO4J_PASSWORD:-cyp450kg}"
  --neo4j-db "${NEO4J_DATABASE:-neo4j}"
  --load-neo4j true
  --ensure-neo4j-schema true
  --rematerialize-schema "${PRING_REMATERIALIZE_SCHEMA:-false}" \
  --rematerialize-csv "${PRING_REMATERIALIZE_CSV:-false}" \
  --validate-dot-schema "${PRING_VALIDATE_DOT_SCHEMA:-false}" \
  --reserve-system-memory-mb "${PRING_RESERVE_SYSTEM_MEMORY_MB:-512}" \
  --complete-similar-compound-nodes false
  --allow-network "${PRING_ALLOW_NETWORK:-false}"
  --batch-size "${PRING_LOAD_BATCH_SIZE:-1000}"
  --candidate-pair-mode "${PRING_CANDIDATE_PAIR_MODE:-all}"
  --activity-threshold-um "${PRING_ACTIVITY_THRESHOLD_UM:-10}"
  --weak-activity-as-negative "${PRING_WEAK_ACTIVITY_AS_NEGATIVE:-true}"
)

if [ -n "${PRING_SCHEMA_DOT:-}" ]; then
  CMD+=(--schema-dot "${PRING_SCHEMA_DOT}")
fi

echo "Loading PRING run into Neo4j: $RUN_DIR"
printf 'Command: '; printf '%q ' "${CMD[@]}"; printf '\n'
"${CMD[@]}"

REPORT_DIR="/reports/neo4j"
mkdir -p "$REPORT_DIR"
python - <<'PY'
import json, os
from pathlib import Path
from neo4j import GraphDatabase
uri=os.getenv('NEO4J_URI','bolt://neo4j:7687')
user=os.getenv('NEO4J_USER','neo4j')
pw=os.getenv('NEO4J_PASSWORD','cyp450kg')
db=os.getenv('NEO4J_DATABASE','neo4j')
out=Path('/reports/neo4j/load_summary.json')
driver=GraphDatabase.driver(uri, auth=(user,pw))
with driver.session(database=db) as s:
    labels=s.run('CALL db.labels() YIELD label CALL { WITH label MATCH (n) WHERE label IN labels(n) RETURN count(n) AS count } RETURN label, count ORDER BY label').data()
    rels=s.run('CALL db.relationshipTypes() YIELD relationshipType CALL { WITH relationshipType MATCH ()-[r]->() WHERE type(r)=relationshipType RETURN count(r) AS count } RETURN relationshipType, count ORDER BY relationshipType').data()
summary={'node_counts': labels, 'relationship_counts': rels}
out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
driver.close()
PY
