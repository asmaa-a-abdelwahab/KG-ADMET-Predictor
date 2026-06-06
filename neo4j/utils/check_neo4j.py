"""Small Neo4j readiness check used by Docker scripts."""
from __future__ import annotations

import os
import sys
import time

from neo4j import GraphDatabase

uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
user = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "cyp450kg")
database = os.getenv("NEO4J_DATABASE", "neo4j")
timeout_s = int(os.getenv("NEO4J_WAIT_TIMEOUT", "180"))

start = time.time()
last_error: Exception | None = None
while time.time() - start < timeout_s:
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as session:
            session.run("RETURN 1 AS ok").single()
        driver.close()
        print(f"Neo4j is ready at {uri}/{database}.")
        sys.exit(0)
    except Exception as exc:  # pragma: no cover - runtime helper
        last_error = exc
        print(f"Waiting for Neo4j: {exc}")
        time.sleep(5)

print(f"Neo4j was not ready after {timeout_s}s: {last_error}", file=sys.stderr)
sys.exit(1)
