"""Check that Neo4j Graph Data Science is installed and callable."""
from __future__ import annotations

import os
import sys
from neo4j import GraphDatabase

uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
user = os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD", "cyp450kg")
database = os.getenv("NEO4J_DATABASE", "neo4j")

driver = GraphDatabase.driver(uri, auth=(user, password))
try:
    with driver.session(database=database) as session:
        version = session.run("RETURN gds.version() AS version").single()["version"]
        procedures = session.run(
            "SHOW PROCEDURES YIELD name "
            "WHERE name STARTS WITH 'gds.' "
            "RETURN count(name) AS n"
        ).single()["n"]
    print(f"Neo4j GDS is available. version={version}; gds_procedures={procedures}")
except Exception as exc:
    print("Neo4j GDS is NOT available or not callable.", file=sys.stderr)
    print(str(exc), file=sys.stderr)
    sys.exit(1)
finally:
    driver.close()
