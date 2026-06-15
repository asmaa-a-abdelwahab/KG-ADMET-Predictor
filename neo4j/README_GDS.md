# Neo4j Docker image with Graph Data Science (GDS)

This update enables the Neo4j Graph Data Science plugin required by Stage 1 PRING/GDS baselines.

## What changed

- `Dockerfile` now sets `NEO4J_PLUGINS='["graph-data-science"]'` so the official Neo4j Docker entrypoint installs GDS at container startup.
- `conf/neo4j.conf` allowlists/unrestricts `gds.*` procedures.
- Removed invalid `gds.memory.transaction.max` because Neo4j 5.26/GDS does not recognize it and strict validation prevents startup.
- `utils/check_gds.py` verifies `RETURN gds.version()` and available `gds.*` procedures.

## Important docker-compose note

If your `docker-compose.yml` already defines `NEO4J_PLUGINS`, it overrides the Dockerfile `ENV` value. In that case, set it to include GDS:

```yaml
environment:
  NEO4J_PLUGINS: '["graph-data-science"]'
  NEO4J_dbms_security_procedures_unrestricted: "gds.*"
  NEO4J_dbms_security_procedures_allowlist: "gds.*"
```

If you also use APOC:

```yaml
environment:
  NEO4J_PLUGINS: '["apoc","graph-data-science"]'
  NEO4J_dbms_security_procedures_unrestricted: "apoc.*,gds.*"
  NEO4J_dbms_security_procedures_allowlist: "apoc.*,gds.*"
```

## Rebuild and verify

```bash
docker compose build neo4j
docker compose up -d neo4j
docker compose logs -f neo4j
```

If the previous failed container keeps restarting, recreate it cleanly:

```bash
docker compose down
docker compose build --no-cache neo4j
docker compose up -d neo4j
```

Verify:

```bash
docker compose exec neo4j cypher-shell -u neo4j -p cyp450kg "RETURN gds.version() AS gds_version;"
docker compose exec neo4j cypher-shell -u neo4j -p cyp450kg "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'gds.' RETURN name ORDER BY name LIMIT 20;"
```

If `RETURN gds.version()` works, rerun the Stage 1 GDS projection/embedding Cypher scripts and then `stage1_export_gds_features`.
