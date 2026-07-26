# Troubleshooting Neo4j startup and PRING loading

## Symptom

`docker compose --profile load up --build pring-loader` ends with:

```text
dependency failed to start: container pring-app-neo4j is unhealthy
```

This means the PRING loader did not run yet. Docker Compose stopped because the
Neo4j service did not pass its health check.

## First diagnosis commands

Run these from the PRING-APP repository root:

```powershell
docker compose ps
docker logs pring-app-neo4j --tail 200
```

If Neo4j is still running but marked unhealthy, inspect the health check result:

```powershell
docker inspect pring-app-neo4j --format '{{json .State.Health}}'
```

## Common fixes

### Plugin folder permission error

If Neo4j logs show:

```text
Folder /plugins is not accessible for user: 7474 or group 7474
```

remove any mounted `/plugins` volume. The project no longer mounts
`neo4j-plugins:/plugins`; the official Neo4j image can download APOC into its
internal plugins folder when `NEO4J_PLUGINS=["apoc"]` is set.

If you already created the broken volume, reset it with:

```powershell
docker compose down -v
docker compose up -d --build neo4j
```

This deletes the local Neo4j Docker volumes, so only use it when you do not need
the previous local database state.


### 1. Old Neo4j volume has a different password

Neo4j stores the initial password in the database volume. Changing `.env` later
does not reset an existing volume. If this is a fresh local database and you do
not need the previous Neo4j data, reset it:

```powershell
docker compose down -v
docker compose up -d --build neo4j
```

Then rerun:

```powershell
docker compose --profile load up --build pring-loader
```

### 2. Docker Desktop memory is too low

For local testing, start with conservative values in `.env`:

```dotenv
NEO4J_HEAP_INITIAL=1G
NEO4J_HEAP_MAX=3G
NEO4J_PAGECACHE=2G
```

After Neo4j starts reliably, increase these values if Docker Desktop has enough
allocated memory.


### 3. Duplicate or too-large Neo4j memory settings

The previous compose defaults were too large for many Docker Desktop installs.
This version uses conservative defaults directly in `docker-compose.yml`:

```dotenv
NEO4J_HEAP_INITIAL=1G
NEO4J_HEAP_MAX=2G
NEO4J_PAGECACHE=1G
```

The `neo4j/conf/neo4j.conf` file is intentionally minimal. Do not also hard-code
heap/pagecache or APOC settings there, because Docker environment variables also
write Neo4j config values. Duplicate config keys can stop Neo4j before it becomes
healthy.

### 4. Neo4j is healthy but the old health check used wget

Some Neo4j images do not include `wget`. This project now uses `cypher-shell` in
the health check, which is included in the Neo4j image and verifies the Bolt
connection used by PRING, modeling, and Streamlit.

## Manual startup sequence

If `depends_on` keeps blocking while you diagnose, start Neo4j first:

```powershell
docker compose up -d --build neo4j
docker logs -f pring-app-neo4j
```

When the logs show that Bolt is enabled, run:

```powershell
docker compose --profile load up --build pring-loader
```


## Neo4j startup loop: `sed: cannot rename ... Device or resource busy`

This happens when `neo4j.conf` is mounted as a single read-only file, for example:

```yaml
- ./neo4j/conf/neo4j.conf:/var/lib/neo4j/conf/neo4j.conf:ro
```

When `NEO4J_PLUGINS=["apoc"]` is enabled, the official Neo4j image copies APOC and tries to update `neo4j.conf` during startup. Docker Desktop cannot safely replace a bind-mounted single file, so the startup script fails with `sed: cannot rename ... Device or resource busy`.

Fix: do not bind-mount `neo4j.conf`. Bake the file into the image through `neo4j/Dockerfile`, as this project now does. After changing the compose file, reset failed containers and volumes if the database has not loaded yet:

```powershell
docker compose down -v
docker compose build --no-cache neo4j
docker compose up -d neo4j
docker logs -f pring-app-neo4j
```

If you want the most minimal local Neo4j without APOC, remove `NEO4J_PLUGINS` and the `NEO4J_apoc_*` environment variables from `docker-compose.yml`; PRING loading does not require APOC for the standard path.
