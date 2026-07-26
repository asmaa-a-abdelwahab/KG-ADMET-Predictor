# Troubleshooting PRING installation inside Docker

## Symptom

```text
Installing PRING from package spec: pring
ERROR: Could not find a version that satisfies the requirement pring
ERROR: No matching distribution found for pring
```

## Cause

The container is trying to install `pring` from PyPI. PRING is not published on PyPI, so the runtime must install it from either:

1. a mounted local clone, or
2. the public GitHub repository source archive.

## Recommended `.env` options

### Option A: use your local PRING clone

```dotenv
PRING_PROJECT_DIR=A:/Repositories/PRING-PACKAGE
PRING_PACKAGE_DIR=/workspace/pring
PRING_PACKAGE_SPEC=https://github.com/asmaa-a-abdelwahab/PRING-PACKAGE/archive/refs/heads/main.zip
```

The local clone is preferred when it contains `pyproject.toml` or `setup.py`.

### Option B: install from GitHub

```dotenv
PRING_PACKAGE_SPEC=https://github.com/asmaa-a-abdelwahab/PRING-PACKAGE/archive/refs/heads/main.zip
```

This works without `git` because pip downloads the source ZIP archive directly. The Dockerfiles also include `git`, so this also works:

```dotenv
PRING_PACKAGE_SPEC=git+https://github.com/asmaa-a-abdelwahab/PRING-PACKAGE.git
```

## Rebuild after changing `.env`

```powershell
docker compose down
docker compose build --no-cache streamlit pring-loader modeling
docker compose up -d neo4j streamlit
```

Then run the loader when Neo4j is healthy:

```powershell
docker compose --profile load up --build pring-loader
```
