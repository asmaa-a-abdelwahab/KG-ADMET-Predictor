# Installation and quick start

## Prerequisites

- Docker Engine or Docker Desktop with Compose
- A versioned PRING-PACKAGE run
- Sufficient memory for Neo4j and optional graph models

## Configure

```bash
cp .env.example .env
```

Replace the development credentials before any shared or production
deployment. Set:

- `PRING_RUNS_DIR` and `PRING_RUN_DIR`;
- `PRING_GRAPH_VERSION`;
- `NEO4J_PASSWORD`;
- `PREDICTION_API_KEY`.

## Start the services

```bash
docker compose up -d --build neo4j predictor streamlit
```

Published ports bind to localhost by default:

| Service | Default URL |
|---|---|
| Neo4j Browser | `http://127.0.0.1:7474` |
| Prediction API | `http://127.0.0.1:8000` |
| Streamlit | `http://127.0.0.1:8501` |

## Load a package run

```bash
docker compose --profile load up --build pring-loader
```

The loader uses PRING-PACKAGE schema constraints and graph artifacts. Repeating
the same load is idempotent, while distinct parallel evidence relationships
remain separate.

## Train

```bash
docker compose --profile train up --build modeling
```

Training is not automatically publication-valid. Confirm the registered split,
graph scope, out-of-fold provenance, calibration, and release gates before
promoting a bundle.

## Production overlay

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.production.yml \
  config
```

The production overlay refuses missing secrets and graph identity. It also
requires a verified model digest and compatible serialization runtime.

