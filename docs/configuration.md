# Configuration

Copy `.env.example` to `.env` for local configuration. Never commit the
resulting `.env`.

## Core paths

| Variable | Purpose |
|---|---|
| `PRING_PROJECT_DIR` | Local PRING-PACKAGE checkout mounted into containers |
| `PRING_RUNS_DIR` | Parent directory for package runs |
| `PRING_RUN_DIR` | Selected immutable run inside the container |
| `MODEL_PREPARED_DIR` | Train-only modeling exports and graph tensors |

## Graph

| Variable | Purpose |
|---|---|
| `NEO4J_URI` | Bolt endpoint |
| `NEO4J_USER` | Database user |
| `NEO4J_PASSWORD` | Secret supplied outside version control |
| `NEO4J_DATABASE` | Target database |
| `PRING_GRAPH_VERSION` | Run/snapshot identity used by prediction caching |

## Prediction safety

| Variable | Safe production value |
|---|---|
| `PREDICTION_ALLOW_STALE_CACHE` | `false` |
| `PREDICTION_ALLOW_RUNTIME_MISMATCH` | `false` |
| `PREDICTION_REQUIRE_VERIFIED_ARTIFACTS` | `true` |
| `PREDICTION_REQUIRE_PUBLISHABLE_ARTIFACTS` | `true` |
| `PREDICTION_PARITY_REQUIRED` | `true` |
| `PREDICTION_PARITY_SAMPLE_SIZE` | `200` or a larger justified stratified sample |
| `PREDICTION_PARITY_MAE_MAX` | `0.01` |
| `PREDICTION_PARITY_MAX_ABS_ERROR` | `0.05` |
| `PREDICTION_PARITY_SPEARMAN_MIN` | `0.99` |
| `PREDICTION_PARITY_DECISION_AGREEMENT_MIN` | `0.995` |
| `PREDICTION_DEBUG_ERRORS` | `false` |
| `PREDICTION_API_KEY` | Long random secret |

## Network exposure

`PRING_BIND_ADDRESS` defaults to `127.0.0.1`. Use a non-loopback binding only
behind reviewed TLS termination, authentication, access control, rate limits,
and monitoring.
