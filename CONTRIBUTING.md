# Contributing to PRING-APP

## Development setup

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Run tests with:

```bash
PYTHONPATH=modeling:streamlit python -m pytest -q
docker compose -f docker-compose.yml config --quiet
```

Tests marked `production_assets` require local model and result artifacts and
skip automatically in a clean clone.

## Scientific changes

- Use one registered outer split across components.
- Fit preprocessing, calibration, and thresholds on the correct partitions.
- Add parity tests when changing any live scorer.
- Preserve reference/cache isolation and versioned cache identity.
- Keep predicted relationships excluded from training evidence.
- Update reports and UI warnings when scientific semantics change.

## Documentation

```bash
python -m pip install -r requirements-docs.txt
mkdocs build --strict
```

Do not commit secrets, local Neo4j state, generated models/results, caches,
virtual environments, or documentation build output.

