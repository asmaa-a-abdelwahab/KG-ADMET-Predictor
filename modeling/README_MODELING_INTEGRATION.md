# Modeling integration summary

This project archive updates the `modeling` Docker image so it contains all modeling stages, stage notebooks, comparison tools, and Neo4j prediction export.

Main entry point inside Docker:

```bash
/opt/kg/bin/run_modeling.sh
```

Default command from `docker-compose.yml`:

```bash
docker compose --profile train up --build modeling
```

The default workflow runs Stage 1 + Stage 2 and writes outputs to `artifacts/models` and `artifacts/reports/modeling`. Set `MODEL_STAGES="stage1 stage2 stage3"` to include Stage 3 when a PyG HeteroData export is available.
