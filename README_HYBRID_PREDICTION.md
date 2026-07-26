# PRING hybrid prediction and persistent score-cache patch

## Requested behavior

The predictor now follows this sequence for every compound-CYP450 pair:

1. Resolve the canonical `compound_key` and `target_key`.
2. Look for exact component scores in `finalized_training_frame.csv`.
3. If found, return the cached prediction immediately.
4. If not found, run the deployable component models:
   - Stage 1 Extra Trees
   - Stage 3 sampled R-GCN
   - Stage 3 sampled HGT
5. Apply the locked production ensemble, Platt calibration, and validation-selected threshold.
6. Atomically append the newly generated component scores and final prediction metadata to the CSV.
7. Update the in-memory index so the same pair is served from cache on the next request.

The Streamlit landing page and common analysis interface are preserved. The prediction output uses only these result tabs:

- **Prediction**
- **Model Explanation**
- **Evidence**
- **Download Report**

No prediction-specific landing-page hero or extra main-page cards are added.

## Files to replace

Copy these files into the project while preserving their relative paths:

```text
modeling/Dockerfile
modeling/requirements.txt
modeling/pring_modeling/live_prediction.py
modeling/pring_modeling/prediction_service.py
modeling/pring_modeling/prediction_api.py
modeling/pring_modeling/pyg_runtime.py
streamlit/app.py
streamlit/utils/ui_utils.py
streamlit/utils/prediction_ui.py
streamlit/utils/prediction_client.py
docker-compose.yml
docker-compose.production.yml
```

Merge `hybrid_prediction.env.snippet` into the existing `.env`. Do not replace the existing `.env`, because it contains local paths and credentials.

## Required artifact locations

```text
artifacts/
├── models/
│   ├── production/
│   │   ├── production_ensemble.joblib
│   │   ├── manifest.json
│   │   ├── component_feature_importance.csv
│   │   ├── stage1_feature_importance.csv
│   │   ├── per_target_metrics.csv
│   │   └── explainability_background.csv
│   └── improved_v2/
│       ├── stage1_gds_extra_trees/
│       │   ├── stage1_tabular_extra_trees.joblib
│       │   └── feature_columns.json
│       ├── stage3_rgcn_sampled/
│       │   ├── best_model.pt
│       │   └── rgcn_sampled_metadata.json
│       └── stage3_hgt_sampled/
│           ├── best_model.pt
│           └── hgt_sampled_metadata.json
├── modeling_prepared/
│   └── Stage 3 HeteroData export, node_mapping.csv, edges and features
└── results/
    └── production/
        └── finalized_training_frame.csv
```

The Neo4j `Compound` and `Protein` nodes must retain the exact FastRP property used to train Stage 1. The default is `pringFastRP`.

## Writable result mount

The updated Compose file mounts:

```yaml
- ./artifacts/results:/results
```

Do not change it back to `:ro`. The predictor needs write access to update:

```text
/results/production/finalized_training_frame.csv
```

## Scientific data-integrity rule

New online predictions are not observed training labels. They are appended with:

```text
final_split=production_inference
record_type=production_prediction_cache
exclude_from_training=true
label=<empty>
```

Any future training, validation, or test code must filter them out. For example:

```python
training_rows = frame[
    frame["exclude_from_training"].astype(str).str.lower().ne("true")
]
```

The original train/validation/test rows remain unchanged. Before the first write, the service creates a timestamped backup next to the CSV unless `PREDICTION_CACHE_BACKUP=false`.

## Memory behavior

The predictor:

- loads one shared Stage 3 graph for both models;
- scores all cache-miss pairs as a batch;
- loads R-GCN, scores, and unloads it;
- then loads HGT, scores, and unloads it;
- runs with one Uvicorn worker;
- serializes memory-intensive Stage 3 inference.

This avoids holding two large Stage 3 checkpoints and two graph copies simultaneously.

If the shared graph itself keeps memory too high after requests, set:

```env
PREDICTION_UNLOAD_SHARED_GRAPH_AFTER_SCORE=true
```

This lowers retained memory but makes every new cache miss reload the graph.

## Rebuild

```powershell
cd A:\Repositories\PRING-APP

docker compose stop predictor streamlit
docker compose rm -f predictor streamlit

docker compose build --no-cache predictor streamlit
docker compose up -d --force-recreate predictor
```

Wait for liveness:

```powershell
docker compose ps predictor
Invoke-RestMethod http://localhost:8000/live
```

Check full status:

```powershell
Invoke-RestMethod http://localhost:8000/health |
    ConvertTo-Json -Depth 30
```

The important fields are:

```text
status=ready
hybrid_ready=true
precomputed_score_store.writable=true
live_inference.ready=true
```

Then start Streamlit:

```powershell
docker compose up -d --force-recreate streamlit
```

## Test the cache behavior

Choose one compound-target pair already present in the CSV. It should return quickly with a precomputed score source.

Then choose one graph pair that is absent from the CSV. The first request should report:

```text
score_source=live_component_inference
prediction_cache.status=written
```

Verify the new row:

```powershell
Import-Csv .\artifacts\results\production\finalized_training_frame.csv |
    Where-Object { $_.record_type -eq "production_prediction_cache" } |
    Select-Object -Last 10
```

Request the same pair again. It should report:

```text
score_source=live_component_inference_cached
live_predictions_generated=0
```

## Validation performed

- Python compilation for all replacement modules.
- YAML parsing for both Compose files.
- Unit test confirming:
  - precomputed-first behavior;
  - live fallback for a cache miss;
  - atomic CSV persistence;
  - prediction-only row markers;
  - cache reuse without a second live-model call.
- Comparison confirming that the original Streamlit welcome-page function is unchanged.
