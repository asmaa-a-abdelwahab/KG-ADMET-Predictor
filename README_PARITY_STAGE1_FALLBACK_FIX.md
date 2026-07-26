# PRING live-parity Stage 1 correction and validated Stage 3 fallback

## Why parity failed

The reported parity result is scientifically useful:

- R-GCN parity passed exactly.
- HGT parity passed within a small error.
- Final class decisions agreed for all sampled pairs.
- Stage 1 Extra Trees did not reproduce its validated component score.

The Stage 1 model consumes five pair features derived from the training-time
`pringFastRP` node vectors. FastRP coordinates are tied to the exact graph
projection, graph version, random seed and GDS settings. A Neo4j database that
was rematerialized or had FastRP regenerated can contain a property with the
same name but different coordinates. That changes dot product, cosine, L2 and
absolute-difference features and therefore changes the Stage 1 score.

Do not disable parity and do not merely increase the MAE threshold.

## What this update does

1. Looks for the exact Stage 1 pair-feature exports used during modeling.
2. Uses those rows before attempting to derive features from Neo4j embeddings.
3. Continues to require strict Stage 1 parity for the primary three-component
   ensemble.
4. Adds a separately trained and calibrated R-GCN + HGT fallback ensemble.
5. Uses the fallback only when both Stage 3 components pass parity.
6. Stores `model_variant` in the production cache so cached probabilities are
   always interpreted by the bundle that generated them.
7. Ignores legacy cache rows that do not contain model-variant provenance.

## Fallback validation results

The included Stage 3 fallback was built from the same finalized seed-5 frame.

| Metric | Frozen test result |
|---|---:|
| MCC | 0.9103 |
| Balanced accuracy | 0.9608 |
| ROC-AUC | 0.9768 |
| Average precision | 0.9919 |
| Specificity | 0.9348 |
| Recall | 0.9867 |
| Brier score | 0.0199 |
| ECE | 0.0187 |
| Threshold | 0.1620 |

The primary three-component production ensemble remains preferred when exact
Stage 1 parity passes. The fallback is slightly weaker but is directly
reproducible from the Stage 3 components that passed the user's parity check.

## Files to replace

```text
modeling/pring_modeling/live_prediction.py
modeling/pring_modeling/prediction_service.py
modeling/pring_modeling/prediction_store.py
docker-compose.yml
docker-compose.production.yml
```

## New files to copy

```text
artifacts/models/production/production_stage3_fallback.joblib
artifacts/models/production/stage3_fallback_manifest.json
modeling/scripts/build_stage3_fallback_bundle.py
parity_stage1_fallback.env.snippet
diagnose_parity_fix.ps1
```

## Optional exact Stage 1 feature files

Copy the original prepared Stage 1 exports into:

```text
artifacts/modeling_prepared/stage1_neo4j_gds_baselines/
├── compound_target_training_pairs_gds_features.csv
└── candidate_pairs_gds_features.csv
```

The training file is used to reproduce parity pairs. The candidate file is used
for new graph pairs that were already materialized as Stage 1 candidates.

When these files are available and match the saved Stage 1 model, parity should
return:

```json
{
  "status": "passed",
  "active_live_variant": "primary"
}
```

When the exact Stage 1 features are unavailable but R-GCN and HGT pass, the
expected result is:

```json
{
  "status": "passed_with_stage3_fallback",
  "active_live_variant": "stage3_fallback",
  "primary_passed": false,
  "stage3_fallback_passed": true
}
```

## Apply

Extract the patch outside the repository and copy the files while preserving
relative paths, or use the included `apply_patch.ps1`.

Merge `parity_stage1_fallback.env.snippet` into the existing `.env`.

## Rebuild

```powershell
cd A:\Repositories\KG-ADMET-Predictor

docker compose stop predictor
docker compose rm -f predictor
docker compose build --no-cache predictor
docker compose up -d --force-recreate predictor
```

Streamlit does not need rebuilding for this correction.

## Validate

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/validate-live-parity?force=true" |
  ConvertTo-Json -Depth 40
```

Or:

```powershell
powershell -ExecutionPolicy Bypass -File .\diagnose_parity_fix.ps1
```

## Cache behavior

New fallback rows contain:

```text
model_variant=stage3_fallback
model_version=stage3-fallback-v1
active_score_columns_json=[R-GCN, HGT]
```

Legacy rows created before model-variant provenance was added are ignored and
recomputed. The finalized training frame remains read-only and is not modified.
