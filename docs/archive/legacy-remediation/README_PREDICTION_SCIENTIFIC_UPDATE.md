# PRING prediction and scientific-report update

This patch implements the requested prediction/report improvements while preserving the existing Streamlit landing page, navigation, common graph functionality and overall design. Only the prediction result tabs remain:

- Prediction
- Model Explanation
- Evidence
- Download Report

## Main implementation changes

### 1. Immutable modeling frame and separate production cache

The validated modeling frame remains read-only:

```text
artifacts/results/production/finalized_training_frame.csv
```

New live predictions are written to:

```text
artifacts/results/production/production_prediction_cache.csv
```

Lookup order:

1. validated finalized frame;
2. production prediction cache;
3. parity-validated live Stage 1/R-GCN/HGT inference.

The service ignores legacy rows marked `record_type=production_prediction_cache` or `final_split=production_inference` if they are still present in the finalized frame.

### 2. Live-versus-precomputed parity gate

Before the first live cache-miss prediction, the service recalculates a target-stratified sample of validated pairs and compares live component scores with the stored outputs. It reports:

- mean absolute error;
- maximum absolute error;
- Spearman correlation;
- final decision agreement.

If parity fails, new predictions are blocked and are not cached. Use:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/validate-live-parity?force=true" |
  ConvertTo-Json -Depth 30
```

### 3. Correct prediction terminology

Results are classified as:

- `known_interaction_rediscovered`;
- `known_interaction_not_rediscovered`;
- `novel_predicted_interaction`;
- `interaction_not_predicted`.

The application no longer describes a below-threshold result as biologically inactive. It says **interaction not predicted at the selected threshold**.

### 4. Model certainty and evidence support are separate

Each result now reports:

- model certainty, based on decision margin, component disagreement, entropy and applicability domain;
- evidence support, based on direct assertions, same-target analogues and provenance completeness.

A high-probability Tier 3 result is therefore clearly presented as high model certainty but low evidence support.

### 5. Target-conditioned explanations

Local replacement explanations use the median component scores for the same CYP target when at least the configured number of validated target rows is available. Otherwise, they use the global validated reference distribution.

Effects are reported in **probability percentage points**. Relative importance percentages are suppressed when the total absolute effect is too small to support a stable ranking.

### 6. Applicability-domain diagnostics

Every component score is compared with target-specific or global validated distributions. The result includes:

- percentile;
- 1st-99th percentile reference range;
- in-domain, borderline or outside-domain status;
- saturation flag for scores close to zero or one.

### 7. Stronger evidence reconstruction

The Neo4j evidence result now includes:

- direct interaction assertions;
- endpoint counts;
- measure-group counts;
- BioAssay counts;
- same-target analogue support;
- source/reference properties when present;
- evidence tier and provenance completeness.

### 8. Improved report

The HTML report now has:

- executive summary across all requested pairs;
- model and graph provenance;
- frozen-test validation metrics shown once;
- probability-versus-threshold visualization;
- component score bars;
- contribution bars in percentage points;
- applicability-domain table;
- direct and analogue evidence tables;
- recommended action per pair;
- explicit scientific limitations.

TreeSHAP is displayed only when computed. A disabled TreeSHAP section is not repeated for every pair.

## Files replaced or added

```text
modeling/Dockerfile                                 replace
modeling/requirements.txt                            replace
modeling/pring_modeling/prediction_store.py          new
modeling/pring_modeling/prediction_service.py        replace
modeling/pring_modeling/prediction_api.py            replace
modeling/pring_modeling/live_prediction.py           replace
modeling/scripts/migrate_prediction_cache.py         new
streamlit/utils/prediction_report.py                 replace
streamlit/utils/prediction_ui.py                     replace
streamlit/utils/neo4j_utils.py                       replace
docker-compose.yml                                   replace
docker-compose.production.yml                        replace
prediction_scientific.env.snippet                    merge into .env
```

## Apply the patch

Extract this package outside the repository, for example:

```text
A:\Patches\PRING_Prediction_Scientific_Report_Update
```

Then run:

```powershell
powershell -ExecutionPolicy Bypass `
  -File "A:\Patches\PRING_Prediction_Scientific_Report_Update\apply_patch.ps1" `
  -PatchRoot "A:\Patches\PRING_Prediction_Scientific_Report_Update" `
  -ProjectRoot "A:\Repositories\PRING-APP"
```

Merge `prediction_scientific.env.snippet` into the existing `.env`. Do not replace the complete `.env`.

## Migrate legacy cached rows

Run a dry run first:

```powershell
docker compose run --rm predictor python `
  /opt/kg/modeling/scripts/migrate_prediction_cache.py `
  --reference-frame /results/production/finalized_training_frame.csv `
  --cache-frame /results/production/production_prediction_cache.csv
```

If legacy rows are found, run with `--apply`:

```powershell
docker compose run --rm predictor python `
  /opt/kg/modeling/scripts/migrate_prediction_cache.py `
  --reference-frame /results/production/finalized_training_frame.csv `
  --cache-frame /results/production/production_prediction_cache.csv `
  --apply
```

The script creates a timestamped backup of the finalized frame before changing it.


## Rebuild

```powershell
cd A:\Repositories\PRING-APP

docker compose stop predictor streamlit
docker compose rm -f predictor streamlit

docker compose build --no-cache predictor streamlit
docker compose up -d --force-recreate predictor
docker compose up -d --force-recreate streamlit
```

Neo4j does not need rebuilding unless its implementation changed separately.

## Validation

Check status:

```powershell
Invoke-RestMethod http://localhost:8000/health |
  ConvertTo-Json -Depth 30
```

Run parity explicitly:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8000/validate-live-parity?force=true" |
  ConvertTo-Json -Depth 30
```

Expected parity status:

```json
{
  "status": "passed",
  "decision_agreement": 1.0
}
```

Monitor memory during the first validation/live request:

```powershell
docker stats pring-app-predictor pring-app-neo4j
```

Start with one compound and one CYP target.

## Important operational boundary

The live scorer supports compounds and proteins already present in the prepared Stage 3 graph and node mapping. A genuinely new compound outside that graph requires graph rematerialization or an inductive molecular encoder.
