# PRING Streamlit Missing-Interaction Prediction Integration

This package adds a **Predict Missing Interaction** action to the existing CYP450-KG Explorer. The prediction workflow is isolated in a dedicated `predictor` Docker service so the Streamlit UI does not need to load PyTorch/PyG checkpoints directly.

## What was implemented

- New Streamlit action: **Predict Missing Interaction**.
- Deployable PRING production ensemble using only reproducible inference components:
  - Stage 1 leakage-safe Extra Trees score.
  - Stage 3 sampled R-GCN score.
  - Stage 3 sampled HGT score.
- Platt probability calibration and an MCC-selected validation threshold.
- FastAPI prediction service with `/health`, `/model`, and `/predict` endpoints.
- Pair-level report with:
  - calibrated probability and class decision;
  - model/component scores;
  - local leave-one-component-out contributions on the calibrated probability;
  - TreeSHAP explanation of the raw Extra Trees ensemble output;
  - global ensemble feature importance;
  - Stage 1 FastRP pair-feature importance;
  - decision margin, model disagreement, predictive entropy, Brier score, and expected calibration error;
  - global and CYP-specific validation metrics;
  - Neo4j direct-evidence, similar-compound support, endpoint paths, and evidence tier;
  - downloadable HTML, JSON, and CSV outputs.

## Why a separate production ensemble was created

The research `improved_v2/finalized_v2` ensemble uses five inputs, including three Stage 1 columns:

- `score__stage1_tabular_extra_trees_holdout`
- `score__stage1_tabular_extra_trees_cv`
- `score__stage1_tabular_extra_trees`

The holdout and cross-validation columns are evaluation predictions, not separately serialized models that can be executed for a new pair. Feeding duplicate Stage 1 scores into these columns at deployment would not reproduce the validated model and would be scientifically invalid.

The production bundle therefore uses only three scores that can be generated for a new pair:

1. `score__stage1_tabular_extra_trees`
2. `score__stage3_rgcn_sampled`
3. `score__stage3_hgt_sampled`

On the supplied seed-5 frozen test partition, this deployable ensemble achieved:

| Metric | Value |
|---|---:|
| MCC | 0.9154 |
| Balanced accuracy | 0.9675 |
| ROC-AUC | 0.9787 |
| Average precision | 0.9936 |
| Specificity | 0.9493 |
| Recall | 0.9857 |
| Brier score | 0.0194 |
| Expected calibration error | 0.0197 |
| Selected threshold | 0.2382 |

The research five-column ensemble remains the best reported evaluation result (MCC 0.9229), while the three-column production ensemble is the strongest directly deployable version created from the supplied outputs.

## Prediction modes

### 1. `auto` — recommended

- Uses exact component outputs from `finalized_training_frame.csv` when the selected pair is already present.
- Uses live Stage 1/R-GCN/HGT inference for pairs not present in the frame when all required artifacts are mounted.

```env
PREDICTION_SCORE_MODE=auto
```

### 2. `precomputed`

Only pairs present in the supplied finalized score frame can be returned. This mode is useful for validating the UI and reports without loading the large model checkpoints.

```env
PREDICTION_SCORE_MODE=precomputed
```

### 3. `live`

Forces new inference through all three component models. The request fails rather than silently substituting precomputed or duplicated scores when a required artifact is missing.

```env
PREDICTION_SCORE_MODE=live
PREDICTION_DEVICE=cuda
```

## Required live artifacts

Copy the actual large artifacts from the original modeling run into these locations:

```text
artifacts/models/
├── production/
│   ├── production_ensemble.joblib
│   ├── manifest.json
│   ├── component_feature_importance.csv
│   ├── explainability_background.csv
│   ├── stage1_feature_importance.csv
│   └── per_target_metrics.csv
└── improved_v2/
    ├── stage1_gds_extra_trees/
    │   ├── stage1_tabular_extra_trees.joblib
    │   └── feature_columns.json
    ├── stage3_rgcn_sampled/
    │   ├── best_model.pt
    │   └── rgcn_sampled_metadata.json
    └── stage3_hgt_sampled/
        ├── best_model.pt
        └── hgt_sampled_metadata.json

artifacts/modeling_prepared/
└── stage3_advanced_models or the Stage 3 folder resolved by PRING
    ├── HeteroData/PyG export files
    ├── node_mapping.csv
    └── graph edge and node feature files
```

The connected Neo4j graph must also retain the GDS embedding property used by Stage 1. The default is:

```env
STAGE1_EMBEDDING_PROPERTY=pringFastRP
```

Set this variable to the exact property name used when `compound_target_training_pairs_gds_features.csv` was generated.

## Start the stack

```bash
docker compose build predictor streamlit
docker compose up -d neo4j predictor streamlit
```

Check readiness:

```bash
curl http://localhost:8000/health
```

The response distinguishes:

- production ensemble readiness;
- precomputed-score availability;
- availability of each live component;
- whether the service is fully ready for unseen-pair inference.

Open the app at:

```text
http://localhost:8501
```

Choose **Predict Missing Interaction**, then select one or more compounds and CYP450 targets.

## Rebuild the production bundle

The included production artifact was created from the supplied seed-5 finalized frame. Rebuild it after rerunning modeling:

```bash
python -m pring_modeling.production_bundle \
  --training-frame /results/improved_v2/finalized_v2/seed_5/finalized_training_frame.csv \
  --output-dir /models/production \
  --seed 5 \
  --source-metrics /results/improved_v2/finalized_v2/metrics.json \
  --stage1-feature-importance /results/improved_v2/stage1_gds_extra_trees/feature_importance.csv \
  --per-target-metrics /results/improved_v2/finalized_v2/seed_5/per_target_ensemble_metrics.csv
```

## Interpretation of explainability outputs

- **Local replacement contribution:** change in the final calibrated probability after replacing one component score with its training-background median. This is faithful to the complete deployed pipeline, including calibration.
- **TreeSHAP:** contribution of each component to the uncalibrated Extra Trees ensemble output. It is shown separately because Platt calibration changes the probability scale.
- **Global component importance:** Extra Trees impurity-based importance over the training data.
- **Stage 1 feature importance:** importance of FastRP dot product, cosine similarity, L2 distance, mean absolute difference, and maximum absolute difference in the leakage-safe Stage 1 classifier.
- **Decision margin:** absolute distance between calibrated probability and the locked threshold.
- **Model disagreement:** standard deviation among Stage 1, R-GCN, and HGT component scores.
- **Predictive entropy:** uncertainty of the calibrated binary probability; values close to 1 bit are most uncertain.
- **Brier score and ECE:** frozen-test calibration quality, not pair-specific correctness guarantees.
- **Evidence tier:** Tier 1 direct evidence, Tier 2 similar-compound support, or Tier 3 model-only hypothesis.

## Important scientific limitations

- A predicted interaction is a prioritization hypothesis, not proof of mechanism or causality.
- A direct Neo4j interaction is reported as rediscovery rather than a novel missing interaction.
- The model is restricted to entities represented in the Stage 3 node mapping. A genuinely new compound that was not present when the graph tensors were built requires graph rematerialization or a separate inductive encoder.
- Live predictions require the exact preprocessing, node mapping, feature dimensions, graph schema, and checkpoints used during training.
- Do not substitute missing component scores with zeros, averages, or duplicated Stage 1 values.
