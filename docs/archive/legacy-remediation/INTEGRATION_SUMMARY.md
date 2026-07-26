# Integration summary

## New application capability

The Streamlit sidebar now includes **Predict Missing Interaction**. Selected compound-target pairs are sent to a dedicated FastAPI predictor service and displayed through four result tabs:

1. Prediction Results
2. Explainability
3. Evidence
4. Prediction Report

## Production model decision

The original best research ensemble cannot be reproduced for a new pair because two of its Stage 1 inputs are holdout/CV evaluation predictions rather than deployable model artifacts. A three-component production ensemble was therefore trained from reproducible inputs:

- Stage 1 Extra Trees
- Stage 3 R-GCN
- Stage 3 HGT

It retains Platt calibration and a validation-selected MCC threshold.

## Included validated production metrics

- MCC: 0.9154
- Balanced accuracy: 0.9675
- ROC-AUC: 0.9787
- Average precision: 0.9936
- Specificity: 0.9493
- Recall: 0.9857
- Brier score: 0.0194
- Expected calibration error: 0.0197
- Threshold: 0.2382

## Explainability included

- TreeSHAP for the raw ensemble output
- Local leave-one-component-out effect on calibrated probability
- Global component importance
- Stage 1 FastRP feature importance
- Decision margin
- Component disagreement
- Predictive entropy
- Brier score and expected calibration error
- Global and target-specific test metrics
- Neo4j evidence tier and analogue support

## Runtime modes

- `precomputed`: works immediately for pairs in the supplied finalized score frame.
- `auto`: uses precomputed scores when available and live model inference otherwise.
- `live`: requires all Stage 1, R-GCN, HGT and Stage 3 prepared-graph artifacts.

## Validation performed

- Production bundle rebuilding
- Active and inactive pair inference
- Explicit error for unavailable pairs in precomputed mode
- TreeSHAP generation
- HTML/JSON/CSV report generation
- FastAPI `/health` and `/predict` endpoint tests
- Python compilation checks
