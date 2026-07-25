# Model artifacts required for fully live prediction

The uploaded results intentionally excluded the large serialized models and Stage 3 graph tensors. The integrated app is therefore immediately testable for pairs already present in the supplied finalized score frame, but true unseen-pair inference becomes available only after the artifacts below are copied from the original modeling environment.

## Stage 1

- `stage1_tabular_extra_trees.joblib`
- `feature_columns.json`
- The same Neo4j FastRP/GDS embedding property used to create the training pair features.

## Stage 3 R-GCN

- `best_model.pt`
- `rgcn_sampled_metadata.json`

## Stage 3 HGT

- `best_model.pt`
- `hgt_sampled_metadata.json`

## Shared Stage 3 prepared graph

- HeteroData/PyG graph export.
- `node_mapping.csv`.
- Compound, protein, endpoint and other node features.
- Typed edge indexes matching the checkpoint metadata.

## Production ensemble

Already included in `artifacts/models/production`:

- `production_ensemble.joblib`
- `manifest.json`
- explainability background and importance tables.

Run `GET /health` on the predictor service to verify every artifact before enabling live prediction in a deployed app.
