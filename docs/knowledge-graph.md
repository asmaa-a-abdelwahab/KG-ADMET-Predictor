# Knowledge graph

## Loading

PRING-APP delegates loading to the PRING-PACKAGE loader. This keeps Neo4j
constraints, node keys, relationship identity, and validation aligned with the
artifacts that created the graph.

## Evidence model

Bioassay endpoint evidence follows:

```text
BioAssay -[:HAS_MEASURE_GROUP]->
MeasureGrp -[:HAS_ENDPOINT]->
Endpoint
```

Direct compound–protein evidence is classified as:

- active;
- inactive;
- conflicting;
- unlabeled.

Predicted links are stored separately from observational assertions and carry:

- model and graph version;
- dataset, split, feature-schema, and label-policy identity;
- creation timestamp;
- `exclude_from_training=true`.

## Query safety

Use parameterized Cypher for user-controlled values. Enforce result limits and
timeouts for exploratory queries. Avoid exposing unrestricted Cypher execution
to anonymous users.

## Graph Data Science

Structural projections used for supervised link prediction must exclude outcome
and evidence-path edges that reveal the held-out interaction. Internal GDS link
prediction is diagnostic unless it follows the same registered outer split and
leakage audit as the primary modeling pipeline.

