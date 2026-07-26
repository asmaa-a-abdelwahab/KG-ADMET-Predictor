# Prediction service

## Resolution order

1. Immutable validated reference scores.
2. Separate production prediction cache.
3. Parity-guarded live component inference.

Cache identity includes the compound, target, model variant, model version, and
graph version. Unknown or stale versions miss by default.

## Startup gates

Production startup requires:

- a manifest with runtime versions;
- a matching serialized-model SHA-256 digest;
- `status=ready`, `publishable=true`, and dataset, split, feature-schema, and
  label-policy identifiers;
- a compatible Python major/minor and exact scientific-library versions;
- a configured graph version;
- production credentials.

## Live parity

Before live inference is trusted, offline and live component scores are compared
on a stratified reference sample using:

- mean absolute error;
- maximum absolute error;
- Spearman rank correlation;
- decision agreement.

If parity fails, the service must not silently substitute live predictions for
validated offline scores.

The default production gate uses at least 200 stratified pairs with MAE ≤ 0.01,
maximum absolute error ≤ 0.05, Spearman correlation ≥ 0.99, and decision
agreement ≥ 99.5%. A project may impose stricter, scientifically justified
limits.

## Evidence-aware interpretation

Results distinguish known active, known inactive, conflicting, analogue-only,
and model-only evidence. Reports expose the threshold, calibrated probability,
component contributions, applicability diagnostics, heuristic confidence,
provenance, and
scientific limitations.

The confidence band is not an uncertainty interval. It summarizes decision
margin, component dispersion, and predictive entropy and is labeled accordingly
throughout the API, UI, and reports.

## API security

Production endpoints require `PREDICTION_API_KEY`. Error responses use a
correlation ID and omit stack traces, filesystem paths, credentials, and raw
queries unless an explicit local debug mode is enabled.
