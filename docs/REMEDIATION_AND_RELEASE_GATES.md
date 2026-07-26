# PRING framework remediation and release gates

This document is the implementation companion to the 2026-07-26 technical
assessment. It covers the repositories `PRING-PACKAGE` and `PRING-APP`.

## Artifact preservation policy

Existing files under `artifacts/results`, `artifacts/models`, generated reports,
and historical run folders are evidence from previous experiments. The
remediation does not rewrite, delete, or silently relabel them.

The current production training frame is retained for reproducibility, but the
result audit identifies it as diagnostic when all component `split__*` columns
are held out and `final_split` later reassigns those rows to train, validation,
and test. The serving application may display legacy predictions, but it must
show the scientific-validity warning. The frame cannot be used to build a new
production bundle unless the explicit diagnostic override is supplied; an
override-created bundle remains non-publishable.

Run the read-only audit:

```bash
python modeling/scripts/audit_modeling_results.py \
  --results-dir artifacts/results \
  --model-dir artifacts/models/production
```

Current preserved artifact audit (2026-07-26):

- `finalized_training_frame.csv`: 5,861 rows; SHA-256
  `80ca1fef03f95c5fb50989790505abbafa5dba2b8f562c839e26a83f3b93ba1d`.
- All five component `split__*` fields contain only held-out/test scores, while
  `final_split` later assigns 3,792 train, 952 validation, and 1,117 test rows.
- The legacy manifest says `ready` but has no publishable flag, runtime record,
  or model digest. The release gate therefore classifies the current assets as
  diagnostic, not production- or publication-ready.
- The reference frame contains zero production-cache rows and remains separate
  from `production_prediction_cache.csv`.

## Implemented critical controls

| Area | Control | Required behavior |
|---|---|---|
| Feature leakage | Identifier-like columns and their missingness masks are excluded from model tensors | IDs remain only in metadata sidecars |
| Graph leakage | Default PyG `HeteroData` contains train-only edges and declares `graph_scope=train_only` | Full graph is an explicitly named diagnostic artifact |
| Stage 3 loading | Unscoped raw `HeteroData` is rejected | Legacy use requires `PRING_ALLOW_UNSCOPED_HETERODATA=true` |
| Evaluation | Thresholds and calibration are fitted on validation data | Test data is evaluated exactly once |
| Seed selection | Best seed is selected by validation MCC | Test MCC cannot select the reported seed |
| Ensemble | Fixed equal weights require common registered validation/test scores; learned stacking requires OOF training scores | Validation selects calibration/threshold; test remains untouched |
| Production bundle | Re-split held-out component predictions are rejected | Override is diagnostic-only and non-publishable |
| Model artifact integrity | New bundles record a SHA-256 digest and exact Python/scientific-library versions | Production rejects digest mismatch, runtime mismatch, and unverified legacy bundles |
| Cache isolation | Cache is separate from the immutable reference frame | Cache rows carry `exclude_from_training=true` |
| Cache identity | Pair, variant, model version, and graph version form the cache identity | Stale/unversioned cache misses by default |
| Prediction graph writes | Predictions have content-derived IDs and model/data/graph provenance | They are never supervised evidence |
| Evidence | BioAssay paths follow `BioAssay-HAS_MEASURE_GROUP-MeasureGrp-HAS_ENDPOINT-Endpoint` | Active, inactive, and conflicting assertions are distinguished |
| API security | Optional shared-key authentication, generic production errors, localhost port binding | Production overlay requires an API key |
| Neo4j security | APOC file import/export disabled and APOC removed from unrestricted procedures | Enable only for a reviewed operational need |
| Implementation selection | No recursive source deletion or symlink replacement | `PYTHONPATH` selects snapshots safely |

The training and predictor images share exact pandas, NumPy, scikit-learn, and
joblib pins. Python compatibility is checked at major/minor granularity because
patch releases preserve the serialization ABI; the full patch version remains
recorded for provenance.

## Required publication-grade retraining

1. Generate a versioned PRING-PACKAGE run and retain its `manifest.json`,
   modeling manifests, schema, source-query configuration, and checksums.
2. Freeze one group/scaffold-aware outer split registry before feature fitting.
   The same compound or similarity component must not cross outer partitions.
3. Generate each base model's ensemble feature as an out-of-fold prediction for
   training rows. Generate validation/test scores only from models that did not
   see those rows, labels, endpoint evidence paths, or derived target links.
4. Fit preprocessing on training folds only. Record every fitted imputer,
   normalizer, vocabulary, feature schema, and graph projection seed.
5. Select hyperparameters, model seed, calibration method, and operating
   threshold without consulting the outer test outcomes.
6. Lock the complete pipeline and run the outer test once. Report confidence
   intervals, per-CYP results, class prevalence, and all failed/abstained cases.
7. Validate on a temporally or externally sourced dataset with deduplication and
   overlap checks against training, validation, test, graph evidence, and
   similarity components.
8. Build the production bundle only from the locked registered artifacts and
   pass live-versus-offline parity tests.

## Validation tests and acceptance criteria

### PRING-PACKAGE

- Unit tests pass on Python 3.10, 3.11, and 3.12.
- Two parallel evidence relationships with different identity/properties both
  survive Neo4j loading; rerunning the same load creates no duplicates.
- No tensor feature column matches identifier metadata rules, including
  projected names such as `molgraph_cid` and `missing_molgraph_cid`.
- `heterodata.pt` reports `graph_scope=train_only`; held-out target interaction
  and evidence-path edges are absent.
- Manifest and split registry IDs are stable across two identical runs.
- A schema conformance test covers every emitted node label, key, relationship,
  source label, and target label.

### Modeling

- Pair, compound/similarity-group, endpoint, and evidence-path overlap is zero
  between registered outer partitions.
- Preprocessing-fit records contain training rows only.
- Perturbing test labels does not change fitted preprocessing, hyperparameters,
  selected seed, calibration, or threshold.
- Perturbing validation labels can change selection but never fitted training
  membership.
- Every stacking training feature is traceable to an out-of-fold prediction.
- Re-running with the same environment lock, manifest, and seed reproduces split
  IDs, predictions, and metrics within declared numeric tolerance.
- Production manifests record exact Python, scikit-learn, NumPy, pandas, and
  joblib versions plus the serialized model SHA-256 digest.
- Calibration is evaluated with reliability plots, Brier score, ECE, and enough
  samples per calibration bin; bootstrap confidence intervals are reported.

### Predictor and application

- Offline and live component scores meet declared MAE, maximum-error, rank, and
  decision-agreement limits on a stratified parity set.
- Cache hits require exact model and graph versions; a version change causes a
  miss and no stale score is returned.
- Predictor startup fails on a model-digest or runtime-version mismatch.
  Production also fails on legacy manifests without either record.
- Reference data remains byte-identical after prediction requests.
- Production prediction relationships carry `exclude_from_training=true` and
  cannot enter package/modeling label materialization.
- API authentication tests cover missing, invalid, and valid keys.
- Error responses contain a correlation ID but no file paths, credentials,
  stack traces, or raw queries when debug mode is off.
- Reports distinguish known active, known inactive, conflicting, analogue-only,
  and model-only evidence and include task/clinical limitations.
- Load, concurrency, restart, cache-corruption, Neo4j-outage, and disk-full tests
  meet documented service-level targets.

### Deployment

- `docker compose config` succeeds for development.
- Production compose refuses missing Neo4j password, prediction API key, and
  graph version.
- Images build from pinned dependency locks and pass vulnerability and secret
  scans with no unresolved critical finding.
- Containers run as non-root where supported, use read-only model/run mounts,
  expose only required ports, and have health/readiness checks.
- Backup/restore and rollback are tested for Neo4j and the prediction cache.

## Release gates

### Thesis submission

- [ ] Research question, estimand/task, label policy, and candidate semantics are explicit.
- [ ] All thesis tables/figures resolve to immutable manifests and scripts.
- [ ] Diagnostic legacy results are clearly separated from defensible results.
- [ ] Leakage, bias, uncertainty, negative-label, and external-validity limitations are discussed.
- [ ] Repeated/nested evaluation and confidence intervals are complete.
- [ ] Independent or temporal validation is complete, or its absence is an explicit limitation.

### Software release

- [ ] Repository names are `PRING-PACKAGE` and `PRING-APP`.
- [ ] Versioned changelog, license, citation file, contributor guidance, and code of conduct are present.
- [ ] Package/app documentation, examples, CLI help, and schema diagrams match implementation.
- [ ] Unit, integration, schema, security, and reproducibility tests pass in CI.
- [ ] Clean-clone installation and one small end-to-end example pass.

### Production deployment

- [ ] A newly retrained, publishable production bundle passes asset validation.
- [ ] API key, Neo4j secret, graph version, TLS, access control, logging, and monitoring are configured.
- [ ] Cache identity and invalidation are verified against the deployed versions.
- [ ] Live/offline parity, load, failure recovery, backup, restore, and rollback pass.
- [ ] Scientific disclaimers and diagnostic provenance are visible in UI and exports.

### Publication

- [ ] The final test was untouched until the locked analysis.
- [ ] Baselines share the same outer splits and evaluation protocol.
- [ ] Hyperparameter/seed/threshold/calibration selection uses no test outcomes.
- [ ] External comparison datasets are deduplicated and overlap-audited.
- [ ] Per-target metrics, uncertainty intervals, calibration, ablations, and negative controls are reported.
- [ ] Code, environment lock, manifests, split registry, model cards, data statement, and allowable derived artifacts are archived.

Until all applicable gates pass, the framework is suitable for controlled
research and thesis engineering evidence, not an unqualified production or
publication-ready scientific predictor.
