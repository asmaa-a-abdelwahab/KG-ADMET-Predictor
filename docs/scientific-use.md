# Scientific use

## Current preserved results

The current `finalized_training_frame.csv` is preserved as historical evidence:

| Property | Value |
|---|---|
| Rows | 5,861 |
| SHA-256 | `80ca1fef03f95c5fb50989790505abbafa5dba2b8f562c839e26a83f3b93ba1d` |
| Final train / validation / test | 3,792 / 952 / 1,117 |
| Component split status | All five component split fields contain held-out/test scores |
| Cache rows in reference frame | 0 |
| Scientific classification | Diagnostic only |

The legacy manifest says `ready`, but it lacks a publishable flag, runtime
record, and model digest. The production validator intentionally rejects it.

## Defensible use

The current artifacts can support:

- software demonstrations;
- graph and UI engineering evaluation;
- historical thesis discussion with explicit limitations;
- hypothesis generation.

They cannot support an unqualified claim of production or publication readiness.

## Publication-grade replacement

A replacement experiment needs:

- one frozen group/scaffold-aware outer split;
- out-of-fold training scores for every ensemble component;
- preprocessing fitted within training folds;
- validation-only selection and calibration;
- one locked outer-test evaluation;
- uncertainty intervals, per-target analysis, calibration, ablations, and
  negative controls;
- external or temporal validation with overlap auditing.

## Interpretation boundary

Generated interactions are computational hypotheses. Reports should not imply
clinical action, causality, or confirmed metabolism without appropriate
experimental and domain validation.

