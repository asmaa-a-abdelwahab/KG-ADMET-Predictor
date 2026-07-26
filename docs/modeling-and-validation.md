# Modeling pipeline

## Split discipline

The authoritative split is created before model fitting and shared across
components. Compound or similarity groups must not cross outer partitions.

```mermaid
flowchart LR
    Registry["Registered outer split"] --> Train["Training folds"]
    Train --> OOF["Out-of-fold component scores"]
    Registry --> Valid["Validation partition"]
    Valid --> Select["Model, calibration, threshold selection"]
    Registry --> Test["Locked test partition"]
    Select --> Locked["Locked pipeline"]
    Locked --> Test
    Test --> Report["Final metrics and uncertainty"]
```

## Selection rules

- Fit preprocessing and models on training data only.
- Create stacking features out of fold for training rows.
- Select hyperparameters and seeds using training/CV or validation evidence.
- Fit calibration and select the operating threshold on validation data.
- Evaluate the locked test partition once.

The simple ensemble and generated re-splits are marked `diagnostic_only`.

## Production bundle gate

Bundle creation rejects:

- all-component held-out scores that are later re-split;
- duplicate compound–target pairs;
- compounds crossing final partitions;
- component/final split disagreement;
- missing train, validation, or test partitions.

New bundles record runtime versions, training-frame hash, split and feature
identity, model version, and model-file digest.

## Required evaluation

Report discrimination, class-balanced metrics, calibration, uncertainty,
per-target results, abstentions, and applicability-domain behavior. Include an
external or temporal validation set whenever the scientific claim extends
beyond the sampled graph.

