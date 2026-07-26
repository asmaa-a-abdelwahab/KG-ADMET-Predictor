<div class="hero" markdown>

# Explore, model, and test knowledge-graph hypotheses

PRING-APP loads versioned PRING knowledge graphs into Neo4j, provides an
interactive Streamlit experience, and integrates leakage-aware modeling,
prediction, evidence review, downloads, and reports.

[Run the application](getting-started.md){ .md-button .md-button--primary }
[Understand prediction safeguards](prediction-service.md){ .md-button }

</div>

## Application capabilities

<div class="grid cards" markdown>

-   :material-graph:{ .lg .middle } **Graph exploration**

    ---

    Inspect entities, relationships, evidence paths, schema structure, and
    domain-specific neighborhoods in Neo4j.

-   :material-magnify:{ .lg .middle } **Interactive queries**

    ---

    Search compounds and targets, explore graph context, and download structured
    records without modifying source run artifacts.

-   :material-brain:{ .lg .middle } **Modeling and prediction**

    ---

    Train with registered splits, validate calibrated ensembles, and infer
    unknown compound–target interactions through a guarded prediction API.

-   :material-file-chart:{ .lg .middle } **Transparent reporting**

    ---

    Export JSON, CSV, and HTML reports with model, graph, dataset, split,
    evidence, uncertainty, and scientific-validity provenance.

</div>

## System boundary

PRING-APP consumes runs created by
[PRING-PACKAGE](https://asmaa-a-abdelwahab.github.io/PRING-PACKAGE/).
The package owns collection and graph-data construction. The app owns Neo4j
operations, user interaction, modeling orchestration, prediction serving, and
reporting.

!!! danger "Predictions are hypotheses"
    A calibrated score is not proof of metabolism, inhibition, induction,
    toxicity, efficacy, or clinical safety. Review the underlying evidence,
    applicability domain, validation protocol, and uncertainty before use.

## Current artifact status

The preserved legacy CYP450 result set is intentionally classified as
**diagnostic only**. Its component predictions were generated on held-out rows
and subsequently re-split for the final ensemble. The application can display
those historical predictions with warnings, but production deployment and
publication require a newly registered, out-of-fold training run.

