# PRING-PACKAGE integration

PRING-APP expects a selected package run rather than raw source downloads.

## Required contract

- run manifest and checksums;
- graph JSONL records and schema;
- validation summary;
- CSV mirrors where required by operational tools;
- modeling manifest, split registry, pair tables, mappings, and train-only
  graphs for ML workflows.

## Compatibility rules

1. Read identity and scope from manifests; do not infer them from directory
   names.
2. Fail when required schema or split fields are missing.
3. Keep package runs read-only after validation.
4. Tie Neo4j snapshots and prediction-cache keys to the selected run identity.
5. Never merge live predictions back into package evidence or training labels.

## Local package installation

Containers prefer the mounted `PRING_PROJECT_DIR`. If it is unavailable,
`PRING_PACKAGE_SPEC` can point to the versioned GitHub repository archive.
Production deployments should pin a tag or immutable commit instead of a moving
branch.

