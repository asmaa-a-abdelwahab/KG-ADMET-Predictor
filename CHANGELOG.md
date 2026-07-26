# Changelog

## Unreleased

- Add a searchable GitHub Pages documentation site and automated deployment.
- Add application CI, contribution guidance, security reporting, citation
  metadata, and consistent text-file rules.
- Make production-artifact integration tests skip cleanly when large local
  artifacts are unavailable.
- Remove committed build products and ignore future generated documentation
  and coverage output.
- Move one-off prediction patch instructions and diagnostic scripts into an
  excluded documentation archive.

## 0.2.0 — 2026-07-26

- Align application, compose, container, workspace, documentation, and package
  metadata with `PRING-APP` and `PRING-PACKAGE`.
- Select thresholds, calibration, and best seeds without test-set optimization.
- Reject diagnostic re-splits when building production bundles.
- Add read-only audits for preserved modeling results.
- Add version-safe prediction caching and inference-only Neo4j provenance.
- Correct evidence-path traversal and distinguish active, inactive, and
  conflicting direct assertions.
- Add optional prediction API authentication, safe error handling, localhost
  binding, stricter Neo4j defaults, and production configuration gates.
- Use a dedicated predictor image with a verified pinned PyTorch/PyG stack.
- Replace destructive implementation switching with `PYTHONPATH` selection.

## 0.1.1

- Previous application and modeling implementation.
