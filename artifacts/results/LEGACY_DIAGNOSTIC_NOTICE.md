# Legacy diagnostic modeling results

The retained files below this directory predate the current leakage, provenance,
and scientific-release gates. They are preserved for traceability and regression
testing, not as production or publication evidence.

In particular, the retained production frame re-splits component predictions
whose recorded component partitions are already held out. The audit command
therefore reports `production_ready=false` and `publication_ready=false`.

Do not overwrite these files to make them appear compliant. Generate a new,
content-addressed run and production bundle from registered train/out-of-fold,
validation, and untouched-test predictions.
