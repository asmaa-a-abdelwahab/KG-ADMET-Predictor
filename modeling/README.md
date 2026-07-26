# PRING modeling implementations

This folder contains three selectable modeling implementations:

- `legacy` — original algorithmic implementation, retained for historical comparison.
- `improved` — first improved implementation with leakage-safe Stage 1, imbalance-aware thresholds, Stage 3 tuning, and stacked ensemble.
- `improved_v2` — canonical implementation with validation-selected thresholds/seeds,
  calibrated probabilities, uncertainty outputs, per-target diagnostics, candidate
  ranking, external-validation hooks, and production-serving modules.

The active `modeling/pring_modeling` package and
`implementations/improved_v2/pring_modeling` are kept synchronized. Selection
scripts no longer delete or replace source trees with symlinks; wrappers select
an implementation through `PYTHONPATH`.

Select an implementation by setting `MODEL_IMPL`:

```bash
MODEL_IMPL=legacy sbatch modeling/scripts/run_all_models_compare_hpc.sh
MODEL_IMPL=improved sbatch modeling/scripts/run_all_models_compare_hpc.sh
MODEL_IMPL=improved_v2 sbatch modeling/scripts/run_all_models_compare_hpc.sh
```

The default output folders are separated automatically:

```text
models_all_stages_legacy/      reports/all_stages_legacy/
models_all_stages_improved/    reports/all_stages_improved/
models_all_stages_improved_v2/ reports/all_stages_improved_v2/
```

## Recommended final run

```bash
cd /home/asmaaali/PRING-APP

PROJECT_DIR=/home/asmaaali/PRING-APP \
MODEL_ROOT=/home/asmaaali/PRING-APP/modeling \
MODEL_IMPL=improved_v2 \
MODEL_STAGE2_MODELS="complex distmult rotate" \
RUN_STAGE1=true \
RUN_STAGE2=true \
RUN_STAGE3_RGCN=true \
RUN_STAGE3_HGT=true \
RUN_ENSEMBLE=true \
RUN_FINAL_VALIDATION=true \
RUN_COMPARE=true \
MODEL_PRIMARY_COMPARE_METRIC=mcc \
MODEL_MIN_SPECIFICITY=0.50 \
MODEL_STAGE2_SCORE_BATCH_SIZE=65536 \
MODEL_FINAL_SEEDS="1 2 3 4 5" \
MODEL_FINAL_SPLIT_STRATEGY=compound \
MODEL_FINAL_CALIBRATION=platt \
sbatch modeling/scripts/run_all_models_compare_hpc.sh
```

For a potentially publishable final ensemble, every base score must be produced
out-of-fold under the same registered outer split (or a defensible nested-CV
design). Merely re-splitting held-out base predictions is diagnostic only. Use
the strict guard:

```bash
MODEL_FINAL_STRICT_LEAKAGE_FREE=true
```

## Final validation only

After Stage 1/2/3 have finished, rerun only final validation:

```bash
MODEL_IMPL=improved_v2 \
MODEL_FINAL_SEEDS="1 2 3 4 5" \
MODEL_FINAL_SPLIT_STRATEGY=compound \
sbatch modeling/scripts/run_final_validation_hpc.sh
```

Important final outputs:

```text
models_all_stages_improved_v2/finalized_v2/metrics.json
models_all_stages_improved_v2/finalized_v2/seed_metrics.csv
models_all_stages_improved_v2/finalized_v2/seed_metric_summary.csv
models_all_stages_improved_v2/finalized_v2/seed_<N>/common_test_model_metrics.csv
models_all_stages_improved_v2/finalized_v2/seed_<N>/per_target_metrics.csv
models_all_stages_improved_v2/finalized_v2/seed_<N>/per_target_ensemble_metrics.csv
models_all_stages_improved_v2/finalized_v2/seed_<N>/calibration_bins.csv
models_all_stages_improved_v2/finalized_v2/seed_<N>/top_k_by_target.csv
models_all_stages_improved_v2/finalized_v2/seed_<N>/most_uncertain_predictions.csv
```

## HPO plan generation

```bash
bash modeling/scripts/use_implementation.sh improved_v2
PYTHONPATH=modeling/implementations/improved_v2 \
python -m pring_modeling.hpo_plan \
  --output-dir reports/hpo_plan_improved_v2 \
  --stage all \
  --max-jobs 30
```

This creates:

```text
reports/hpo_plan_improved_v2/hpo_plan.csv
reports/hpo_plan_improved_v2/submit_hpo.sh
```

## Same-split comparison across legacy, improved and improved_v2

For a reliable comparison, generate one canonical split manifest and run all three implementations against the same materialized modeling directory:

```bash
cd /home/asmaaali/PRING-APP

PROJECT_DIR=/home/asmaaali/PRING-APP \
MODEL_ROOT=/home/asmaaali/PRING-APP/modeling \
MODEL_IMPLS="legacy improved improved_v2" \
MODEL_SHARED_SPLIT_STRATEGY=compound \
MODEL_SHARED_SPLIT_SEED=42 \
MODEL_SHARED_TEST_SIZE=0.15 \
MODEL_SHARED_VALID_SIZE=0.15 \
MODEL_STAGE2_MODELS="complex distmult rotate" \
RUN_STAGE1=true \
RUN_STAGE2=true \
RUN_STAGE3_RGCN=true \
RUN_STAGE3_HGT=true \
RUN_ENSEMBLE=true \
RUN_FINAL_VALIDATION=true \
RUN_COMPARE=true \
MODEL_PRIMARY_COMPARE_METRIC=mcc \
sbatch --export=ALL modeling/scripts/run_all_implementations_same_splits_hpc.sh
```

The command creates:

- `shared_splits/<run_id>/split_manifest.csv`
- `shared_splits/<run_id>/split_summary.json`
- `shared_splits/<run_id>/modeling_prepared/` with the same `split`, `split_group` and `stage_use` columns materialized into all supervised pair CSVs
- `models_all_stages_same_splits/<run_id>/legacy/`
- `models_all_stages_same_splits/<run_id>/improved/`
- `models_all_stages_same_splits/<run_id>/improved_v2/`
- `reports/all_stages_same_splits/<run_id>/cross_implementation/cross_implementation_comparison.md`

Stage 1 was updated in all implementations to honor an existing `split` column before falling back to its own random/group split. Stage 2 and Stage 3 already use explicit split columns when present. This makes the comparison across implementations use the same train/validation/test assignments.
