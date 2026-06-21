# PRING modeling implementations

This folder contains three selectable modeling implementations:

- `legacy` — original implementation, kept unchanged for reproducibility.
- `improved` — first improved implementation with leakage-safe Stage 1, imbalance-aware thresholds, Stage 3 tuning, and stacked ensemble.
- `improved_v2` — finalization implementation with the previous improvements plus leakage-aware final validation, common-test evaluation, calibration, uncertainty outputs, per-target metrics/models, seed aggregation, candidate ranking, external-validation hooks, and HPO planning.

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
cd /home/asmaaali/KG-ADMET-Predictor

PROJECT_DIR=/home/asmaaali/KG-ADMET-Predictor \
MODEL_ROOT=/home/asmaaali/KG-ADMET-Predictor/modeling \
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

For a stricter publishable final ensemble, use only base prediction files that already contain explicit train/valid/test split annotations and add:

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
MODEL_IMPL=improved_v2 bash modeling/scripts/use_implementation.sh improved_v2
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
