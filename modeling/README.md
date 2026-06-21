# PRING / KG-ADMET Modeling — Dual Implementation Layout

This `modeling` folder contains both implementations:

```text
modeling/
├── implementations/
│   ├── legacy/      # original/old implementation
│   └── improved/    # performance-improved implementation
├── scripts/         # wrapper scripts that select legacy or improved
├── pring_modeling -> implementations/<active>/pring_modeling
├── stages         -> implementations/<active>/stages
└── pyproject.toml -> implementations/<active>/pyproject.toml
```

The active implementation is selected with:

```bash
bash modeling/scripts/use_implementation.sh improved
# or
bash modeling/scripts/use_implementation.sh legacy
```

After activation, normal commands such as the following work against the selected implementation:

```bash
python -m pip install -e ./modeling
python -m pring_modeling.stage1_tabular --help
```

## Run all models with the improved implementation

```bash
cd /home/asmaaali/KG-ADMET-Predictor

MODEL_IMPL=improved \
MODEL_STAGE2_MODELS="complex distmult rotate" \
RUN_STAGE1=true \
RUN_STAGE2=true \
RUN_STAGE3_RGCN=true \
RUN_STAGE3_HGT=true \
RUN_ENSEMBLE=true \
RUN_COMPARE=true \
MODEL_PRIMARY_COMPARE_METRIC=mcc \
MODEL_MIN_SPECIFICITY=0.50 \
sbatch modeling/scripts/run_all_models_compare_hpc.sh
```

By default, improved results are written to:

```text
/home/asmaaali/KG-ADMET-Predictor/models_all_stages_improved
/home/asmaaali/KG-ADMET-Predictor/reports/all_stages_improved
```

## Run all models with the legacy implementation

```bash
cd /home/asmaaali/KG-ADMET-Predictor

MODEL_IMPL=legacy \
MODEL_STAGE2_MODELS="complex distmult rotate" \
RUN_STAGE1=true \
RUN_STAGE2=true \
RUN_STAGE3_RGCN=true \
RUN_STAGE3_HGT=true \
RUN_COMPARE=true \
sbatch modeling/scripts/run_all_models_compare_hpc.sh
```

By default, legacy results are written to:

```text
/home/asmaaali/KG-ADMET-Predictor/models_all_stages_legacy
/home/asmaaali/KG-ADMET-Predictor/reports/all_stages_legacy
```

## Run only selected stages

Use the same wrapper script and set stage switches:

```bash
MODEL_IMPL=improved \
RUN_STAGE1=true \
RUN_STAGE2=false \
RUN_STAGE3_RGCN=false \
RUN_STAGE3_HGT=false \
RUN_COMPARE=false \
sbatch modeling/scripts/run_all_models_compare_hpc.sh
```

or run a specific stage script:

```bash
MODEL_IMPL=legacy sbatch modeling/scripts/run_stage2_fast_no_neo4j.sh
MODEL_IMPL=improved sbatch modeling/scripts/run_stage3_sampled_gpu_no_neo4j.sh
```

## Override output directories

The wrapper keeps old and improved outputs separate by default. To override manually:

```bash
MODEL_IMPL=improved \
MODEL_OUTPUT_DIR=/home/asmaaali/KG-ADMET-Predictor/models_all_stages_custom \
MODEL_REPORT_DIR=/home/asmaaali/KG-ADMET-Predictor/reports/all_stages_custom \
sbatch modeling/scripts/run_all_models_compare_hpc.sh
```

## Notes

- `MODEL_IMPL=legacy`, `MODEL_IMPL=old`, `MODEL_IMPLEMENTATION=legacy`, and `MODEL_IMPLEMENTATION=old` select the old implementation.
- `MODEL_IMPL=improved`, `MODEL_IMPL=new`, `MODEL_IMPLEMENTATION=improved`, and `MODEL_IMPLEMENTATION=new` select the improved implementation.
- The wrapper automatically activates the selected implementation before running.
- The improved implementation adds imbalance-aware diagnostics, specificity-constrained thresholding, per-target metrics, optional Stage 1 molecular descriptors, improved Stage 2 diagnostics, improved Stage 3 defaults, and ensemble support.
