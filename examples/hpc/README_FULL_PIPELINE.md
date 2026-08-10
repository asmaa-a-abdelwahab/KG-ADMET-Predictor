# Complete five-CYP Slurm workflow

`04_full_cyp450_pipeline.sbatch` is the canonical end-to-end implementation
and Slurm entry point for:

1. collecting or reusing PRING source data;
2. creating a new, non-destructive modeling-ready run;
3. generating the EDA report;
4. loading that run into a dedicated Neo4j database;
5. creating outcome-safe FastRP pair features from training-positive and
   label-independent similarity relationships;
6. training Stage 1, ComplEx, DistMult, RotatE, R-GCN, and HGT;
7. running fixed-mean, Platt-calibrated, multi-seed final validation; and
8. producing machine-readable and Markdown readiness reports.

The workflow uses the five principal CYP targets:

| Enzyme | UniProt accession |
|---|---|
| CYP3A4 | P08684 |
| CYP1A2 | P05177 |
| CYP2C19 | P33261 |
| CYP2C9 | P11712 |
| CYP2D6 | P10635 |

## 1. Cluster prerequisites

- Slurm;
- Python 3.10 or newer;
- a CUDA-compatible PyTorch and PyTorch Geometric installation for the full
  R-GCN/HGT run;
- a dedicated Neo4j 5 database with the compatible Graph Data Science plugin;
- a compute node allowed to contact the configured Neo4j Bolt endpoint; and
- internet access for a new PubChem run, or an existing run supplied through
  `PRING_SOURCE_RUN_DIR`.

Do not use a shared or previously populated Neo4j database. Extra nodes alter
FastRP coordinates and therefore invalidate offline/online model parity. The
script refuses a non-empty database unless `ALLOW_NONEMPTY_NEO4J=true` is
explicitly set; that override is not recommended for a final experiment.

## 2. Prepare the Python environment once

Run this on the login/build node, adapting the module and CUDA commands to the
cluster:

```bash
module load python/3.11 cuda/12.4

python -m venv /home/USER/venvs/pring-hpc
source /home/USER/venvs/pring-hpc/bin/activate
python -m pip install --upgrade pip

python -m pip install -e "/home/USER/PRING-Framework/PRING-PACKAGE[analysis]"
python -m pip install \
  -r /home/USER/PRING-Framework/PRING-APP/modeling/requirements.txt
python -m pip install \
  -e /home/USER/PRING-Framework/PRING-APP/modeling
```

Install PyTorch for the cluster's CUDA version, followed by matching PyG wheels.
For a CUDA 12.4 environment, the repository provides:

```bash
python -m pip install \
  -r /home/USER/PRING-Framework/PRING-APP/modeling/requirements-pyg-cu124.txt
```

Verify the environment before submitting:

```bash
/home/USER/venvs/pring-hpc/bin/python - <<'PY'
import torch
import torch_geometric
import pring
import pring_modeling

print("torch", torch.__version__)
print("torch-geometric", torch_geometric.__version__)
print("CUDA available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU", torch.cuda.get_device_name(0))
PY
```

`BOOTSTRAP_ENV=true` can create and populate the Python environment inside the
job. Set `BOOTSTRAP_TORCH_PROFILE=cpu` or `cu124` only when that exact runtime
matches the node; leave it at `none` for a preinstalled cluster-specific
PyTorch/PyG build. A prebuilt environment is recommended.

Final runs require both repositories to be clean Git checkouts by default.
Commit and push the exact implementation before submission. A deliberately
non-final smoke run may set `REQUIRE_CLEAN_GIT=false`, but its performance must
not be reported as reproducible final evidence.

## 3. Configure without committing secrets

Copy the template outside the repository:

```bash
cp examples/hpc/full_pipeline.env.example \
  /secure/path/pring-five-cyp.env
chmod 600 /secure/path/pring-five-cyp.env
```

Edit paths, modules, and resource settings. Do not write `NEO4J_PASSWORD` into
the repository. Inject it using the site's secret facility, or export it in the
protected submission shell:

```bash
set -a
source /secure/path/pring-five-cyp.env
set +a

read -r -s -p "Neo4j password: " NEO4J_PASSWORD
export NEO4J_PASSWORD
```

If compute nodes cannot access PubChem, first create or transfer a complete
source run, then set:

```bash
export PRING_SOURCE_RUN_DIR=/project/pring/runs/source_run
```

The source run is read but never modified. A new `_ready` run is created by
`pring load-run`, so the original evidence remains available for audit.
The script rejects reused sources whose quality report does not confirm an
uncapped run, all candidate pairs, and pipeline-validation readiness without
blockers.

## 4. Submit

Create a unique run and output identifier for every final experiment:

```bash
export PRING_RUN_ID=cyp450_final_20260726
export PRING_READY_RUN_ID=cyp450_final_20260726_ready
export PIPELINE_OUTPUT_ROOT=/scratch/USER/pring/results/cyp450_final_20260726

sbatch --export=ALL examples/hpc/04_full_cyp450_pipeline.sbatch
```

Cluster-specific scheduler settings can be supplied at submission:

```bash
sbatch \
  --account=YOUR_ACCOUNT \
  --partition=gpu \
  --gres=gpu:1 \
  --cpus-per-task=16 \
  --mem=250G \
  --time=7-00:00:00 \
  --export=ALL \
  examples/hpc/04_full_cyp450_pipeline.sbatch
```

Monitor without modifying the run:

```bash
squeue -u "$USER"
sacct -j JOB_ID \
  --format=JobID,JobName,State,Elapsed,MaxRSS,AllocCPUS,ReqMem,ExitCode
tail -f "$PIPELINE_OUTPUT_ROOT/logs/pipeline.log"
```

## 5. Outputs

The prepared PRING run is written under:

```text
PRING_RUN_ROOT/PRING_READY_RUN_ID/
├── manifest.json
└── graph/ml/modeling/
    ├── modeling_stage_manifest.json
    ├── stage1_neo4j_gds_baselines/
    │   ├── stage1_outcome_safe_gds_summary.json
    │   ├── compound_target_training_pairs_gds_features.csv
    │   └── candidate_pairs_gds_features.csv
    ├── stage2_kg_embedding_baselines/
    └── stage3_heterogeneous_gnn/
```

The independent pipeline output directory contains:

```text
PIPELINE_OUTPUT_ROOT/
├── PIPELINE_SUMMARY.md
├── logs/
│   └── pipeline.log
├── models/
│   ├── stage1_gds_extra_trees/
│   ├── stage2_*_supervised/
│   ├── stage3_rgcn_sampled/
│   ├── stage3_hgt_sampled/
│   └── finalized_v2/
│       ├── metrics.json
│       ├── seed_metrics.csv
│       └── seed_metric_summary.csv
├── reports/
│   ├── eda/
│   │   ├── eda_report.html
│   │   ├── eda_report.md
│   │   ├── eda_summary.json
│   │   ├── figures/
│   │   └── tables/
│   └── modeling/comparison/
├── readiness/
│   ├── prepared_readiness.json
│   ├── prepared_readiness.md
│   ├── final_readiness.json
│   └── final_readiness.md
└── status/
    ├── run_environment.txt
    ├── *.complete
    └── pipeline.complete
```

The final readiness validator requires:

- content-addressed dataset, feature-schema, split-registry, and label-policy
  identifiers;
- an uncapped PRING quality report with all candidate pairs and no
  pipeline-readiness blockers;
- the cold-compound, train-only graph scope;
- explicit exclusion of predictions from training;
- all five CYP accessions;
- train, validation, and locked-test partitions;
- no duplicate supervised pair or split-group overlap;
- EDA reports, tables, and figures;
- Stage 1, KGE, R-GCN, HGT, and multi-seed final metrics;
- separate completed metrics for ComplEx, DistMult, and RotatE;
- at least five finalized seeds with common-test, calibration, per-target,
  uncertainty, and ranking artifacts for each seed;
- a Stage 1 GDS audit proving that held-out outcome relationship types were
  excluded and that dataset/split identifiers match;
- a successful strict leakage gate; and
- `publishable=true` from the corrected final-validation implementation.

`publishable=true` is a software/artifact gate, not proof of biological,
clinical, or external validity. Because the fixed ensemble includes the
outcome-safe FastRP component, the ensemble protocol is transductive with
respect to node presence and the compound-similarity graph. It must not be
reported as a purely inductive cold-compound result. Predictions remain
computational hypotheses.

## 6. Scientific execution rules

- Freeze the label policy, primary metric, specificity constraint, seeds,
  candidate policy, and data sources before opening the locked-test results.
- Do not use a failed or partially skipped SPARQL extraction for a final run.
  The script sets `--sparql-skip-failed-chunks false`.
- Do not add validation/test outcome edges to the training graph.
- Describe Stage 1 FastRP as outcome-safe but transductive with respect to node
  presence and the compound-similarity graph. Do not relabel it as a strictly
  inductive cold-compound representation.
- Do not tune the model, threshold, calibrator, or ensemble after inspecting the
  locked test.
- Do not copy precomputed or live predictions into observation, training, or
  evaluation files.
- Preserve the complete run directory, both Git commit hashes, environment
  specification, Slurm log, readiness reports, and model artifacts.
- Keep weak numeric activity unlabeled by default
  (`PRING_WEAK_ACTIVITY_AS_NEGATIVE=false`); changing this switch creates a
  different label policy and therefore a different experiment.
- Use a new run/output identifier for every rerun. The script refuses to
  overwrite or resume scientific results.

## 7. Bounded smoke run

A CPU smoke run can check commands and file contracts, but it is not a thesis
experiment:

```bash
export MODEL_DEVICE=cpu
export MODEL_STAGE1_DEVICE=cpu
export MODEL_STAGE2_DEVICE=cpu
export MODEL_STAGE3_DEVICE=cpu
export MODEL_STAGE1_N_ESTIMATORS=20
export MODEL_STAGE2_EPOCHS=1
export MODEL_STAGE2_TARGET_TRAIN_REPEAT=1
export MODEL_RGCN_EPOCHS=1
export MODEL_HGT_EPOCHS=1
export MODEL_FINAL_SEEDS=1
export MODEL_FINAL_MIN_SEEDS=1
export MODEL_FINAL_BOOTSTRAP_RESAMPLES=20
export STAGE1_MAX_CANDIDATE_ROWS=1000
export REQUIRE_CLEAN_GIT=false
```

Use a separately named smoke-run directory and do not cite its performance.
