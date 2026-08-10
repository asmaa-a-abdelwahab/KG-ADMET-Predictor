# Complete five-CYP local workflow

`05_full_cyp450_pipeline.sh` is the local launcher for the same canonical
end-to-end implementation used by the Slurm script. It collects or reuses the
five-target source run, rematerializes modeling-ready data, creates EDA and
leakage audits, builds outcome-safe Neo4j GDS/FastRP features, trains every
implemented canonical model, and runs the strict final readiness gate.

The fixed targets are CYP3A4 (`P08684`), CYP1A2 (`P05177`), CYP2C19
(`P33261`), CYP2C9 (`P11712`), and CYP2D6 (`P10635`).

## Prerequisites

- Linux, macOS, or WSL2 with Bash (WSL2 is recommended on Windows);
- Python 3.10 or newer;
- Docker for the default isolated Neo4j 5.26+GDS runtime, or a dedicated empty
  external Neo4j+GDS database;
- internet access when collecting a new PRING run and when first bootstrapping
  the Python/Docker runtimes; and
- sufficient RAM, disk, and time for uncapped extraction and all models.

A full local CPU run can take a long time. For thesis results, use a compatible
CUDA environment or the Slurm launcher. The script does not silently cap data
or model rows to make a run fit a small workstation.

## Configure and run

Copy the environment template outside the repository and protect it:

```bash
cp examples/local/full_pipeline.env.example /secure/path/pring-local.env
chmod 600 /secure/path/pring-local.env
```

Edit its paths. Keep `PRING_WEAK_ACTIVITY_AS_NEGATIVE=false` unless a new label
policy is intentionally being defined. The pipeline defaults to complete
ComplEx, DistMult, RotatE, R-GCN, HGT, Stage 1 ExtraTrees, and calibrated
five-seed ensemble evaluation.

Export the Neo4j secret and run:

```bash
read -r -s -p "Neo4j password: " NEO4J_PASSWORD
export NEO4J_PASSWORD

bash examples/local/05_full_cyp450_pipeline.sh \
  /secure/path/pring-local.env
```

The default `LOCAL_NEO4J_MODE=managed` creates a uniquely named container and
volume, verifies GDS, and removes only those pipeline-created resources on
exit. Set `LOCAL_KEEP_NEO4J=true` to retain them. Set
`LOCAL_NEO4J_MODE=external` and provide `NEO4J_URI` to use a dedicated empty
database that you manage yourself.

For a first portable installation, use:

```bash
BOOTSTRAP_ENV=true
BOOTSTRAP_TORCH_PROFILE=cpu
MODEL_DEVICE=auto
```

For CUDA 12.4, set `BOOTSTRAP_TORCH_PROFILE=cu124`. For another CUDA stack,
prepare the virtual environment yourself and use:

```bash
BOOTSTRAP_ENV=false
BOOTSTRAP_TORCH_PROFILE=none
PRING_VENV=/absolute/path/to/prepared/venv
MODEL_DEVICE=cuda
```

## Reuse already collected data

To avoid repeating PubChem and enrichment collection, point at a completed
source run:

```bash
export PRING_SOURCE_RUN_DIR=/data/pring/runs/completed_source_run
```

The source is never modified. A distinct `_ready` run is generated with fresh
schema, split, and modeling exports. Reuse is accepted only when the source
quality report identifies uncapped data, all unobserved candidates, no
pipeline-readiness blockers, and a modeling-ready export.

## Result contract

The ready PRING run is under
`PRING_RUN_ROOT/PRING_READY_RUN_ID/graph/ml/modeling`. The independent
`PIPELINE_OUTPUT_ROOT` contains:

- EDA HTML, Markdown, JSON, figures, and tables;
- Stage 1 ExtraTrees metrics and predictions;
- separate ComplEx, DistMult, and RotatE metrics and predictions;
- R-GCN and HGT metrics and predictions;
- calibrated per-seed and aggregate final metrics, common-test comparisons,
  per-target metrics, uncertainty tables, and top-k rankings;
- the all-model comparison report; and
- prepared and final machine-readable readiness reports.

Success requires all five CYPs, frozen train/validation/test partitions,
train-only graph exports, no split-group overlap, all three KGEs, both GNNs,
five completed final-validation seeds, provenance agreement, and
`publishable=true`. That flag confirms the software/artifact protocol; it does
not establish external biological or clinical validity.

Every final run must use unique run/output identifiers. The pipeline refuses
existing output directories and does not overwrite or mix partial scientific
results. Use the Slurm guide in `../hpc/README_FULL_PIPELINE.md` for HPC setup.
