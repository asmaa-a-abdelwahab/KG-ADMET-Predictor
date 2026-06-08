# Integrated PRING CYP450 modeling image

This modeling package is designed for the Dockerized PRING/Neo4j/Streamlit application. It consumes PRING modeling exports and writes predictions back to Neo4j as `PREDICTED_INTERACTION` relationships.

## What is included

- **Stage 1**: memory-safe tabular/GDS baseline from `compound_target_training_pairs_for_gds.csv` and `candidate_pairs_for_gds_scoring.csv`.
- **Optional Stage 1 MLP**: PyTorch MLP over exported Neo4j GDS embeddings.
- **Stage 2**: PyTorch DistMult, ComplEx, and RotatE knowledge-graph embedding baselines from TSV triples.
- **Stage 3**: R-GCN and HGT + MLP decoder for PyTorch Geometric `HeteroData` exports.
- **Stage 4**: Neo4j evidence-path explanation report for predicted interactions.
- **Comparison**: metrics ranking, prediction score agreement, static plots, and HTML report.
- **Notebooks**: one notebook per modeling stage plus a comparison notebook in `modeling/notebooks/`.

## Accepted input layouts

The scripts accept any of these:

```text
/runs/current                                 # full PRING run
/runs/current/graph/ml/modeling              # modeling export folder
/runs/current/graph/ml/modeling/stage2_...   # standalone stage folder
/path/to/stage1_neo4j_gds_baselines.zip      # zip containing a stage folder
```

The attached standalone Stage 1 and Stage 2 folders are supported directly.

## Docker usage

Build and start Neo4j plus Streamlit:

```bash
docker compose up --build neo4j streamlit
```

Load a PRING run into Neo4j:

```bash
docker compose --profile load up --build pring-loader
```

Run the default modeling workflow. By default this runs Stage 1 and Stage 2, compares metrics, exports predictions to Neo4j, and skips unavailable stages without stopping the workflow:

```bash
docker compose --profile train up --build modeling
```

Useful environment examples:

```bash
# Run only Stage 1 tabular baseline
MODEL_STAGE=stage1 docker compose --profile train up --build modeling

# Run Stage 2 RotatE with lower memory settings
MODEL_STAGE=stage2 MODEL_KGE_MODEL=rotate MODEL_KGE_DIM=64 MODEL_MAX_GRAPH_TRAIN_TRIPLES=500000 docker compose --profile train up --build modeling

# Run Stage 3 R-GCN when stage3_heterogeneous_gnn/pyg_export/heterodata.pt exists
MODEL_STAGE=stage3 MODEL_STAGE3_MODEL=rgcn MODEL_STAGE3_EPOCHS=50 docker compose --profile train up --build modeling

# Run all requested stages, including Stage 3
MODEL_STAGE=run_all MODEL_STAGES="stage1 stage2 stage3" docker compose --profile train up --build modeling
```

Outputs are written to:

```text
artifacts/models/
artifacts/reports/modeling/
```

## Local CLI usage

```bash
cd modeling
pip install -r requirements.txt
export PYTHONPATH=$PWD

python -m pring_modeling.stage1_tabular \
  --modeling-dir /path/to/stage1_neo4j_gds_baselines \
  --output-dir outputs/stage1_tabular \
  --max-training-rows 100000 \
  --max-scoring-rows 100000

python -m pring_modeling.stage2_kge \
  --modeling-dir /path/to/stage2_kg_embedding_baselines \
  --model rotate \
  --epochs 20 \
  --dim 64 \
  --max-graph-train-triples 500000 \
  --max-candidate-triples 100000 \
  --output-dir outputs/stage2_rotate

python -m pring_modeling.stage3_rgcn \
  --modeling-dir /path/to/graph/ml/modeling \
  --epochs 50 \
  --hidden-dim 128 \
  --output-dir outputs/stage3_rgcn

python -m pring_modeling.compare metrics \
  --outputs-root outputs \
  --output-dir outputs/comparison

python -m pring_modeling.compare visualize \
  --comparison-csv outputs/comparison/model_comparison.csv \
  --output-dir outputs/comparison/figures
```

## Neo4j prediction export

All trainable stages support:

```bash
--export-neo4j --max-neo4j-predictions 25000
```

The exporter parses PRING node references such as:

```text
Compound|cid=10002960
Protein|protein_id=P08684
```

and writes:

```cypher
(:Compound)-[:PREDICTED_INTERACTION {score, raw_score, predicted_label, model, stage, updated_at}]->(:Protein)
```

## Notes for the attached datasets

- The Stage 1 candidate file is very large, so the default scorer reads only the first `MODEL_MAX_SCORING_ROWS` rows for Docker safety.
- The Stage 2 graph triple file can create a very large embedding table. The default `MODEL_MAX_GRAPH_TRAIN_TRIPLES=500000` caps graph-context triples and appends `target_relation_train.tsv` so the target relation is included.
- KGE `score` values are `sigmoid(raw_score)` for ranking convenience. They are not calibrated probabilities.
- Stage 3 requires a PyTorch Geometric `HeteroData` export. If it is not present, `run_all` logs the error and continues when `MODEL_CONTINUE_ON_ERROR=true`.

## Local notebook import fix

When opening notebooks directly from `modeling/notebooks` in VS Code or Jupyter, the Python kernel may not automatically include the parent `modeling` folder on `sys.path`. The notebooks now include a bootstrap cell that finds `pring_modeling` automatically. As an alternative, install the package in editable mode from the project root:

```bash
python -m pip install -e ./modeling
```


## Local notebook environment fix

If a notebook raises `ModuleNotFoundError` for `pring_modeling`, `loguru`, `sklearn`, `neo4j`, or another package, install the local modeling package and its notebook dependencies from the repository root:

```bash
python -m pip install -e "./modeling[notebooks]"
```

On Windows PowerShell:

```powershell
python -m pip install -e ".\modeling[notebooks]"
```

For Stage 2/Stage 3 PyTorch notebooks, install the PyTorch/PyG extras only after selecting the correct CPU/CUDA wheel for your machine:

```bash
python -m pip install -e "./modeling[torch]"
```

The code also contains a fallback logger, so a missing `loguru` package should no longer block Stage 1 imports.

### Stage 1 notebook default paths

The Stage 1 notebook now uses this default standalone Stage 1 export path:

```python
MODEL_INPUT = Path("A:/Repositories/PRING/runs/cyp450_5enzymes_uncapped_raw_rematerialized/graph/ml/modeling/stage1_neo4j_gds_baselines")
OUTPUT_DIR = Path("/models/notebook_stage1_tabular")
REPORT_DIR = Path("/reports/modeling")
```

When running inside Docker or HPC, change `MODEL_INPUT` to the mounted Linux path for the same folder.


## End-to-end automation update

The Docker entrypoint now supports two modes:

```bash
# Start the modeling container but do not train
MODEL_AUTO_TRAIN=false docker compose --profile modeling up --build modeling

# Train immediately when the container starts
MODEL_AUTO_TRAIN=true docker compose --profile modeling up --build modeling
```

`MODEL_STAGE=run_all` now trains all configured model families, selects the best model per stage, runs comparison visualizations, exports best predictions to Neo4j, and creates Stage 4 explanation reports.

Default multi-model configuration:

```bash
MODEL_STAGES="stage1 stage2 stage3 stage4"
MODEL_STAGE1_MODELS="random_forest extra_trees"
MODEL_STAGE2_MODELS="distmult complex rotate"
MODEL_STAGE3_MODELS="rgcn hgt"
MODEL_EXPORT_SCOPE="best_only"
```

For HPC/local non-Docker use:

```bash
bash scripts/run_all_modeling_local.sh
sbatch scripts/run_all_modeling_slurm.sh
```
