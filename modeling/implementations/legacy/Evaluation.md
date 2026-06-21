The results suggest the main problem is **not only model architecture**. It is mainly:

1. **Strong class imbalance**: Stage 1 and Stage 3 look around **90% positive / active** and only **10% negative / inactive**.
2. **Low specificity**: the models are predicting too many compounds as active.
3. **Weak negative examples**: inactive/weak cases are probably too few or not biologically hard enough.
4. **Stage 3 is probably under-featured**: R-GCN/HGT are using mostly graph structure, and many nodes may have weak or type-only features.
5. **Threshold/calibration is not stable**: several models have high recall but very poor ability to reject inactive pairs.

The most important observation is this:

| Model              | Main issue                                                                                                |
| ------------------ | --------------------------------------------------------------------------------------------------------- |
| Stage 1 ExtraTrees | Best MCC, but specificity is only **0.118**, so it misses most negatives.                                 |
| Stage 2 ComplEx    | Best specificity **0.778**, but ROC-AUC is near random, so ranking quality is weak.                       |
| Stage 3 R-GCN      | ROC-AUC **0.651** shows useful signal, but specificity **0.088** means threshold/imbalance is hurting it. |
| Stage 3 HGT        | More conservative than R-GCN, better specificity, but recall drops too much.                              |
| RotatE             | Currently failed; it predicts almost everything positive. Deprioritize unless retuned.                    |

## 1. Fix the evaluation first

Do not rely on **accuracy, F1, or average precision alone**. Because the dataset is mostly positive, these metrics look better than the real performance.

For your thesis/model comparison, rank mainly by:

```text
MCC
balanced accuracy
specificity
ROC-AUC
per-CYP isoform MCC
```

Also report the positive/negative ratio for each split:

```text
train positive %
validation positive %
test positive %
per CYP1A2 / CYP2C9 / CYP2C19 / CYP2D6 / CYP3A4 positive %
```

This is important because an aggregate model can look acceptable while failing on one CYP isoform.

## 2. Improve negative sample quality

This is probably the biggest performance improvement opportunity.

Right now, the model seems very good at finding positives but weak at identifying true negatives. You need stronger inactive examples.

Use these negative classes separately:

| Negative type                                                          | Why useful                                                |
| ---------------------------------------------------------------------- | --------------------------------------------------------- |
| Confirmed inactive from assay data                                     | Most reliable negatives                                   |
| Weak/inconclusive treated as separate class or low-confidence negative | Avoids noisy binary labels                                |
| Similar compounds to active compounds but inactive against same CYP    | Hard negatives                                            |
| Active against another CYP but inactive against target CYP             | Teaches isoform selectivity                               |
| Random unobserved compound-target pairs                                | Use only as weak/unknown negatives, not primary negatives |

For CYP450, the best hard negative is:

```text
Compound is structurally similar to a known active compound,
but experimentally inactive against the same CYP isoform.
```

This forces the model to learn more than simple chemical similarity.

## 3. Use balanced training and balanced validation

Your Stage 1/3 test sets look around 90% positive. Keep the real-world test set if needed, but also create a **balanced diagnostic test set**.

Recommended split strategy:

```text
Train: natural or moderately balanced
Validation: balanced or near-balanced
Test 1: natural distribution
Test 2: balanced diagnostic set
Test 3: per-CYP isoform test set
```

This will show whether the model is genuinely learning discrimination or just learning the dominant positive class.

## 4. Improve threshold selection

The models are too permissive. Stage 1 and R-GCN have very high recall but extremely poor specificity.

Instead of selecting the threshold only by MCC, test thresholds with a specificity constraint:

```text
Choose the threshold that maximizes MCC
while keeping specificity >= 0.50
```

For thesis purposes, you can report multiple operating points:

| Operating point            | Purpose                           |
| -------------------------- | --------------------------------- |
| High recall threshold      | Screening/discovery use case      |
| Balanced MCC threshold     | General classifier use            |
| High specificity threshold | Reducing false active predictions |

For CYP450 ADMET prediction, false positives and false negatives both matter, so a single default threshold is not enough.

## 5. Stage 1 improvements

Stage 1 is currently your strongest MCC model, so improve it first.

Recommended direct run:

```bash
python -m pring_modeling.stage1_tabular \
  --modeling-dir "$RUN_DIR" \
  --output-dir "$PROJECT_DIR/models_tuned/stage1_extra_trees_tuned" \
  --report-dir "$PROJECT_DIR/reports/tuned/stage1" \
  --feature-policy leakage_safe \
  --prediction-scope supervised \
  --classifier extra_trees \
  --n-estimators 1000 \
  --min-samples-leaf 5 \
  --threshold-selection mcc \
  --cv-folds 5 \
  --n-jobs 16
```

Try this grid:

| Parameter           | Values                                                   |
| ------------------- | -------------------------------------------------------- |
| classifier          | `extra_trees`, `random_forest`, `hist_gradient_boosting` |
| n_estimators        | 500, 1000, 1500                                          |
| min_samples_leaf    | 2, 5, 10, 20                                             |
| threshold_selection | `mcc`, `balanced_accuracy`, `youden`                     |
| feature_policy      | `leakage_safe`, then separately `structural_only`        |

Feature improvements for Stage 1:

```text
Compound descriptors:
- molecular weight
- logP
- TPSA
- H-bond donors/acceptors
- rotatable bonds
- Morgan/ECFP fingerprints

Graph features:
- FastRP embeddings
- GraphSAGE embeddings
- node degree
- compound similarity count
- number of supporting assays
- number of neighboring CYP-related entities

Pair features:
- compound embedding + protein embedding
- absolute difference
- elementwise product
- cosine similarity
- shortest-path features
```

Avoid direct leakage features such as endpoint result, active/inactive terms, IC50 labels, assay outcome text, or evidence labels used to construct the target.

## 6. Stage 2 KGE improvements

ComplEx is currently the only Stage 2 model worth improving. DistMult is weaker, and RotatE failed in this run.

Try ComplEx with larger embeddings, more negatives, more target-relation focus, and no graph triple cap if memory allows:

```bash
python -m pring_modeling.stage2_kge \
  --modeling-dir "$RUN_DIR" \
  --output-dir "$PROJECT_DIR/models_tuned/stage2_complex_d128_neg5" \
  --model complex \
  --epochs 150 \
  --dim 128 \
  --batch-size 16384 \
  --score-batch-size 262144 \
  --max-graph-train-triples 0 \
  --target-train-repeat 20 \
  --negatives-per-positive 5 \
  --eval-negatives-per-positive 10 \
  --loss softplus \
  --optimizer auto \
  --checkpoint-metric roc_auc \
  --train-supervised-decoder \
  --supervised-decoder extra_trees \
  --supervised-threshold-selection mcc \
  --device cuda
```

Also test:

```text
dim: 64, 128, 256
negatives_per_positive: 1, 5, 10
target_train_repeat: 5, 10, 20, 50
loss: softplus, bce
checkpoint_metric: roc_auc, average_precision, mcc
decoder: hist_gradient_boosting, extra_trees, logistic_regression
```

For RotatE, do not use it as-is. Try it only after this:

```bash
python -m pring_modeling.stage2_kge \
  --modeling-dir "$RUN_DIR" \
  --output-dir "$PROJECT_DIR/models_tuned/stage2_rotate_retuned" \
  --model rotate \
  --epochs 200 \
  --dim 128 \
  --margin 9.0 \
  --lr 0.0005 \
  --negatives-per-positive 10 \
  --eval-negatives-per-positive 10 \
  --target-train-repeat 20 \
  --loss margin \
  --checkpoint-metric roc_auc \
  --train-supervised-decoder \
  --supervised-decoder extra_trees \
  --supervised-threshold-selection mcc \
  --device cuda
```

But practically, I would prioritize **ComplEx** over RotatE.

## 7. Stage 3 R-GCN/HGT improvements

Stage 3 should theoretically outperform Stage 1, but your current R-GCN/HGT results suggest the model is not getting enough useful node features.

The current R-GCN result is interesting:

```text
ROC-AUC = 0.651
AP = 0.938
specificity = 0.088
```

This means the model has some ranking signal, but its decision threshold is too positive-biased.

Recommended R-GCN run:

```bash
MODEL_STAGE3_MODEL=rgcn \
MODEL_STAGE3_EPOCHS=100 \
MODEL_HIDDEN_DIM=128 \
MODEL_NUM_LAYERS=2 \
MODEL_NUM_NEIGHBORS="15,10" \
MODEL_BATCH_SIZE=128 \
MODEL_FEATURELESS_MODE=type \
MODEL_LOSS=weighted_bce_bpr \
MODEL_BPR_WEIGHT=0.3 \
MODEL_CLASS_WEIGHTING=negative_ratio \
MODEL_BALANCED_BATCHES=true \
MODEL_BALANCE_RATIO=1.0 \
MODEL_THRESHOLD_SELECTION=mcc \
MODEL_EARLY_STOPPING_METRIC=mcc \
MODEL_PATIENCE=15 \
MODEL_SCORE_CANDIDATES=false \
sbatch scripts/run_stage3_sampled_gpu_no_neo4j.sh
```

Then test HGT:

```bash
MODEL_STAGE3_MODEL=hgt \
MODEL_STAGE3_EPOCHS=100 \
MODEL_HIDDEN_DIM=64 \
MODEL_NUM_LAYERS=2 \
MODEL_NUM_NEIGHBORS="10,5" \
MODEL_BATCH_SIZE=64 \
MODEL_HGT_HEADS=2 \
MODEL_FEATURELESS_MODE=type \
MODEL_LOSS=weighted_bce_bpr \
MODEL_BPR_WEIGHT=0.3 \
MODEL_CLASS_WEIGHTING=negative_ratio \
MODEL_BALANCED_BATCHES=true \
MODEL_BALANCE_RATIO=1.0 \
MODEL_THRESHOLD_SELECTION=mcc \
MODEL_EARLY_STOPPING_METRIC=mcc \
MODEL_PATIENCE=15 \
MODEL_SCORE_CANDIDATES=false \
sbatch scripts/run_stage3_sampled_gpu_no_neo4j.sh
```

The biggest Stage 3 improvement would be adding real node features:

### Compound features

```text
Morgan fingerprints
MACCS keys
molecular descriptors
PubChem physicochemical properties
structural alerts
similarity-cluster ID
```

### Protein/CYP features

```text
CYP isoform one-hot encoding
protein family encoding
sequence embedding if available
target-specific prior activity statistics
```

### Assay/evidence features

```text
assay type
endpoint type
activity value bucket
evidence count
source reliability
assay organism/cell context if available
```

Right now, if many nodes are featureless and `featureless-mode=type`, then all nodes of the same type start almost identical. The GNN then depends heavily on neighborhood structure only, which may not be enough.

## 8. Train per-CYP isoform models

Since there are only five target proteins, aggregate performance can hide isoform-specific problems.

Run separate binary classifiers for:

```text
CYP1A2
CYP2C9
CYP2C19
CYP2D6
CYP3A4
```

Then compare:

```text
macro-MCC
macro-balanced accuracy
per-isoform specificity
per-isoform recall
```

This is especially important because CYP3A4 often has much more data than other isoforms, and the model may become biased toward patterns from the largest target.

## 9. Add a stacked ensemble

You should not choose only one stage. The stages capture different signals:

```text
Stage 1: structural/tabular graph features
Stage 2: global KG embedding similarity
Stage 3: local heterogeneous message passing
```

Create an ensemble dataset with:

```text
compound_id
protein_id
true_label
stage1_score
stage2_complex_score
stage2_distmult_score
stage3_rgcn_score
stage3_hgt_score
```

Then train a small meta-classifier:

```text
logistic regression
ExtraTrees
HistGradientBoosting
```

This may improve MCC faster than trying to make R-GCN perfect.

Expected best practical path:

```text
Stage 1 ExtraTrees + Stage 2 ComplEx decoder + Stage 3 R-GCN score
```

## 10. Priority order

I would improve the work in this order:

1. **Fix negative sampling and split design.**
2. **Report balanced/per-isoform metrics.**
3. **Tune Stage 1 ExtraTrees with better structural + molecular features.**
4. **Retune ComplEx with more negatives and larger embedding size.**
5. **Improve Stage 3 node features.**
6. **Use MCC/specificity-aware thresholding.**
7. **Build an ensemble from Stage 1 + Stage 2 + Stage 3.**

The best interpretation is:

> The models already show useful signal, especially Stage 1 ExtraTrees and Stage 3 R-GCN, but the current pipeline is biased toward predicting active interactions because of class imbalance and weak negative discrimination. The next improvement should focus on higher-quality inactive examples, balanced validation, specificity-aware thresholding, and richer compound/protein node features.