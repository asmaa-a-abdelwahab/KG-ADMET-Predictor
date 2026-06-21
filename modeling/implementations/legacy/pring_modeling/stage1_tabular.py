from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import (
    STAGE1,
    binary_metrics,
    coerce_binary_label,
    ensure_dir,
    optimize_binary_threshold,
    read_table,
    resolve_stage_dir,
    save_json,
)
from .logging_utils import get_logger
from .neo4j_export import export_predictions_dataframe

logger = get_logger(__name__)

IDENTIFIER_TOKENS = (
    "node_id", "node_ref", "cid", "protein_id", "accession", "compound_id", "target_id",
    "smiles", "inchi", "xref", "identifier",
)

# These terms are direct evidence/outcome terms in the current PRING exports.
# Using them gives a perfect-looking but invalid Stage 1 model, because the label
# is derived from these columns.  Stage 1 should be a structural/GDS baseline.
LEAKAGE_TOKENS = (
    "label", "active", "inactive", "weak", "positive", "negative", "ambiguous",
    "endpoint", "evidence", "assay", "reference", "bindingdb", "textmine",
    "ic50", "ki_", "kd_", "affinity", "molar", "um", "nm", "best_value", "min_",
    "label_rule", "negative_source", "stage_use", "split", "target_relation",
)

STRUCTURAL_HINTS = (
    "fastrp", "graphsage", "embedding", "degree", "pagerank", "article_rank", "triangle",
    "community", "louvain", "wcc", "centrality", "similarity", "jaccard", "adamic", "preferential",
    "common_neighbor", "common_neighbour", "shortest_path", "path", "topolog", "gds",
)

METADATA_COLUMNS = [
    "compound_node_id", "compound_node_ref", "protein_node_id", "protein_node_ref",
    "label", "split", "split_group", "stage_use", "target_relation", "label_rule",
    "negative_source", "candidate_sampling_method",
]


def _is_identifier_like(col: str) -> bool:
    lower = col.lower()
    return any(tok in lower for tok in IDENTIFIER_TOKENS)


def _is_leakage_like(col: str) -> bool:
    lower = col.lower()
    return any(tok in lower for tok in LEAKAGE_TOKENS)


def _is_structural_like(col: str) -> bool:
    lower = col.lower()
    return any(tok in lower for tok in STRUCTURAL_HINTS)


def find_training_file(stage_dir: Path) -> Path:
    for name in [
        "compound_target_training_pairs_for_gds.csv",
        "compound_target_training_pairs.csv",
        "positive_negative_compound_target_pairs.csv",
    ]:
        p = stage_dir / name
        if p.exists():
            return p
    candidates = sorted(stage_dir.glob("*training*pair*.csv")) + sorted(stage_dir.glob("*pair*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No Stage 1 training pair CSV found under {stage_dir}")
    return candidates[0]


def find_candidate_file(stage_dir: Path) -> Path | None:
    for name in ["candidate_pairs_for_gds_scoring.csv", "candidate_pairs.csv", "compound_target_candidate_pairs.csv"]:
        p = stage_dir / name
        if p.exists():
            return p
    hits = sorted(stage_dir.glob("*candidate*pair*.csv"))
    return hits[0] if hits else None


def _sample_by_label(df: pd.DataFrame, label_col: str, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.copy()
    frac = max_rows / float(len(df))
    parts = []
    for _, group in df.groupby(label_col, dropna=False):
        n = max(1, int(round(len(group) * frac)))
        parts.append(group.sample(n=min(n, len(group)), random_state=seed))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed)
    return out.reset_index(drop=True)


def _candidate_feature_columns(df: pd.DataFrame, feature_policy: str) -> tuple[list[str], list[str]]:
    selected: list[str] = []
    excluded: list[str] = []
    for col in df.columns:
        if col.startswith("Unnamed") or col == "_label" or _is_identifier_like(col):
            excluded.append(col); continue
        if col in {"label", "split", "split_group", "stage_use", "label_rule", "negative_source", "candidate_sampling_method"}:
            excluded.append(col); continue
        if feature_policy == "leakage_safe" and _is_leakage_like(col):
            excluded.append(col); continue
        if feature_policy == "structural_only" and not _is_structural_like(col):
            excluded.append(col); continue
        selected.append(col)
    return selected, excluded


def make_feature_matrix(df: pd.DataFrame, feature_columns: list[str] | None = None, feature_policy: str = "leakage_safe") -> tuple[pd.DataFrame, list[str], list[str]]:
    work = df.copy()
    excluded: list[str] = []
    if feature_columns is None:
        candidate_cols, excluded = _candidate_feature_columns(work, feature_policy)
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        for col in candidate_cols:
            coerced = pd.to_numeric(work[col], errors="coerce")
            if coerced.notna().sum() > 0 and coerced.notna().mean() >= 0.05 and coerced.nunique(dropna=True) > 1:
                work[col] = coerced.astype("float32")
                numeric_cols.append(col)
            else:
                nunique = work[col].dropna().astype(str).nunique()
                if 1 < nunique <= 25 and feature_policy == "allow_all":
                    categorical_cols.append(col)
                else:
                    excluded.append(col)
        X = work[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=work.index)
        if categorical_cols:
            X = pd.concat([X, pd.get_dummies(work[categorical_cols].astype(str), prefix=categorical_cols, dtype="float32")], axis=1)
        feature_columns = list(X.columns)
    else:
        X_tmp, _, _ = make_feature_matrix(work, feature_columns=None, feature_policy=feature_policy)
        for col in feature_columns:
            if col not in X_tmp.columns:
                X_tmp[col] = np.nan
        X = X_tmp[feature_columns]
    X = X.replace([np.inf, -np.inf], np.nan).astype("float32", copy=False)
    return X, feature_columns, excluded


def build_estimator(args: argparse.Namespace):
    if args.classifier == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=args.min_samples_leaf,
            max_depth=args.max_depth,
            n_jobs=args.n_jobs,
            class_weight="balanced",
            random_state=args.seed,
        )
    if args.classifier == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=args.n_estimators,
            max_leaf_nodes=31 if args.max_depth is None else max(2, 2 ** min(args.max_depth, 10)),
            learning_rate=0.04,
            l2_regularization=1e-3,
            random_state=args.seed,
        )
    if args.classifier == "logistic_regression":
        return LogisticRegression(max_iter=2000, n_jobs=args.n_jobs, class_weight="balanced", random_state=args.seed)
    return RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_depth=args.max_depth,
        n_jobs=args.n_jobs,
        class_weight="balanced_subsample",
        random_state=args.seed,
    )


def _make_pipeline(args: argparse.Namespace) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
        ("classifier", build_estimator(args)),
    ])


def write_feature_importance(model: Pipeline, feature_columns: list[str], out_dir: Path) -> str | None:
    clf = model.named_steps.get("classifier")
    values = None
    if hasattr(clf, "feature_importances_"):
        values = getattr(clf, "feature_importances_")
    elif hasattr(clf, "coef_"):
        coef = getattr(clf, "coef_")
        values = np.abs(coef[0] if getattr(coef, "ndim", 1) > 1 else coef)
    if values is None or len(values) != len(feature_columns):
        return None
    fi = pd.DataFrame({"feature": feature_columns, "importance": values}).sort_values("importance", ascending=False)
    path = out_dir / "feature_importance.csv"
    fi.to_csv(path, index=False)
    return str(path)


def _read_candidate_rows(candidate_file: Path, max_rows: int) -> pd.DataFrame:
    if max_rows > 0:
        return read_table(candidate_file, nrows=max_rows)
    return read_table(candidate_file)


def _split_indices(df: pd.DataFrame, y: pd.Series, args: argparse.Namespace):
    if args.group_split and args.group_column in df.columns and df[args.group_column].nunique() > 1:
        groups = df[args.group_column].astype(str)
        gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        return next(gss.split(df, y, groups=groups))
    stratify = y if y.value_counts().min() >= 2 else None
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=args.test_size, stratify=stratify, random_state=args.seed)


def _write_empty_summary(args, stage_dir, train_file, candidate_file, out_dir, report_dir, reason, excluded_columns) -> dict:
    summary = {
        "stage": "Stage 1 — Neo4j GDS/tabular baseline",
        "model": f"stage1_tabular_{args.classifier}",
        "status": "skipped",
        "skip_reason": reason,
        "stage_dir": str(stage_dir),
        "training_file": str(train_file),
        "candidate_file": str(candidate_file) if candidate_file else None,
        "feature_policy": args.feature_policy,
        "leakage_warning": "The current Stage 1 export appears to contain only evidence/outcome-derived columns. Re-export FastRP/GraphSAGE/topological features from Neo4j GDS for a valid structural baseline.",
        "excluded_feature_columns": excluded_columns[:200],
        "metrics": {},
    }
    save_json(summary, out_dir / "metrics.json")
    (out_dir / "predictions.csv").write_text("score,predicted_label,model,stage\n", encoding="utf-8")
    (report_dir / "stage1_modeling_summary.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def run(args: argparse.Namespace) -> dict:
    resolved = resolve_stage_dir(args.modeling_dir, STAGE1)
    stage_dir = resolved.root
    out_dir = ensure_dir(args.output_dir)
    report_dir = ensure_dir(args.report_dir or out_dir)
    try:
        train_file = find_training_file(stage_dir)
        candidate_file = find_candidate_file(stage_dir)
        logger.info("Stage 1 directory: {}", stage_dir)
        logger.info("Training file: {}", train_file)
        if candidate_file:
            logger.info("Candidate scoring file: {}", candidate_file)

        train_df = read_table(train_file)
        if args.target_column not in train_df.columns:
            raise KeyError(f"Target column {args.target_column!r} was not found in {train_file}")
        train_df["_label"] = coerce_binary_label(train_df[args.target_column])
        supervised = train_df[train_df["_label"].isin([0.0, 1.0])].copy()
        supervised = _sample_by_label(supervised, "_label", args.max_training_rows, args.seed)
        if len(supervised) < 4 or supervised["_label"].nunique() < 2:
            raise ValueError("Stage 1 needs at least four supervised rows and both positive/negative labels.")

        X, feature_columns, excluded_columns = make_feature_matrix(supervised, feature_policy=args.feature_policy)
        leakage_risk = args.feature_policy == "allow_all"
        if X.empty or not feature_columns:
            return _write_empty_summary(args, stage_dir, train_file, candidate_file, out_dir, report_dir, "no_leakage_safe_features_found", excluded_columns)

        y = supervised["_label"].astype(int).reset_index(drop=True)
        train_idx, test_idx = _split_indices(supervised.reset_index(drop=True), y, args)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = _make_pipeline(args)
        model.fit(X_train, y_train)
        test_score = model.predict_proba(X_test)[:, 1]
        selected_threshold, metrics = optimize_binary_threshold(y_test.to_numpy(), test_score, metric=args.threshold_selection)

        # Optional out-of-fold diagnostics on the supervised training set.
        cv_metrics = None
        if args.cv_folds and args.cv_folds >= 3 and y.value_counts().min() >= args.cv_folds:
            cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
            cv_model = _make_pipeline(args)
            cv_score = cross_val_predict(cv_model, X, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
            _, cv_metrics = optimize_binary_threshold(y.to_numpy(), cv_score, metric=args.threshold_selection)
            cv_pred = supervised[METADATA_COLUMNS].copy() if all(c in supervised.columns for c in METADATA_COLUMNS) else supervised[[c for c in METADATA_COLUMNS if c in supervised.columns]].copy()
            cv_pred["score"] = cv_score
            cv_pred["label"] = y.to_numpy()
            cv_pred["model"] = f"stage1_tabular_{args.classifier}_cv"
            cv_pred["stage"] = "Stage 1 — Neo4j GDS/tabular baseline"
            cv_pred.to_csv(out_dir / "eval_predictions.csv", index=False)

        if args.prediction_scope == "supervised" or candidate_file is None:
            score_df = supervised.iloc[test_idx].copy().reset_index(drop=True)
            X_score, _, _ = make_feature_matrix(score_df, feature_columns, feature_policy=args.feature_policy)
        else:
            score_df = _read_candidate_rows(candidate_file, args.max_scoring_rows)
            if args.target_column not in score_df.columns:
                score_df[args.target_column] = "unknown"
            X_score, _, _ = make_feature_matrix(score_df, feature_columns, feature_policy=args.feature_policy)
        score = model.predict_proba(X_score)[:, 1]
        meta_cols = [c for c in METADATA_COLUMNS if c in score_df.columns]
        preds = score_df[meta_cols].copy()
        preds["score"] = score
        preds["predicted_label"] = (score >= selected_threshold).astype(int)
        preds["model"] = f"stage1_tabular_{args.classifier}"
        preds["stage"] = "Stage 1 — Neo4j GDS/tabular baseline"
        preds = preds.sort_values("score", ascending=False).reset_index(drop=True)
        preds_to_write = preds.head(args.max_predictions_file_rows).copy() if args.max_predictions_file_rows > 0 else preds

        model_path = out_dir / f"stage1_tabular_{args.classifier}.joblib"
        pred_path = out_dir / "predictions.csv"
        joblib.dump(model, model_path)
        preds_to_write.to_csv(pred_path, index=False)
        save_json(feature_columns, out_dir / "feature_columns.json")
        save_json(excluded_columns, out_dir / "excluded_feature_columns.json")
        feature_importance_file = write_feature_importance(model, feature_columns, out_dir)
        summary = {
            "stage": "Stage 1 — Neo4j GDS/tabular baseline",
            "model": f"stage1_tabular_{args.classifier}",
            "status": "trained_with_leakage_warning" if leakage_risk else "trained",
            "stage_dir": str(stage_dir),
            "training_file": str(train_file),
            "candidate_file": str(candidate_file) if candidate_file else None,
            "training_rows_used": int(len(supervised)),
            "feature_count": int(len(feature_columns)),
            "feature_policy": args.feature_policy,
            "excluded_feature_count": int(len(excluded_columns)),
            "leakage_warning": "Metrics are not publishable if feature_policy=allow_all because evidence/outcome columns are used." if leakage_risk else None,
            "group_split": bool(args.group_split),
            "group_column": args.group_column if args.group_split else None,
            "selected_threshold": float(selected_threshold),
            "threshold_selection": args.threshold_selection,
            "prediction_scope": args.prediction_scope,
            "prediction_rows_written": int(len(preds_to_write)),
            "model_file": str(model_path),
            "predictions_file": str(pred_path),
            "eval_predictions_file": str(out_dir / "eval_predictions.csv") if (out_dir / "eval_predictions.csv").exists() else None,
            "feature_importance_file": feature_importance_file,
            "metrics": metrics,
            "cv_metrics": cv_metrics,
            **{k: v for k, v in metrics.items()},
        }
        if args.export_neo4j:
            try:
                summary["neo4j_export"] = export_predictions_dataframe(preds, model_name=f"stage1_tabular_{args.classifier}", max_rows=args.max_neo4j_predictions)
            except Exception as exc:
                summary["neo4j_export"] = {"exported": 0, "error": str(exc)}
        save_json(summary, out_dir / "metrics.json")
        (report_dir / "stage1_modeling_summary.md").write_text(render_markdown(summary), encoding="utf-8")
        return summary
    finally:
        resolved.cleanup()


def render_markdown(summary: dict) -> str:
    lines = ["# Stage 1 modeling summary", ""]
    for key, value in summary.items():
        if key in {"metrics", "cv_metrics", "neo4j_export"}:
            continue
        lines.append(f"- **{key}**: `{value}`")
    lines.append("\n## Metrics")
    for k, v in summary.get("metrics", {}).items():
        lines.append(f"- **{k}**: `{v}`")
    if summary.get("cv_metrics"):
        lines.append("\n## Cross-validation metrics")
        for k, v in summary.get("cv_metrics", {}).items():
            lines.append(f"- **{k}**: `{v}`")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 1 leakage-safe tabular/GDS baseline for PRING CYP450 pair exports.")
    p.add_argument("--modeling-dir", "--run-path", dest="modeling_dir", required=True, help="Full run, modeling folder, standalone Stage 1 folder, or zip.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--report-dir", default=None)
    p.add_argument("--target-column", default="label")
    p.add_argument("--prediction-scope", choices=["candidates", "supervised"], default="candidates")
    p.add_argument("--feature-policy", choices=["leakage_safe", "structural_only", "allow_all"], default="leakage_safe")
    p.add_argument("--max-training-rows", type=int, default=100000)
    p.add_argument("--max-scoring-rows", type=int, default=100000)
    p.add_argument("--max-predictions-file-rows", type=int, default=100000)
    p.add_argument("--threshold", type=float, default=0.5, help="Fallback threshold only; default evaluation selects threshold on holdout.")
    p.add_argument("--threshold-selection", choices=["mcc", "balanced_accuracy", "youden", "f1", "accuracy"], default="mcc")
    p.add_argument("--classifier", choices=["random_forest", "extra_trees", "hist_gradient_boosting", "logistic_regression"], default="random_forest")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--group-split", action="store_true", default=True)
    p.add_argument("--no-group-split", dest="group_split", action="store_false")
    p.add_argument("--group-column", default="compound_node_id")
    p.add_argument("--cv-folds", type=int, default=0)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--min-samples-leaf", type=int, default=2)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--export-neo4j", action="store_true")
    p.add_argument("--max-neo4j-predictions", type=int, default=25000)
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
