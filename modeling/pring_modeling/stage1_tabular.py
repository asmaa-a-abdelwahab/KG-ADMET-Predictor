from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from .logging_utils import get_logger

logger = get_logger(__name__)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import STAGE1, binary_metrics, coerce_binary_label, ensure_dir, read_table, resolve_stage_dir, save_json
from .neo4j_export import export_predictions_dataframe

IDENTIFIER_TOKENS = (
    "node_id", "node_ref", "cid", "protein_id", "accession", "compound_id", "target_id",
    "smiles", "inchi", "xref", "identifier",
)
METADATA_COLUMNS = [
    "compound_node_id", "compound_node_ref", "protein_node_id", "protein_node_ref",
    "label", "split", "split_group", "stage_use", "target_relation", "label_rule",
    "negative_source", "candidate_sampling_method",
]


def _is_identifier_like(col: str) -> bool:
    lower = col.lower()
    return any(tok in lower for tok in IDENTIFIER_TOKENS)


def find_training_file(stage_dir: Path) -> Path:
    for name in ["compound_target_training_pairs_for_gds.csv", "compound_target_training_pairs.csv", "positive_negative_compound_target_pairs.csv"]:
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


def make_feature_matrix(df: pd.DataFrame, feature_columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    work = df.copy()
    if feature_columns is None:
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        for col in work.columns:
            if col.startswith("Unnamed") or col == "_label" or _is_identifier_like(col):
                continue
            if col in {"label", "split", "split_group", "stage_use", "label_rule", "negative_source", "candidate_sampling_method"}:
                continue
            coerced = pd.to_numeric(work[col], errors="coerce")
            if coerced.notna().sum() > 0 and coerced.notna().mean() >= 0.05 and coerced.nunique(dropna=True) > 1:
                work[col] = coerced.astype("float32")
                numeric_cols.append(col)
            else:
                nunique = work[col].dropna().astype(str).nunique()
                if 1 < nunique <= 25 and any(t in col.lower() for t in ["type", "rule", "source", "method", "confidence", "relation"]):
                    categorical_cols.append(col)
        X = work[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=work.index)
        if categorical_cols:
            X = pd.concat([X, pd.get_dummies(work[categorical_cols].astype(str), prefix=categorical_cols, dtype="float32")], axis=1)
        feature_columns = list(X.columns)
    else:
        # Recreate numeric/categorical columns, then align to training columns.
        X_tmp, _ = make_feature_matrix(work, feature_columns=None)
        for col in feature_columns:
            if col not in X_tmp.columns:
                X_tmp[col] = np.nan
        X = X_tmp[feature_columns]
    X = X.replace([np.inf, -np.inf], np.nan).astype("float32", copy=False)
    return X, feature_columns


def build_estimator(args: argparse.Namespace):
    """Build the Stage 1 supervised baseline classifier.

    The Stage 1 export can contain Neo4j GDS FastRP/GraphSAGE-derived
    features plus pair/topological properties. Training multiple lightweight
    classifiers over the same leakage-safe export lets the workflow select the
    strongest Stage 1 baseline automatically.
    """
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
            learning_rate=0.05,
            random_state=args.seed,
        )
    if args.classifier == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            n_jobs=args.n_jobs,
            class_weight="balanced",
            random_state=args.seed,
        )
    return RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_depth=args.max_depth,
        n_jobs=args.n_jobs,
        class_weight="balanced_subsample",
        random_state=args.seed,
    )


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
    fi = pd.DataFrame({"feature": feature_columns, "importance": values})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    path = out_dir / "feature_importance.csv"
    fi.to_csv(path, index=False)
    return str(path)


def _read_candidate_rows(candidate_file: Path, max_rows: int) -> pd.DataFrame:
    if max_rows > 0:
        return read_table(candidate_file, nrows=max_rows)
    return read_table(candidate_file)


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

        X, feature_columns = make_feature_matrix(supervised)
        if X.empty:
            raise ValueError("No usable Stage 1 numeric/categorical features were found.")
        y = supervised["_label"].astype(int)
        stratify = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X, y, np.arange(len(y)), test_size=args.test_size, stratify=stratify, random_state=args.seed
        )
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("classifier", build_estimator(args)),
        ])
        model.fit(X_train, y_train)
        test_score = model.predict_proba(X_test)[:, 1]
        metrics = binary_metrics(y_test.to_numpy(), test_score, threshold=args.threshold)

        if args.prediction_scope == "supervised" or candidate_file is None:
            score_df = supervised.iloc[idx_test].copy().reset_index(drop=True)
            X_score, _ = make_feature_matrix(score_df, feature_columns)
        else:
            score_df = _read_candidate_rows(candidate_file, args.max_scoring_rows)
            if args.target_column not in score_df.columns:
                score_df[args.target_column] = "unknown"
            X_score, _ = make_feature_matrix(score_df, feature_columns)
        score = model.predict_proba(X_score)[:, 1]
        meta_cols = [c for c in METADATA_COLUMNS if c in score_df.columns]
        preds = score_df[meta_cols].copy()
        preds["score"] = score
        preds["predicted_label"] = (score >= args.threshold).astype(int)
        preds["model"] = f"stage1_tabular_{args.classifier}"
        preds["stage"] = "Stage 1 — Neo4j GDS/tabular baseline"
        preds = preds.sort_values("score", ascending=False).reset_index(drop=True)
        if args.max_predictions_file_rows > 0:
            preds_to_write = preds.head(args.max_predictions_file_rows).copy()
        else:
            preds_to_write = preds

        model_path = out_dir / f"stage1_tabular_{args.classifier}.joblib"
        pred_path = out_dir / "predictions.csv"
        joblib.dump(model, model_path)
        preds_to_write.to_csv(pred_path, index=False)
        save_json(feature_columns, out_dir / "feature_columns.json")
        feature_importance_file = write_feature_importance(model, feature_columns, out_dir)
        summary = {
            "stage": "Stage 1 — Neo4j GDS/tabular baseline",
            "model": f"stage1_tabular_{args.classifier}",
            "status": "trained",
            "stage_dir": str(stage_dir),
            "training_file": str(train_file),
            "candidate_file": str(candidate_file) if candidate_file else None,
            "training_rows_used": int(len(supervised)),
            "feature_count": int(len(feature_columns)),
            "prediction_scope": args.prediction_scope,
            "prediction_rows_written": int(len(preds_to_write)),
            "model_file": str(model_path),
            "predictions_file": str(pred_path),
            "feature_importance_file": feature_importance_file,
            "metrics": metrics,
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
        if key in {"metrics", "neo4j_export"}:
            continue
        lines.append(f"- **{key}**: `{value}`")
    lines.append("\n## Metrics")
    for k, v in summary.get("metrics", {}).items():
        lines.append(f"- **{k}**: `{v}`")
    if "neo4j_export" in summary:
        lines += ["", "## Neo4j export", "```json", json.dumps(summary["neo4j_export"], indent=2), "```"]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 1 memory-safe tabular/GDS baseline for PRING CYP450 pair exports.")
    p.add_argument("--modeling-dir", "--run-path", dest="modeling_dir", required=True, help="Full run, modeling folder, standalone Stage 1 folder, or zip.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--report-dir", default=None)
    p.add_argument("--target-column", default="label")
    p.add_argument("--prediction-scope", choices=["candidates", "supervised"], default="candidates")
    p.add_argument("--max-training-rows", type=int, default=100000)
    p.add_argument("--max-scoring-rows", type=int, default=100000)
    p.add_argument("--max-predictions-file-rows", type=int, default=100000)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--classifier", choices=["random_forest", "extra_trees", "hist_gradient_boosting", "logistic_regression"], default="random_forest")
    p.add_argument("--test-size", type=float, default=0.25)
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
