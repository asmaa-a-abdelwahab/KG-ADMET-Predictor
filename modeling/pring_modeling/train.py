"""Memory-safe modeling pipeline for PRING-generated CYP450 compound-target data.

The pipeline consumes PRING ``graph/ml`` exports.  It is intentionally designed
for large CYP450 runs where the pair table may contain millions of unknown
candidate pairs.  By default it trains on supervised positive/negative rows and
scores a bounded candidate sample instead of materializing every pair-feature
combination in memory.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

IDENTIFIER_COLUMNS = {
    "compound_node_id",
    "protein_node_id",
    "compound_node_ref",
    "protein_node_ref",
    "compound_id",
    "protein_id",
    "cid",
    "accession",
    "node_id",
    "node_ref",
    "split",
    "split_group",
    "split_strategy",
    "label_rule",
    "negative_source",
    "candidate_sampling_method",
}

PAIR_FILE_PREFERENCE = [
    "compound_target_link_prediction_pairs.csv",
    "compound_target_training_pairs.csv",
    "positive_compound_target_pairs.csv",
]

FEATURE_FILE_SPECS = [
    ("compound", "compound_node_ref", [
        "node_features_compound_model_matrix.csv",
        "node_features_compound_tensor.csv",
        "node_features_compound_normalized.csv",
        "node_features_compound.csv",
    ]),
    ("protein", "protein_node_ref", [
        "node_features_protein_model_matrix.csv",
        "node_features_protein_tensor.csv",
        "node_features_protein_normalized.csv",
        "node_features_protein.csv",
    ]),
    ("protembed", "protein_node_ref", [
        "node_features_protembed_model_matrix.csv",
        "node_features_protembed_tensor.csv",
        "node_features_protembed_normalized.csv",
        "node_features_protembed.csv",
    ]),
]

LEAKAGE_TOKENS = (
    "cid", "node_id", "node_ref", "identifier", "accession", "inchi", "smiles",
    "compound_id", "protein_id", "pubchem", "xref", "iri", "url", "uri",
)


@dataclass
class RunPaths:
    root: Path
    ml_dir: Path
    temporary_dir: Optional[tempfile.TemporaryDirectory[str]] = None

    def cleanup(self) -> None:
        if self.temporary_dir is not None:
            self.temporary_dir.cleanup()


def _find_run_root(path: Path) -> Path:
    """Resolve a PRING run root from a directory that may be nested."""
    path = path.resolve()
    if (path / "graph" / "ml").exists():
        return path
    if path.name == "ml" and path.parent.name == "graph":
        return path.parent.parent
    if path.name == "graph" and (path / "ml").exists():
        return path.parent
    for candidate in [path, *path.parents]:
        if (candidate / "graph" / "ml").exists():
            return candidate
    for graph_dir in path.glob("**/graph"):
        if (graph_dir / "ml").exists():
            return graph_dir.parent
    raise FileNotFoundError(
        f"Could not find a PRING run root below {path}. Expected graph/ml/*.csv."
    )


def resolve_run_paths(run_path: str | Path) -> RunPaths:
    """Resolve a PRING run directory or ZIP into usable paths."""
    path = Path(run_path)
    if path.is_file() and path.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory(prefix="pring_modeling_")
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp.name)
        root = _find_run_root(Path(tmp.name))
        return RunPaths(root=root, ml_dir=root / "graph" / "ml", temporary_dir=tmp)
    root = _find_run_root(path)
    return RunPaths(root=root, ml_dir=root / "graph" / "ml")


def _read_csv(path: Path, *, usecols: Optional[list[str]] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read CSV with settings that avoid pandas mixed-type chunk warnings."""
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, usecols=usecols, nrows=nrows)


def choose_pair_file(ml_dir: Path) -> Path:
    for name in PAIR_FILE_PREFERENCE:
        candidate = ml_dir / name
        if candidate.exists():
            return candidate
    candidates = sorted(ml_dir.glob("*pair*.csv")) + sorted(ml_dir.glob("*interaction*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No compound-target pair CSV found in {ml_dir}")
    return candidates[0]


def _normalize_node_ref(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _coerce_label(series: pd.Series) -> pd.Series:
    def convert(value: object) -> float:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        text = str(value).strip().lower()
        if text in {"1", "1.0", "positive", "active", "true", "yes"}:
            return 1.0
        if text in {"0", "0.0", "negative", "inactive", "weak", "false", "no"}:
            return 0.0
        if text in {"-1", "-1.0", "unknown", "candidate", "nan", "none", ""}:
            return -1.0
        try:
            return float(text)
        except ValueError:
            return np.nan
    return series.map(convert)


def _sample_by_label(df: pd.DataFrame, label_col: str, max_rows: int, random_state: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.copy()
    frac = max_rows / float(len(df))
    parts = []
    for _, group in df.groupby(label_col, dropna=False):
        n = max(1, int(round(len(group) * frac)))
        parts.append(group.sample(n=min(n, len(group)), random_state=random_state))
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_state)
    return sampled.reset_index(drop=True)


def _choose_scoring_rows(
    pairs: pd.DataFrame,
    supervised_mask: pd.Series,
    candidate_mask: pd.Series,
    scope: str,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    """Select rows to score without materializing all unknown candidate pairs by default."""
    if scope == "supervised":
        selected = pairs.loc[supervised_mask].copy()
    elif scope == "all":
        selected = pairs.copy()
        if max_rows > 0 and len(selected) > max_rows:
            logger.warning(
                "prediction-scope=all requested but rows_total={} exceeds max_scoring_rows={}; sampling.",
                len(selected),
                max_rows,
            )
            selected = selected.sample(n=max_rows, random_state=random_state)
    elif scope == "candidates":
        selected = pairs.loc[candidate_mask].copy()
        if max_rows > 0 and len(selected) > max_rows:
            selected = selected.sample(n=max_rows, random_state=random_state)
    else:  # supervised_plus_candidates
        supervised = pairs.loc[supervised_mask].copy()
        remaining = max(0, max_rows - len(supervised)) if max_rows > 0 else 0
        candidates = pairs.loc[candidate_mask].copy()
        if remaining > 0 and len(candidates) > remaining:
            candidates = candidates.sample(n=remaining, random_state=random_state)
        elif remaining == 0 and max_rows > 0:
            candidates = candidates.iloc[0:0]
        selected = pd.concat([supervised, candidates], ignore_index=True)
    return selected.reset_index(drop=True)


def _is_leaky_feature_name(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in LEAKAGE_TOKENS)


def _read_feature_table(
    path: Path,
    *,
    max_feature_columns: int,
    allowed_node_refs: Optional[set[str]],
) -> pd.DataFrame:
    """Read a node feature table while dropping identifier-like and non-numeric columns.

    The function limits columns before merging so large PRING feature tensors do not
    explode memory inside Docker Desktop.
    """
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    if "node_ref" not in header.columns:
        raise ValueError(f"Feature table {path} does not contain node_ref")

    candidate_cols = [c for c in header.columns if c not in {"node_id", "node_ref"} and not _is_leaky_feature_name(c)]
    if max_feature_columns > 0:
        candidate_cols = candidate_cols[:max_feature_columns]
    usecols = ["node_ref", *candidate_cols]
    features = _read_csv(path, usecols=usecols)
    features["node_ref"] = features["node_ref"].map(_normalize_node_ref)
    if allowed_node_refs is not None:
        features = features[features["node_ref"].isin(allowed_node_refs)]
    for col in candidate_cols:
        features[col] = pd.to_numeric(features[col], errors="coerce")
    keep_numeric = [c for c in candidate_cols if features[c].notna().any() and features[c].nunique(dropna=True) > 1]
    return features[["node_ref", *keep_numeric]]


def _merge_node_features(
    rows: pd.DataFrame,
    ml_dir: Path,
    *,
    use_node_features: bool,
    max_feature_columns: int,
) -> pd.DataFrame:
    """Merge selected PRING node feature matrices when explicitly enabled."""
    if not use_node_features:
        return rows.copy()

    out = rows.copy()
    for prefix, pair_key, names in FEATURE_FILE_SPECS:
        if pair_key not in out.columns:
            continue
        feature_path = next((ml_dir / n for n in names if (ml_dir / n).exists()), None)
        if not feature_path:
            continue
        try:
            out[pair_key] = out[pair_key].map(_normalize_node_ref)
            refs = set(out[pair_key].dropna().astype(str).unique())
            features = _read_feature_table(
                feature_path,
                max_feature_columns=max_feature_columns,
                allowed_node_refs=refs,
            )
        except Exception as exc:
            logger.warning("Could not read feature table {}: {}", feature_path, exc)
            continue
        feature_cols = [c for c in features.columns if c != "node_ref"]
        if not feature_cols:
            continue
        features = features.rename(columns={c: f"{prefix}__{c}" for c in feature_cols})
        out = out.merge(features, how="left", left_on=pair_key, right_on="node_ref")
        out = out.drop(columns=["node_ref"], errors="ignore")
        logger.info("Merged {} {} features from {}", len(feature_cols), prefix, feature_path.name)
    return out


def _safe_numeric_features(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, list[str]]:
    """Build a safe numeric matrix from PRING pair and optional node features."""
    excluded = set(IDENTIFIER_COLUMNS) | {target_column, "_label"}
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in df.columns:
        if col in excluded or col.startswith("Unnamed") or _is_leaky_feature_name(col):
            continue
        series = df[col]
        coerced = pd.to_numeric(series, errors="coerce")
        if coerced.notna().sum() > 0 and coerced.notna().mean() >= 0.05:
            df[col] = coerced.astype("float32")
            if coerced.nunique(dropna=True) > 1:
                numeric_cols.append(col)
        else:
            nunique = series.dropna().astype(str).nunique()
            if 1 < nunique <= 30 and any(token in col.lower() for token in ["type", "rule", "source", "method", "outcome"]):
                categorical_cols.append(col)

    matrix = df[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=df.index)
    if categorical_cols:
        cats = pd.get_dummies(df[categorical_cols].astype(str), prefix=categorical_cols, dummy_na=False, dtype="float32")
        matrix = pd.concat([matrix, cats], axis=1)
    matrix = matrix.replace([np.inf, -np.inf], np.nan).astype("float32", copy=False)
    return matrix, list(matrix.columns)


def _align_features(X: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    for col in feature_columns:
        if col not in X.columns:
            X[col] = np.nan
    return X[feature_columns]


def _split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    splits: Optional[pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if splits is not None:
        split_text = splits.astype(str).str.lower()
        train_mask = split_text.isin(["train", "training"])
        test_mask = split_text.isin(["test", "val", "valid", "validation"])
        if train_mask.sum() >= 2 and test_mask.sum() >= 1 and y[train_mask].nunique() > 1:
            return X[train_mask], X[test_mask], y[train_mask], y[test_mask]

    stratify = y if y.value_counts().min() >= 2 and y.nunique() > 1 else None
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)


def _evaluate(y_true: pd.Series, scores: np.ndarray, labels: np.ndarray) -> dict[str, float | int | None]:
    out: dict[str, float | int | None] = {
        "n_eval": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, labels)) if len(y_true) else None,
        "f1": float(f1_score(y_true, labels, zero_division=0)) if len(y_true) else None,
    }
    if len(set(y_true.astype(int))) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["average_precision"] = float(average_precision_score(y_true, scores))
    else:
        out["roc_auc"] = None
        out["average_precision"] = None
    return out


def _extract_ref_part(ref: str, label: str) -> str:
    ref = str(ref or "")
    prefix = f"{label}:"
    if ref.startswith(prefix):
        return ref[len(prefix):]
    if ":" in ref:
        return ref.split(":", 1)[1]
    return ref


def export_predictions_to_neo4j(predictions: pd.DataFrame, max_rows: int = 50000) -> dict[str, int | str]:
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "cyp450kg")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    required = {"compound_node_ref", "protein_node_ref", "score"}
    if not required.issubset(predictions.columns):
        return {"exported": 0, "reason": "missing required ref/score columns"}

    rows = []
    for _, row in predictions.head(max_rows).iterrows():
        compound_ref = str(row.get("compound_node_ref") or "")
        protein_ref = str(row.get("protein_node_ref") or "")
        score = row.get("score")
        if not compound_ref or not protein_ref or pd.isna(score):
            continue
        rows.append({
            "compound_ref": compound_ref,
            "protein_ref": protein_ref,
            "cid": _extract_ref_part(compound_ref, "Compound"),
            "protein_id": _extract_ref_part(protein_ref, "Protein"),
            "score": float(score),
            "predicted_label": int(row.get("predicted_label", score >= 0.5)),
            "model": "pring_memory_safe_tabular_baseline",
        })

    if not rows:
        return {"exported": 0, "reason": "no valid rows"}

    cypher = """
    UNWIND $rows AS row
    MATCH (c:Compound {cid: row.cid})
    MATCH (p:Protein {protein_id: row.protein_id})
    MERGE (c)-[r:PREDICTED_INTERACTION]->(p)
    SET r.score = row.score,
        r.predicted_label = row.predicted_label,
        r.model = row.model,
        r.compound_node_ref = row.compound_ref,
        r.protein_node_ref = row.protein_ref,
        r.updated_at = datetime()
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session(database=database) as session:
        session.execute_write(lambda tx: tx.run(cypher, rows=rows).consume())
    driver.close()
    return {"exported": len(rows), "neo4j_uri": uri, "database": database}


def _write_feature_recommendations(feature_columns: list[str], output_dir: Path) -> None:
    rows = []
    for col in feature_columns:
        action = "keep"
        reason = "usable numeric/categorical feature"
        if _is_leaky_feature_name(col):
            action = "drop_identifier"
            reason = "identifier-like column name"
        rows.append({"feature": col, "recommended_action": action, "reason": reason})
    pd.DataFrame(rows).to_csv(output_dir / "model_feature_recommendations.csv", index=False)


def run_pipeline(args: argparse.Namespace) -> dict:
    paths = resolve_run_paths(args.run_path)
    output_dir = Path(args.output_dir).resolve()
    report_dir = Path(args.report_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    try:
        pair_file = choose_pair_file(paths.ml_dir)
        logger.info("Reading pair file {}", pair_file)
        pairs = _read_csv(pair_file)
        if args.target_column not in pairs.columns:
            raise KeyError(f"Target column '{args.target_column}' not found in {pair_file}")

        pairs["_label"] = _coerce_label(pairs[args.target_column])
        supervised_mask = pairs["_label"].isin([0.0, 1.0])
        candidate_mask = pairs["_label"].eq(-1.0) | ~supervised_mask

        training_rows = pairs.loc[supervised_mask].copy()
        training_rows = _sample_by_label(training_rows, "_label", args.max_training_rows, args.random_state)
        scoring_rows = _choose_scoring_rows(
            pairs,
            supervised_mask,
            candidate_mask,
            args.prediction_scope,
            args.max_scoring_rows,
            args.random_state,
        )

        # Free the full pair table before feature merging/training. The full table is
        # the main source of Docker OOMs for multi-million-row candidate spaces.
        rows_total = int(len(pairs))
        rows_supervised = int(supervised_mask.sum())
        rows_candidate = int(candidate_mask.sum())
        del pairs, supervised_mask, candidate_mask

        training_enriched = _merge_node_features(
            training_rows,
            paths.ml_dir,
            use_node_features=args.use_node_features,
            max_feature_columns=args.max_node_feature_columns,
        )
        X_supervised, feature_columns = _safe_numeric_features(training_enriched, args.target_column)
        y_supervised = training_enriched["_label"].astype(int)

        summary: dict = {
            "run_root": str(paths.root),
            "ml_dir": str(paths.ml_dir),
            "pair_file": str(pair_file),
            "rows_total": rows_total,
            "rows_supervised": rows_supervised,
            "rows_candidate_or_unknown": rows_candidate,
            "training_rows_used": int(len(training_enriched)),
            "scoring_rows_selected": int(len(scoring_rows)),
            "prediction_scope": args.prediction_scope,
            "use_node_features": bool(args.use_node_features),
            "max_node_feature_columns": int(args.max_node_feature_columns),
            "feature_count": int(len(feature_columns)),
            "feature_columns_file": str(output_dir / "feature_columns.json"),
        }

        if X_supervised.empty:
            raise ValueError("No usable numeric/categorical model features were found in PRING ML exports.")

        if len(y_supervised) < 4 or y_supervised.nunique() < 2:
            summary["status"] = "not_trained"
            summary["reason"] = "Need at least two supervised classes and at least four supervised rows."
            (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            (report_dir / "modeling_summary.md").write_text(_render_markdown(summary), encoding="utf-8")
            logger.warning(summary["reason"])
            return summary

        X_train, X_test, y_train, y_test = _split_train_test(
            X_supervised,
            y_supervised,
            training_enriched["split"] if "split" in training_enriched.columns else None,
        )

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("classifier", RandomForestClassifier(
                n_estimators=args.n_estimators,
                random_state=args.random_state,
                class_weight="balanced_subsample",
                min_samples_leaf=args.min_samples_leaf,
                max_depth=args.max_depth,
                n_jobs=args.n_jobs,
            )),
        ])
        model.fit(X_train, y_train)
        test_scores = model.predict_proba(X_test)[:, 1]
        test_labels = (test_scores >= args.threshold).astype(int)
        metrics = _evaluate(y_test, test_scores, test_labels)

        scoring_enriched = _merge_node_features(
            scoring_rows,
            paths.ml_dir,
            use_node_features=args.use_node_features,
            max_feature_columns=args.max_node_feature_columns,
        )
        X_scoring, _ = _safe_numeric_features(scoring_enriched, args.target_column)
        X_scoring = _align_features(X_scoring, feature_columns)
        all_scores = model.predict_proba(X_scoring)[:, 1]

        metadata_cols = [c for c in [
            "compound_node_id", "protein_node_id", "compound_node_ref", "protein_node_ref",
            args.target_column, "split", "split_group", "candidate_sampling_method", "label_rule",
        ] if c in scoring_enriched.columns]
        predictions = scoring_enriched[metadata_cols].copy()
        predictions["score"] = all_scores
        predictions["predicted_label"] = (all_scores >= args.threshold).astype(int)
        predictions["is_supervised"] = scoring_enriched["_label"].isin([0.0, 1.0]).values
        predictions["is_candidate_or_unknown"] = ~predictions["is_supervised"].values
        predictions = predictions.sort_values("score", ascending=False)

        if args.max_predictions_file_rows > 0 and len(predictions) > args.max_predictions_file_rows:
            predictions_to_write = predictions.head(args.max_predictions_file_rows).copy()
        else:
            predictions_to_write = predictions

        model_path = output_dir / "pring_memory_safe_tabular_baseline.joblib"
        pred_path = output_dir / "predictions.csv"
        metrics_path = output_dir / "metrics.json"
        feature_path = output_dir / "feature_columns.json"

        joblib.dump(model, model_path)
        predictions_to_write.to_csv(pred_path, index=False)
        feature_path.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
        _write_feature_recommendations(feature_columns, output_dir)

        export_summary = None
        if args.export_neo4j:
            try:
                export_summary = export_predictions_to_neo4j(
                    predictions,
                    max_rows=args.max_neo4j_predictions,
                )
            except Exception as exc:
                export_summary = {"exported": 0, "error": str(exc)}
                logger.warning("Could not export predictions to Neo4j: {}", exc)

        summary.update({
            "status": "trained",
            "model_file": str(model_path),
            "predictions_file": str(pred_path),
            "predictions_file_rows": int(len(predictions_to_write)),
            "metrics_file": str(metrics_path),
            "train_rows": int(len(y_train)),
            "eval_rows": int(len(y_test)),
            "threshold": float(args.threshold),
            "metrics": metrics,
            "neo4j_export": export_summary,
            "memory_safety_note": (
                "Default mode trains on supervised rows and scores a bounded candidate sample. "
                "Increase MODEL_MAX_SCORING_ROWS or enable MODEL_USE_NODE_FEATURES only when Docker/HPC memory allows."
            ),
        })
        metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (report_dir / "modeling_summary.md").write_text(_render_markdown(summary), encoding="utf-8")
        logger.info("Modeling pipeline complete: {}", metrics_path)
        return summary
    finally:
        paths.cleanup()


def _render_markdown(summary: dict) -> str:
    lines = ["# PRING Modeling Summary", ""]
    for key in [
        "status", "run_root", "pair_file", "rows_total", "rows_supervised",
        "rows_candidate_or_unknown", "training_rows_used", "scoring_rows_selected",
        "prediction_scope", "use_node_features", "feature_count",
    ]:
        if key in summary:
            lines.append(f"- **{key}**: `{summary[key]}`")
    if "metrics" in summary:
        lines.extend(["", "## Metrics"])
        for k, v in summary["metrics"].items():
            lines.append(f"- **{k}**: `{v}`")
    if "memory_safety_note" in summary:
        lines.extend(["", "## Memory safety", "", summary["memory_safety_note"]])
    if "reason" in summary:
        lines.extend(["", f"**Reason:** {summary['reason']}"])
    if summary.get("neo4j_export") is not None:
        lines.extend(["", "## Neo4j export", "", "```json", json.dumps(summary["neo4j_export"], indent=2), "```"])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a memory-safe baseline model from PRING graph/ml exports.")
    parser.add_argument("--run-path", required=True, help="Path to a PRING run directory or ZIP.")
    parser.add_argument("--output-dir", default="models", help="Where to write model.joblib and predictions.csv.")
    parser.add_argument("--report-dir", default="reports/modeling", help="Where to write markdown reports.")
    parser.add_argument("--target-column", default="label", help="Supervision column in compound-target pair CSV.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for predicted_label.")
    parser.add_argument("--n-estimators", type=int, default=100, help="Random forest trees.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for scikit-learn. Use 1 in Docker Desktop to reduce memory.")
    parser.add_argument("--max-depth", type=int, default=None, help="Optional maximum random-forest depth.")
    parser.add_argument("--min-samples-leaf", type=int, default=2, help="Minimum samples per RF leaf.")
    parser.add_argument("--max-training-rows", type=int, default=100000, help="Maximum supervised rows used for training; stratified if sampled.")
    parser.add_argument("--max-scoring-rows", type=int, default=100000, help="Maximum rows selected for prediction scoring.")
    parser.add_argument("--max-predictions-file-rows", type=int, default=100000, help="Maximum rows written to predictions.csv.")
    parser.add_argument("--prediction-scope", choices=["supervised", "candidates", "supervised_plus_candidates", "all"], default="candidates", help="Which rows to score after training.")
    parser.add_argument("--use-node-features", action="store_true", help="Merge PRING node feature matrices. More informative but much more memory intensive.")
    parser.add_argument("--max-node-feature-columns", type=int, default=128, help="Max numeric node feature columns per feature table when --use-node-features is enabled.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for sampling and model training.")
    parser.add_argument("--export-neo4j", action="store_true", help="Write PREDICTED_INTERACTION relationships to Neo4j.")
    parser.add_argument("--max-neo4j-predictions", type=int, default=25000, help="Maximum predictions to export to Neo4j.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_pipeline(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
