from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import binary_metrics, coerce_binary_label, ensure_dir, optimize_binary_threshold, read_table, save_json
from .model_diagnostics import (
    balanced_diagnostic_metrics,
    class_distribution,
    infer_target_group_column,
    operating_point_metrics,
    per_group_binary_metrics,
    write_dataframe_if_not_empty,
)


def _safe_name(text: str) -> str:
    return str(text or "model").strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")


def find_ensemble_prediction_files(outputs_root: str | Path) -> list[Path]:
    root = Path(outputs_root)
    priority_names = [
        "holdout_eval_predictions.csv",
        "supervised_eval_predictions.csv",
        "eval_predictions.csv",
        "predictions.csv",
    ]
    found: list[Path] = []
    for name in priority_names:
        found.extend(root.rglob(name))
    unique: list[Path] = []
    seen: set[Path] = set()
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def _compound_col(df: pd.DataFrame) -> str | None:
    for c in ["compound_node_ref", "compound_entity_id", "compound_node_id", "compound_id", "compound", "head", "source"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "compound" in c.lower():
            return c
    return None


def _label_col(df: pd.DataFrame) -> str | None:
    for c in ["_label", "label", "true_label", "y", "target", "activity_label"]:
        if c in df.columns:
            return c
    return None


def _score_col(df: pd.DataFrame) -> str | None:
    for c in ["score", "probability", "prediction", "predicted_probability", "raw_score"]:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    return None


def _load_one_prediction(path: Path, model_name: str | None = None) -> pd.DataFrame | None:
    try:
        df = read_table(path)
    except Exception:
        return None
    if df.empty:
        return None
    c_col = _compound_col(df)
    t_col = infer_target_group_column(df)
    l_col = _label_col(df)
    s_col = _score_col(df)
    if not c_col or not t_col or not l_col or not s_col:
        return None
    y = coerce_binary_label(df[l_col])
    keep = y.isin([0.0, 1.0])
    if not keep.any():
        return None
    name = model_name or str(df["model"].dropna().iloc[0]) if "model" in df.columns and df["model"].dropna().size else path.parent.name
    score_name = f"score__{_safe_name(name)}"
    out = pd.DataFrame({
        "pair_key": df.loc[keep, c_col].astype(str) + "||" + df.loc[keep, t_col].astype(str),
        "compound_key": df.loc[keep, c_col].astype(str),
        "target_key": df.loc[keep, t_col].astype(str),
        "label": y.loc[keep].astype(int).to_numpy(),
        score_name: pd.to_numeric(df.loc[keep, s_col], errors="coerce").to_numpy(dtype=float),
    })
    out = out.dropna(subset=[score_name])
    return out.drop_duplicates(subset=["pair_key", score_name])


def load_ensemble_frame(prediction_files: list[str | Path]) -> pd.DataFrame:
    frames = []
    for p in prediction_files:
        one = _load_one_prediction(Path(p))
        if one is not None and not one.empty:
            frames.append(one)
    if not frames:
        raise ValueError("No labelled prediction files with pair keys and scores were found.")
    base_cols = ["pair_key", "compound_key", "target_key", "label"]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=base_cols, how="outer")
    # Some stages may report the same pair with label; keep rows with a known label and at least two model scores.
    score_cols = [c for c in merged.columns if c.startswith("score__")]
    merged = merged.dropna(subset=["label"])
    merged = merged[merged[score_cols].notna().sum(axis=1) >= min(2, len(score_cols))].copy()
    merged["label"] = merged["label"].astype(int)
    return merged.reset_index(drop=True)


def _make_meta_classifier(name: str, seed: int, n_jobs: int) -> Pipeline:
    name = str(name or "extra_trees").lower()
    if name == "logistic_regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=n_jobs, random_state=seed)),
        ])
    if name == "random_forest":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed)),
        ])
    if name == "hist_gradient_boosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, l2_regularization=1e-3, random_state=seed)),
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", ExtraTreesClassifier(n_estimators=600, min_samples_leaf=3, class_weight="balanced", n_jobs=n_jobs, random_state=seed)),
    ])


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = ensure_dir(args.output_dir)
    files = [Path(p) for p in args.predictions]
    if args.outputs_root:
        files.extend(find_ensemble_prediction_files(args.outputs_root))
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    files = [p for p in files if not (p.resolve() in seen or seen.add(p.resolve()))]
    frame = load_ensemble_frame(files)
    score_cols = [c for c in frame.columns if c.startswith("score__")]
    if len(score_cols) < 2:
        raise ValueError("The ensemble needs at least two labelled model score columns.")
    frame.to_csv(out_dir / "ensemble_training_frame.csv", index=False)

    idx = np.arange(len(frame))
    stratify = frame["label"] if frame["label"].value_counts().min() >= 2 else None
    train_idx, test_idx = train_test_split(idx, test_size=args.test_size, stratify=stratify, random_state=args.seed)
    train, test = frame.iloc[train_idx].reset_index(drop=True), frame.iloc[test_idx].reset_index(drop=True)
    X_train, y_train = train[score_cols], train["label"].astype(int).to_numpy()
    X_test, y_test = test[score_cols], test["label"].astype(int).to_numpy()

    model = _make_meta_classifier(args.meta_classifier, args.seed, args.n_jobs)
    model.fit(X_train, y_train)
    test_score = model.predict_proba(X_test)[:, 1]
    threshold, metrics = optimize_binary_threshold(
        y_test,
        test_score,
        metric=args.threshold_selection,
        min_specificity=args.min_specificity,
        min_recall=args.min_recall,
    )
    operating_points = operating_point_metrics(
        y_test,
        test_score,
        binary_metrics,
        min_specificity=max(args.min_specificity, args.report_min_specificity),
        high_specificity=args.report_high_specificity,
        min_recall=args.report_min_recall,
    )
    balanced_metrics = balanced_diagnostic_metrics(
        y_test,
        test_score,
        binary_metrics,
        threshold=threshold,
        seed=args.seed,
        max_per_class=args.balanced_eval_max_per_class,
    )
    pred = test.copy()
    pred["score"] = test_score
    pred["predicted_label"] = (test_score >= threshold).astype(int)
    pred["decision_threshold"] = float(threshold)
    pred["model"] = f"ensemble_{args.meta_classifier}"
    pred["stage"] = "Ensemble — stacked meta-classifier"
    pred.to_csv(out_dir / "predictions.csv", index=False)
    per_target = per_group_binary_metrics(pred, y_test, test_score, binary_metrics, threshold=threshold, group_col="target_key")
    per_target_metrics_file = write_dataframe_if_not_empty(per_target, out_dir / "per_target_metrics.csv")
    model_path = out_dir / "ensemble_model.joblib"
    joblib.dump(model, model_path)
    summary = {
        "stage": "Ensemble — stacked meta-classifier",
        "model": f"ensemble_{args.meta_classifier}",
        "status": "trained",
        "input_prediction_files": [str(p) for p in files],
        "score_columns": score_cols,
        "rows_used": int(len(frame)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "class_distribution": {"all": class_distribution(frame["label"]), "train": class_distribution(y_train), "test": class_distribution(y_test)},
        "selected_threshold": float(threshold),
        "threshold_selection": args.threshold_selection,
        "metrics": metrics,
        "operating_points": operating_points,
        "balanced_diagnostic_metrics": balanced_metrics,
        "per_target_metrics": per_target.to_dict(orient="records") if per_target_metrics_file else [],
        "per_target_metrics_file": per_target_metrics_file,
        "predictions_file": str(out_dir / "predictions.csv"),
        "training_frame_file": str(out_dir / "ensemble_training_frame.csv"),
        "model_file": str(model_path),
        **{k: v for k, v in metrics.items()},
    }
    save_json(summary, out_dir / "metrics.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a stacked PRING ensemble from labelled Stage 1/2/3 prediction outputs.")
    p.add_argument("--predictions", nargs="*", default=[], help="Prediction/eval CSV files. If omitted, use --outputs-root discovery.")
    p.add_argument("--outputs-root", default=None, help="Root directory containing stage output folders.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--meta-classifier", choices=["extra_trees", "hist_gradient_boosting", "random_forest", "logistic_regression"], default="extra_trees")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--threshold-selection", choices=["mcc", "balanced_accuracy", "youden", "f1", "accuracy", "recall", "specificity"], default="mcc")
    p.add_argument("--min-specificity", type=float, default=0.50)
    p.add_argument("--min-recall", type=float, default=0.0)
    p.add_argument("--report-min-specificity", type=float, default=0.50)
    p.add_argument("--report-high-specificity", type=float, default=0.80)
    p.add_argument("--report-min-recall", type=float, default=0.80)
    p.add_argument("--balanced-eval-max-per-class", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: Iterable[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
