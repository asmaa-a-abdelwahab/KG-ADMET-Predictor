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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import coerce_binary_label, ensure_dir, read_table, save_json
from .model_diagnostics import (
    balanced_diagnostic_metrics,
    class_distribution,
    operating_point_metrics,
    per_group_binary_metrics,
    write_dataframe_if_not_empty,
)


PREDICTION_FILENAMES = [
    "holdout_eval_predictions.csv",
    "supervised_eval_predictions.csv",
    "eval_predictions.csv",
    "predictions.csv",
]


def fast_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Numpy-only binary metrics used by finalized V2 to avoid dependency/version edge cases."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score).astype(float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if y.size == 0:
        return {"roc_auc": None, "average_precision": None, "accuracy": None, "f1": None, "precision": None, "recall": None, "balanced_accuracy": None, "mcc": None, "specificity": None, "threshold": float(threshold)}
    pred = (s >= threshold).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else None
    neg_precision = tn / (tn + fn) if (tn + fn) else None
    accuracy = (tp + tn) / len(y) if len(y) else None
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    balanced = (recall + specificity) / 2 if specificity is not None else None
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    roc_auc = None
    ap = None
    if len(np.unique(y)) == 2:
        # Rank-based AUROC with average ranks for ties.
        order = np.argsort(s)
        ranks = np.empty_like(order, dtype=float)
        sorted_s = s[order]
        i = 0
        while i < len(s):
            j = i + 1
            while j < len(s) and sorted_s[j] == sorted_s[i]:
                j += 1
            ranks[order[i:j]] = (i + j + 1) / 2.0
            i = j
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        if n_pos and n_neg:
            roc_auc = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
        desc = np.argsort(-s)
        y_sorted = y[desc]
        cum_tp = np.cumsum(y_sorted == 1)
        denom_pr = np.arange(1, len(y_sorted) + 1)
        precision_at_k = cum_tp / denom_pr
        n_pos = max(1, int((y == 1).sum()))
        ap = float((precision_at_k * (y_sorted == 1)).sum() / n_pos)
    return {
        "threshold": float(threshold), "roc_auc": roc_auc, "average_precision": ap,
        "accuracy": float(accuracy) if accuracy is not None else None,
        "f1": float(f1), "precision": float(precision), "recall": float(recall),
        "balanced_accuracy": float(balanced) if balanced is not None else None,
        "mcc": float(mcc), "specificity": float(specificity) if specificity is not None else None,
        "negative_precision": float(neg_precision) if neg_precision is not None else None,
        "positive_ratio": float(np.mean(y == 1)), "negative_ratio": float(np.mean(y == 0)),
        "youden_j": float(recall + specificity - 1.0) if specificity is not None else None,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def _metric_value(metrics: dict[str, Any], metric: str) -> float:
    aliases = {"bal_acc": "balanced_accuracy", "youden": "youden_j", "auc": "roc_auc", "auroc": "roc_auc", "ap": "average_precision"}
    key = aliases.get(str(metric or "mcc").lower(), str(metric or "mcc").lower())
    value = metrics.get(key)
    try:
        value = float(value)
    except Exception:
        return -float("inf")
    return value if np.isfinite(value) else -float("inf")


def fast_optimize_binary_threshold(y_true: np.ndarray, y_score: np.ndarray, metric: str = "mcc", min_specificity: float = 0.0, min_recall: float = 0.0, grid_size: int = 201) -> tuple[float, dict[str, Any]]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score).astype(float)
    s = s[np.isfinite(s)]
    if s.size == 0 or np.unique(y).size < 2:
        return 0.5, fast_binary_metrics(y_true, y_score, 0.5)
    candidates = np.unique(np.concatenate([np.linspace(float(s.min()), float(s.max()), grid_size), np.quantile(s, np.linspace(0.001, 0.999, grid_size)), [0.5]]))
    best_any = (0.5, fast_binary_metrics(y_true, y_score, 0.5), -float("inf"))
    best_feasible: tuple[float, dict[str, Any], float] | None = None
    for th in candidates:
        m = fast_binary_metrics(y_true, y_score, float(th))
        val = _metric_value(m, metric)
        if val > best_any[2]:
            best_any = (float(th), m, val)
        feasible = True
        if min_specificity and (m.get("specificity") is None or float(m["specificity"]) < min_specificity):
            feasible = False
        if min_recall and (m.get("recall") is None or float(m["recall"]) < min_recall):
            feasible = False
        if feasible and (best_feasible is None or val > best_feasible[2]):
            best_feasible = (float(th), m, val)
    th, m, _ = best_feasible if best_feasible is not None else best_any
    m = dict(m)
    m.update({"selected_threshold": float(th), "threshold_selection_metric": metric, "min_specificity": float(min_specificity), "min_recall": float(min_recall), "constraint_satisfied": best_feasible is not None})
    return float(th), m


def _safe_name(text: str) -> str:
    out = str(text or "model").strip().lower()
    for ch in [" ", "/", "-", ":", "—", "(", ")", "[", "]"]:
        out = out.replace(ch, "_")
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "model"


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _compound_col(df: pd.DataFrame) -> str | None:
    return _first_existing(df, [
        "compound_node_ref", "compound_entity_id", "compound_node_id", "compound_id", "compound_key",
        "compound", "cid", "head", "source",
    ]) or next((c for c in df.columns if "compound" in c.lower()), None)


def _target_col(df: pd.DataFrame) -> str | None:
    return _first_existing(df, [
        "protein_node_ref", "target_node_ref", "protein_entity_id", "target_entity_id", "protein_node_id",
        "target_node_id", "protein_id", "target_id", "target_key", "protein", "target", "tail",
        "gene", "enzyme", "accession", "uniprot",
    ]) or next((c for c in df.columns if any(t in c.lower() for t in ["target", "protein", "enzyme", "gene"])), None)


def _label_col(df: pd.DataFrame) -> str | None:
    return _first_existing(df, ["label", "_label", "true_label", "y", "target_label", "activity_label"])


def _score_col(df: pd.DataFrame) -> str | None:
    for c in ["score", "probability", "predicted_probability", "prediction", "raw_score", "calibrated_score"]:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    numeric = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8]
    likely = [c for c in numeric if any(t in c.lower() for t in ["score", "prob", "pred"])]
    return likely[0] if likely else None


def _split_col(df: pd.DataFrame) -> str | None:
    return _first_existing(df, ["split", "data_split", "set", "partition"])


def _scaffold_col(df: pd.DataFrame) -> str | None:
    return _first_existing(df, ["bemis_murcko_scaffold", "murcko_scaffold", "scaffold", "scaffold_smiles", "generic_scaffold"])


def _source_group_col(df: pd.DataFrame) -> str | None:
    return _first_existing(df, [
        "assay_id", "bioassay_id", "aid", "source_id", "reference_id", "reference", "pmid",
        "evidence_assays", "evidence_references", "label_rule", "negative_source",
    ])


def find_prediction_files(outputs_root: str | Path) -> list[Path]:
    root = Path(outputs_root)
    priority = {name: index for index, name in enumerate(PREDICTION_FILENAMES)}
    found = [
        path
        for name in PREDICTION_FILENAMES
        for path in root.rglob(name)
    ]
    best_by_directory: dict[Path, Path] = {}
    for path in found:
        current = best_by_directory.get(path.parent.resolve())
        if current is None or priority[path.name] < priority[current.name]:
            best_by_directory[path.parent.resolve()] = path
    unique: list[Path] = []
    seen: set[Path] = set()
    for p in sorted(best_by_directory.values()):
        parent_parts = {part.lower() for part in p.parent.parts}
        if (
            "ensemble_stacked" in parent_parts
            or "finalized_v2" in parent_parts
            or any(part.startswith("seed_") for part in parent_parts)
        ):
            # Avoid training the final ensemble on a previous ensemble/finalized output unless explicitly passed.
            continue
        rp = p.resolve()
        if rp not in seen:
            unique.append(p)
            seen.add(rp)
    return unique


def _load_one_prediction(path: Path) -> pd.DataFrame | None:
    try:
        df = read_table(path)
    except Exception:
        return None
    if df.empty:
        return None
    c_col = _compound_col(df)
    t_col = _target_col(df)
    l_col = _label_col(df)
    s_col = _score_col(df)
    if not c_col or not t_col or not l_col or not s_col:
        return None
    y = coerce_binary_label(df[l_col])
    score = pd.to_numeric(df[s_col], errors="coerce")
    keep = y.isin([0.0, 1.0]) & score.notna()
    if not keep.any():
        return None
    model_base = path.parent.name
    if "model" in df.columns and df.loc[keep, "model"].dropna().size:
        model_base = str(df.loc[keep, "model"].dropna().iloc[0])
    score_name = f"score__{_safe_name(model_base)}"
    out = pd.DataFrame({
        "pair_key": df.loc[keep, c_col].astype(str) + "||" + df.loc[keep, t_col].astype(str),
        "compound_key": df.loc[keep, c_col].astype(str),
        "target_key": df.loc[keep, t_col].astype(str),
        "label": y.loc[keep].astype(int).to_numpy(),
        score_name: score.loc[keep].to_numpy(dtype=float),
    })
    split_col = _split_col(df)
    if split_col:
        out[f"split__{_safe_name(model_base)}"] = df.loc[keep, split_col].astype(str).to_numpy()
    scaffold_col = _scaffold_col(df)
    if scaffold_col:
        out["scaffold_key"] = df.loc[keep, scaffold_col].astype(str).to_numpy()
    source_col = _source_group_col(df)
    if source_col:
        out["source_group_key"] = df.loc[keep, source_col].astype(str).to_numpy()
    # When multiple rows for a pair appear, keep the mean score and first metadata values.
    agg: dict[str, Any] = {score_name: "mean", "compound_key": "first", "target_key": "first", "label": "first"}
    for c in out.columns:
        if c.startswith("split__") or c in {"scaffold_key", "source_group_key"}:
            agg[c] = "first"
    return out.groupby("pair_key", as_index=False).agg(agg)


def load_prediction_frame(prediction_files: list[str | Path], min_model_scores: int = 2) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for p in prediction_files:
        one = _load_one_prediction(Path(p))
        if one is not None and not one.empty:
            frames.append(one)
    if not frames:
        raise ValueError("No labelled prediction files with compound/target/label/score columns were found.")
    base = ["pair_key", "compound_key", "target_key", "label"]
    merged = frames[0]
    for frame in frames[1:]:
        meta_cols = [c for c in frame.columns if c not in base and not c.startswith("score__")]
        merged = merged.merge(frame, on=base, how="outer", suffixes=("", "_dup"))
        for col in [c for c in merged.columns if c.endswith("_dup")]:
            orig = col[:-4]
            if orig in merged.columns:
                merged[orig] = merged[orig].combine_first(merged[col])
            else:
                merged[orig] = merged[col]
            merged = merged.drop(columns=[col])
    score_cols = [c for c in merged.columns if c.startswith("score__")]
    merged = merged.dropna(subset=["label"])
    if not score_cols:
        raise ValueError("No model score columns were found after merging predictions.")
    merged = merged[merged[score_cols].notna().sum(axis=1) >= min(min_model_scores, len(score_cols))].copy()
    merged["label"] = merged["label"].astype(int)
    return merged.reset_index(drop=True)


def _make_classifier(name: str, seed: int, n_jobs: int) -> Pipeline:
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
            ("classifier", RandomForestClassifier(n_estimators=600, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed)),
        ])
    if name == "hist_gradient_boosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", HistGradientBoostingClassifier(max_iter=350, learning_rate=0.035, l2_regularization=1e-3, random_state=seed)),
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, class_weight="balanced", n_jobs=n_jobs, random_state=seed)),
    ])


def _infer_supplied_split(frame: pd.DataFrame) -> tuple[pd.Series | None, dict[str, Any]]:
    split_cols = [c for c in frame.columns if c.startswith("split__")]
    if not split_cols:
        return None, {"split_columns": [], "conflicting_rows": 0, "unknown_rows": len(frame)}
    normalized = pd.DataFrame(index=frame.index)
    for c in split_cols:
        s = frame[c].astype(str).str.lower().str.strip()
        normalized[c] = np.where(
            s.str.contains("test|holdout"),
            "test",
            np.where(
                s.str.contains("valid|val|dev"),
                "valid",
                np.where(s.str.contains("train"), "train", None),
            ),
        )
    conflicting = normalized.nunique(axis=1, dropna=True) > 1
    split = normalized.bfill(axis=1).iloc[:, 0]
    split.loc[conflicting] = None
    if split.notna().sum() == 0:
        return None, {
            "split_columns": split_cols,
            "conflicting_rows": int(conflicting.sum()),
            "unknown_rows": int(len(frame)),
        }
    return split.fillna("unknown"), {
        "split_columns": split_cols,
        "conflicting_rows": int(conflicting.sum()),
        "unknown_rows": int(split.isna().sum() + split.eq("unknown").sum()),
    }


def _assign_split(frame: pd.DataFrame, args: argparse.Namespace, seed: int) -> tuple[pd.Series, dict[str, Any]]:
    supplied, supplied_audit = _infer_supplied_split(frame)
    audit: dict[str, Any] = {
        "supplied_split_found": supplied is not None,
        "split_strategy": args.split_strategy,
        **supplied_audit,
    }
    supplied_complete = supplied is not None and not supplied.astype(str).eq("unknown").any()
    if supplied_complete and {"train", "test"}.issubset(set(supplied.astype(str))):
        split = supplied.copy()
        if "valid" not in set(split):
            train_idx = np.where(split == "train")[0]
            train_frame = frame.iloc[train_idx]
            if train_frame["compound_key"].nunique() > 1:
                inner = GroupShuffleSplit(
                    n_splits=1,
                    test_size=args.valid_size,
                    random_state=seed,
                )
                tr_relative, val_relative = next(
                    inner.split(
                        train_frame,
                        train_frame["label"],
                        groups=train_frame["compound_key"],
                    )
                )
                tr_sub, val_sub = train_idx[tr_relative], train_idx[val_relative]
            else:
                y_train = train_frame["label"]
                strat = y_train if y_train.value_counts().min() >= 2 else None
                tr_sub, val_sub = train_test_split(
                    train_idx,
                    test_size=args.valid_size,
                    stratify=strat,
                    random_state=seed,
                )
            split.iloc[val_sub] = "valid"
            split.iloc[tr_sub] = "train"
            audit["valid_created_from_train"] = True
        audit["diagnostic_split_created"] = False
        return split.reset_index(drop=True), audit

    if args.strict_leakage_free:
        raise ValueError(
            "Strict leakage-free mode requires complete, mutually consistent "
            "train/valid/test split columns for every base prediction row."
        )

    # Diagnostic fallback split. Prefer scaffold/source/compound grouping when available.
    split = pd.Series("train", index=frame.index, dtype=object)
    y = frame["label"].astype(int)
    group_key = None
    if args.split_strategy == "scaffold" and "scaffold_key" in frame.columns:
        group_key = frame["scaffold_key"].astype(str)
    elif args.split_strategy == "source" and "source_group_key" in frame.columns:
        group_key = frame["source_group_key"].astype(str)
    elif args.split_strategy in {"compound", "scaffold", "source"}:
        group_key = frame["compound_key"].astype(str)
    audit["diagnostic_split_created"] = True
    audit["diagnostic_split_warning"] = "No explicit split was found in the prediction files; generated a diagnostic split over already-generated predictions. Prefer out-of-fold base predictions for final publishable ensemble metrics."
    if group_key is not None and group_key.nunique() > 2:
        gss1 = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed)
        train_valid_idx, test_idx = next(gss1.split(frame, y, groups=group_key))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=args.valid_size, random_state=seed + 13)
        tv = frame.iloc[train_valid_idx]
        tr_rel, val_rel = next(gss2.split(tv, tv["label"], groups=group_key.iloc[train_valid_idx]))
        train_idx = train_valid_idx[tr_rel]
        valid_idx = train_valid_idx[val_rel]
        audit["group_column_used"] = "scaffold_key" if args.split_strategy == "scaffold" and "scaffold_key" in frame.columns else ("source_group_key" if args.split_strategy == "source" and "source_group_key" in frame.columns else "compound_key")
    else:
        idx = np.arange(len(frame))
        strat = y if y.value_counts().min() >= 2 else None
        train_valid_idx, test_idx = train_test_split(idx, test_size=args.test_size, stratify=strat, random_state=seed)
        y_tv = y.iloc[train_valid_idx]
        strat_tv = y_tv if y_tv.value_counts().min() >= 2 else None
        train_idx, valid_idx = train_test_split(train_valid_idx, test_size=args.valid_size, stratify=strat_tv, random_state=seed + 13)
        audit["group_column_used"] = None
    split.iloc[test_idx] = "test"
    split.iloc[valid_idx] = "valid"
    split.iloc[train_idx] = "train"
    return split.reset_index(drop=True), audit


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({"bin": i, "lo": lo, "hi": hi, "n": 0, "mean_probability": np.nan, "observed_fraction": np.nan, "abs_error": np.nan})
            continue
        conf = float(p[mask].mean())
        obs = float(y[mask].mean())
        err = abs(conf - obs)
        ece += (n / len(y)) * err if len(y) else 0.0
        rows.append({"bin": i, "lo": lo, "hi": hi, "n": n, "mean_probability": conf, "observed_fraction": obs, "abs_error": err})
    return float(ece), pd.DataFrame(rows)



class NumpyPlattCalibrator:
    """Small dependency-light Platt scaler trained with gradient descent."""
    def __init__(self, lr: float = 0.05, max_iter: int = 2000, l2: float = 1e-4):
        self.lr = lr
        self.max_iter = max_iter
        self.l2 = l2
        self.a = 1.0
        self.b = 0.0

    def fit(self, score: np.ndarray, y: np.ndarray):
        x = np.asarray(score, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size == 0:
            return self
        mu = float(x.mean())
        sd = float(x.std()) or 1.0
        z = (x - mu) / sd
        self.mu = mu
        self.sd = sd
        a, b = 1.0, 0.0
        # Mild target smoothing to avoid infinite logits on tiny validation sets.
        y_s = (y * (len(y) - 1) + 0.5) / max(1, len(y))
        for _ in range(self.max_iter):
            logits = np.clip(a * z + b, -35, 35)
            p = 1.0 / (1.0 + np.exp(-logits))
            err = p - y_s
            grad_a = float(np.mean(err * z) + self.l2 * a)
            grad_b = float(np.mean(err))
            a -= self.lr * grad_a
            b -= self.lr * grad_b
        self.a, self.b = float(a), float(b)
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        x = np.asarray(score, dtype=float)
        z = (x - getattr(self, 'mu', 0.0)) / (getattr(self, 'sd', 1.0) or 1.0)
        logits = np.clip(self.a * z + self.b, -35, 35)
        return 1.0 / (1.0 + np.exp(-logits))


def _fit_calibrator(method: str, y_valid: np.ndarray, valid_score: np.ndarray):
    method = str(method or "platt").lower()
    if method == "none":
        return None
    if method == "isotonic" and len(np.unique(valid_score)) >= 5:
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(valid_score, y_valid)
        return cal
    # Default: Platt/logistic scaling on a single score using a small numpy optimizer.
    return NumpyPlattCalibrator().fit(valid_score, y_valid)


def _apply_calibrator(cal: Any, score: np.ndarray) -> np.ndarray:
    if cal is None:
        return score
    if isinstance(cal, IsotonicRegression):
        return cal.predict(score)
    if hasattr(cal, "predict"):
        return cal.predict(score)
    return score


def _evaluate_single_scores(frame: pd.DataFrame, score_cols: list[str], valid: pd.DataFrame, test: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in score_cols:
        vmask = valid[col].notna().to_numpy()
        tmask = test[col].notna().to_numpy()
        if vmask.sum() < 2 or tmask.sum() < 2 or valid.loc[vmask, "label"].nunique() < 2 or test.loc[tmask, "label"].nunique() < 2:
            continue
        th, valid_m = fast_optimize_binary_threshold(
            valid.loc[vmask, "label"].to_numpy(),
            valid.loc[vmask, col].to_numpy(dtype=float),
            metric=args.threshold_selection,
            min_specificity=args.min_specificity,
            min_recall=args.min_recall,
        )
        test_m = fast_binary_metrics(test.loc[tmask, "label"].to_numpy(), test.loc[tmask, col].to_numpy(dtype=float), threshold=th)
        rows.append({
            "model_score_column": col,
            "n_valid": int(vmask.sum()),
            "n_test": int(tmask.sum()),
            "selected_threshold": float(th),
            "valid_mcc": valid_m.get("mcc"),
            **test_m,
        })
    return pd.DataFrame(rows).sort_values("mcc", ascending=False) if rows else pd.DataFrame()


def _per_target_meta_models(frame: pd.DataFrame, score_cols: list[str], args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, g in frame.groupby("target_key"):
        if len(g) < args.per_target_min_rows:
            continue
        if not {"train", "valid", "test"}.issubset(set(g["final_split"])):
            continue
        train = g[g["final_split"] == "train"]
        valid = g[g["final_split"] == "valid"]
        test = g[g["final_split"] == "test"]
        if min(train["label"].nunique(), valid["label"].nunique(), test["label"].nunique()) < 2:
            continue
        model = _make_classifier(args.meta_classifier, args.seed, args.n_jobs)
        model.fit(train[score_cols], train["label"].astype(int))
        valid_score = model.predict_proba(valid[score_cols])[:, 1]
        th, _ = fast_optimize_binary_threshold(valid["label"].to_numpy(), valid_score, metric=args.threshold_selection, min_specificity=args.min_specificity, min_recall=args.min_recall)
        test_score = model.predict_proba(test[score_cols])[:, 1]
        m = fast_binary_metrics(test["label"].to_numpy(), test_score, threshold=th)
        rows.append({"target_key": str(target), "train_rows": len(train), "valid_rows": len(valid), "test_rows": len(test), "selected_threshold": th, **m})
    return pd.DataFrame(rows).sort_values("mcc", ascending=False) if rows else pd.DataFrame()


def _external_validation(external_path: str | None, pred: pd.DataFrame, threshold: float) -> dict[str, Any]:
    if not external_path:
        return {"status": "not_requested"}
    path = Path(external_path)
    if not path.exists():
        return {"status": "skipped", "reason": "external_file_not_found", "path": str(path)}
    ext = read_table(path)
    c_col, t_col, l_col = _compound_col(ext), _target_col(ext), _label_col(ext)
    if not c_col or not t_col or not l_col:
        return {"status": "skipped", "reason": "missing_compound_target_label_columns", "path": str(path)}
    ext = ext.copy()
    ext["pair_key"] = ext[c_col].astype(str) + "||" + ext[t_col].astype(str)
    ext["external_label"] = coerce_binary_label(ext[l_col])
    ext = ext[ext["external_label"].isin([0.0, 1.0])]
    joined = ext.merge(pred[["pair_key", "calibrated_score"]], on="pair_key", how="inner")
    if joined.empty or joined["external_label"].nunique() < 2:
        return {"status": "skipped", "reason": "no_overlap_or_single_class", "external_rows": int(len(ext)), "overlap_rows": int(len(joined))}
    m = fast_binary_metrics(joined["external_label"].astype(int).to_numpy(), joined["calibrated_score"].to_numpy(dtype=float), threshold=threshold)
    return {"status": "computed", "path": str(path), "external_rows": int(len(ext)), "overlap_rows": int(len(joined)), "metrics": m, **m}


def run_one_seed(base_frame: pd.DataFrame, score_cols: list[str], args: argparse.Namespace, seed: int, seed_dir: Path) -> dict[str, Any]:
    seed_dir.mkdir(parents=True, exist_ok=True)
    frame = base_frame.copy()
    split, split_audit = _assign_split(frame, args, seed)
    frame["final_split"] = split.to_numpy()
    is_diagnostic_split = bool(split_audit.get("diagnostic_split_created"))
    frame["split_origin"] = (
        "generated_from_already_scored_predictions"
        if is_diagnostic_split
        else "supplied_split_registry"
    )
    frame["split_is_diagnostic"] = is_diagnostic_split
    train = frame[frame["final_split"] == "train"].reset_index(drop=True)
    valid = frame[frame["final_split"] == "valid"].reset_index(drop=True)
    test = frame[frame["final_split"] == "test"].reset_index(drop=True)
    if min(train["label"].nunique(), valid["label"].nunique(), test["label"].nunique()) < 2:
        raise ValueError("Train/valid/test split must each contain both classes for finalized evaluation.")

    model = _make_classifier(args.meta_classifier, seed, args.n_jobs)
    model.fit(train[score_cols], train["label"].astype(int).to_numpy())
    valid_score_raw = model.predict_proba(valid[score_cols])[:, 1]
    calibrator = _fit_calibrator(args.calibration, valid["label"].to_numpy(), valid_score_raw)
    valid_score = _apply_calibrator(calibrator, valid_score_raw)
    threshold, valid_metrics = fast_optimize_binary_threshold(
        valid["label"].to_numpy(),
        valid_score,
        metric=args.threshold_selection,
        min_specificity=args.min_specificity,
        min_recall=args.min_recall,
    )
    test_score_raw = model.predict_proba(test[score_cols])[:, 1]
    test_score = _apply_calibrator(calibrator, test_score_raw)
    metrics = fast_binary_metrics(test["label"].to_numpy(), test_score, threshold=threshold)
    brier = float(np.mean((test_score - test["label"].to_numpy(dtype=float)) ** 2))
    ece, cal_bins = expected_calibration_error(test["label"].to_numpy(), test_score, n_bins=args.calibration_bins)
    operating_points = operating_point_metrics(
        test["label"].to_numpy(),
        test_score,
        fast_binary_metrics,
        min_specificity=max(args.min_specificity, args.report_min_specificity),
        high_specificity=args.report_high_specificity,
        min_recall=args.report_min_recall,
    )
    balanced_metrics = balanced_diagnostic_metrics(
        test["label"].to_numpy(),
        test_score,
        fast_binary_metrics,
        threshold=threshold,
        seed=seed,
        max_per_class=args.balanced_eval_max_per_class,
    )
    pred = test.copy()
    pred["raw_ensemble_score"] = test_score_raw
    pred["calibrated_score"] = test_score
    pred["score"] = test_score
    pred["predicted_label"] = (test_score >= threshold).astype(int)
    pred["decision_threshold"] = float(threshold)
    pred["model"] = f"finalized_ensemble_{args.meta_classifier}"
    pred["stage"] = "Finalized V2 — leakage-aware calibrated ensemble"
    pred["base_score_mean"] = pred[score_cols].mean(axis=1)
    pred["base_score_std"] = pred[score_cols].std(axis=1)
    pred["probability_margin"] = np.abs(pred["calibrated_score"] - 0.5)
    pred["uncertainty_rank"] = pred["base_score_std"].rank(method="average", ascending=False)
    pred = pred.sort_values(["target_key", "calibrated_score"], ascending=[True, False]).reset_index(drop=True)
    pred.to_csv(seed_dir / "predictions.csv", index=False)
    frame.to_csv(seed_dir / "finalized_training_frame.csv", index=False)
    cal_bins.to_csv(seed_dir / "calibration_bins.csv", index=False)
    common_metrics = _evaluate_single_scores(frame, score_cols, valid, test, args)
    write_dataframe_if_not_empty(common_metrics, seed_dir / "common_test_model_metrics.csv")
    per_target = per_group_binary_metrics(pred, pred["label"].to_numpy(), pred["calibrated_score"].to_numpy(), fast_binary_metrics, threshold=threshold, group_col="target_key")
    write_dataframe_if_not_empty(per_target, seed_dir / "per_target_metrics.csv")
    per_target_meta = _per_target_meta_models(frame, score_cols, args)
    write_dataframe_if_not_empty(per_target_meta, seed_dir / "per_target_ensemble_metrics.csv")
    topk = pred.groupby("target_key", group_keys=False).head(args.top_k_per_target).copy()
    topk.to_csv(seed_dir / "top_k_by_target.csv", index=False)
    uncertainty = pred.sort_values(["base_score_std", "probability_margin"], ascending=[False, True]).head(args.uncertain_top_n)
    uncertainty.to_csv(seed_dir / "most_uncertain_predictions.csv", index=False)
    external = _external_validation(args.external_labels, pred, threshold)
    joblib.dump({"model": model, "calibrator": calibrator, "score_columns": score_cols, "threshold": threshold}, seed_dir / "finalized_ensemble.joblib")
    summary: dict[str, Any] = {
        "stage": "Finalized V2 — leakage-aware calibrated ensemble",
        "model": f"finalized_ensemble_{args.meta_classifier}",
        "status": "diagnostic_only" if is_diagnostic_split else "trained",
        "publishable": not is_diagnostic_split,
        "seed": int(seed),
        "score_columns": score_cols,
        "class_distribution": {
            "all": class_distribution(frame["label"]),
            "train": class_distribution(train["label"]),
            "valid": class_distribution(valid["label"]),
            "test": class_distribution(test["label"]),
        },
        "split_audit": split_audit,
        "selected_threshold": float(threshold),
        "threshold_selection": args.threshold_selection,
        "validation_metrics": valid_metrics,
        "metrics": {**metrics, "brier_score": brier, "expected_calibration_error": ece},
        "operating_points": operating_points,
        "balanced_diagnostic_metrics": balanced_metrics,
        "per_target_metrics": per_target.to_dict(orient="records") if not per_target.empty else [],
        "per_target_ensemble_metrics_file": str(seed_dir / "per_target_ensemble_metrics.csv") if not per_target_meta.empty else None,
        "common_test_model_metrics_file": str(seed_dir / "common_test_model_metrics.csv") if not common_metrics.empty else None,
        "calibration": {"method": args.calibration, "brier_score": brier, "expected_calibration_error": ece, "bins_file": str(seed_dir / "calibration_bins.csv")},
        "uncertainty": {"method": "base_model_score_std_and_probability_margin", "file": str(seed_dir / "most_uncertain_predictions.csv")},
        "candidate_ranking_file": str(seed_dir / "top_k_by_target.csv"),
        "external_validation": external,
        "predictions_file": str(seed_dir / "predictions.csv"),
        "training_frame_file": str(seed_dir / "finalized_training_frame.csv"),
        "model_file": str(seed_dir / "finalized_ensemble.joblib"),
        **metrics,
        "brier_score": brier,
        "expected_calibration_error": ece,
    }
    save_json(summary, seed_dir / "metrics.json")
    return summary


def aggregate_seed_metrics(summaries: list[dict[str, Any]], out_dir: Path) -> pd.DataFrame:
    rows = []
    metric_keys = ["mcc", "balanced_accuracy", "roc_auc", "average_precision", "specificity", "negative_precision", "accuracy", "f1", "precision", "recall", "brier_score", "expected_calibration_error"]
    for s in summaries:
        row = {"seed": s.get("seed")}
        for k in metric_keys:
            row[k] = s.get("metrics", {}).get(k, s.get(k))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "seed_metrics.csv", index=False)
    stats = []
    for k in metric_keys:
        vals = pd.to_numeric(df[k], errors="coerce") if k in df else pd.Series(dtype=float)
        stats.append({"metric": k, "mean": vals.mean(), "std": vals.std(ddof=1), "min": vals.min(), "max": vals.max(), "n": vals.notna().sum()})
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(out_dir / "seed_metric_summary.csv", index=False)
    return stats_df


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = ensure_dir(args.output_dir)
    files = [Path(p) for p in args.predictions]
    if args.outputs_root:
        files.extend(find_prediction_files(args.outputs_root))
    seen: set[Path] = set()
    files = [p for p in files if p.exists() and not (p.resolve() in seen or seen.add(p.resolve()))]
    if not files:
        raise ValueError("No prediction files found. Run Stage 1/2/3 first or pass --predictions.")
    frame = load_prediction_frame(files, min_model_scores=args.min_model_scores)
    score_cols = [c for c in frame.columns if c.startswith("score__")]
    if len(score_cols) < args.min_model_scores:
        raise ValueError(f"Need at least {args.min_model_scores} model score columns; found {len(score_cols)}")
    if args.strict_leakage_free:
        missing_score_rows = int(frame[score_cols].isna().any(axis=1).sum())
        if missing_score_rows:
            raise ValueError(
                f"Strict leakage-free mode requires every base-model score for every row; "
                f"{missing_score_rows} row(s) have missing component scores."
            )
    frame.to_csv(out_dir / "merged_base_prediction_frame.csv", index=False)
    seed_values = [int(s) for s in str(args.seeds).replace(",", " ").split() if str(s).strip()]
    if not seed_values:
        seed_values = [args.seed]
    summaries = []
    for seed in seed_values:
        seed_dir = out_dir / f"seed_{seed}"
        summaries.append(run_one_seed(frame, score_cols, args, seed, seed_dir))
    seed_summary = aggregate_seed_metrics(summaries, out_dir)
    best = max(
        summaries,
        key=lambda x: float(
            x.get("validation_metrics", {}).get("mcc")
            if x.get("validation_metrics", {}).get("mcc") is not None
            else -999
        ),
    )
    publishable = all(bool(summary.get("publishable")) for summary in summaries)
    final_summary = {
        "stage": "Finalized V2 — leakage-aware calibrated ensemble",
        "model": f"finalized_ensemble_{args.meta_classifier}",
        "status": "trained" if publishable else "diagnostic_only",
        "publishable": publishable,
        "implementation": "improved_v2",
        "input_prediction_files": [str(p) for p in files],
        "score_columns": score_cols,
        "rows_used": int(len(frame)),
        "seeds": seed_values,
        "best_seed": best.get("seed"),
        "best_seed_selection_source": "validation_mcc",
        "best_seed_validation_metrics": best.get("validation_metrics", {}),
        "split_audit": best.get("split_audit", {}),
        "best_seed_metrics_file": str(out_dir / f"seed_{best.get('seed')}" / "metrics.json"),
        "seed_metrics_file": str(out_dir / "seed_metrics.csv"),
        "seed_metric_summary_file": str(out_dir / "seed_metric_summary.csv"),
        "metric_summary": seed_summary.to_dict(orient="records"),
        "metrics": best.get("metrics", {}),
        **{k: best.get("metrics", {}).get(k) for k in ["mcc", "balanced_accuracy", "roc_auc", "average_precision", "specificity", "negative_precision", "accuracy", "f1", "precision", "recall"]},
    }
    save_json(final_summary, out_dir / "metrics.json")
    return final_summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Finalized V2 validation: leakage-aware ensemble, common-test evaluation, calibration, uncertainty, per-target metrics, seed aggregation, and ranking outputs.")
    p.add_argument("--outputs-root", default=None, help="Root directory containing Stage 1/2/3 model outputs.")
    p.add_argument("--predictions", nargs="*", default=[], help="Explicit labelled prediction files to merge.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--meta-classifier", choices=["extra_trees", "hist_gradient_boosting", "random_forest", "logistic_regression"], default="extra_trees")
    p.add_argument("--min-model-scores", type=int, default=2)
    p.add_argument("--strict-leakage-free", action="store_true", help="Fail if supplied train/valid/test split annotations are absent.")
    p.add_argument("--split-strategy", choices=["random", "compound", "scaffold", "source"], default="compound")
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--valid-size", type=float, default=0.20, help="Validation fraction of the non-test data when creating diagnostic splits.")
    p.add_argument("--threshold-selection", choices=["mcc", "balanced_accuracy", "youden", "f1", "accuracy", "recall", "specificity"], default="mcc")
    p.add_argument("--min-specificity", type=float, default=0.50)
    p.add_argument("--min-recall", type=float, default=0.0)
    p.add_argument("--report-min-specificity", type=float, default=0.50)
    p.add_argument("--report-high-specificity", type=float, default=0.80)
    p.add_argument("--report-min-recall", type=float, default=0.80)
    p.add_argument("--balanced-eval-max-per-class", type=int, default=0)
    p.add_argument("--calibration", choices=["platt", "isotonic", "none"], default="platt")
    p.add_argument("--calibration-bins", type=int, default=10)
    p.add_argument("--per-target-min-rows", type=int, default=100)
    p.add_argument("--top-k-per-target", type=int, default=50)
    p.add_argument("--uncertain-top-n", type=int, default=200)
    p.add_argument("--external-labels", default=None, help="Optional external validation CSV with compound, target, and label columns.")
    p.add_argument("--seeds", default="42", help="Space- or comma-separated seeds for repeated finalized ensemble evaluation.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=1)
    return p


def main(argv: Iterable[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
