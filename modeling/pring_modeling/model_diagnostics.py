from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


MetricFn = Callable[[np.ndarray, np.ndarray, float], dict[str, Any]]


def _metric_value(metrics: dict[str, Any], metric: str) -> float:
    aliases = {
        "bal_acc": "balanced_accuracy",
        "youden": "youden_j",
        "youden_j": "youden_j",
        "auroc": "roc_auc",
        "auc": "roc_auc",
        "ap": "average_precision",
        "aupr": "average_precision",
    }
    key = aliases.get(str(metric or "mcc").lower(), str(metric or "mcc").lower())
    if key == "youden_j" and "youden_j" not in metrics:
        rec = metrics.get("recall")
        spec = metrics.get("specificity")
        if rec is not None and spec is not None:
            return float(rec) + float(spec) - 1.0
    value = metrics.get(key)
    if value is None:
        return -float("inf")
    try:
        value = float(value)
    except Exception:
        return -float("inf")
    if np.isnan(value):
        return -float("inf")
    return value


def threshold_candidates(y_score: np.ndarray, grid_size: int = 201) -> np.ndarray:
    y_score = np.asarray(y_score, dtype=float)
    y_score = y_score[np.isfinite(y_score)]
    if y_score.size == 0:
        return np.array([0.5], dtype=float)
    lo, hi = float(np.min(y_score)), float(np.max(y_score))
    if lo == hi:
        return np.array([lo], dtype=float)
    linear = np.linspace(lo, hi, max(3, int(grid_size)))
    quantiles = np.quantile(y_score, np.linspace(0.001, 0.999, max(3, int(grid_size))))
    return np.unique(np.concatenate([linear, quantiles, np.array([0.5], dtype=float)]))


def choose_threshold_with_constraints(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str,
    metric_fn: MetricFn,
    *,
    min_specificity: float = 0.0,
    min_recall: float = 0.0,
    grid_size: int = 201,
    fallback_threshold: float = 0.5,
) -> tuple[float, dict[str, Any]]:
    """Choose a threshold by metric while optionally enforcing specificity/recall.

    This is intentionally separate from model fitting so the same trained model
    can be reported at screening-oriented and specificity-oriented operating
    points. If no threshold satisfies the requested constraints, the best
    unconstrained threshold is returned and the metrics include
    `constraint_satisfied=False`.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    finite = np.isfinite(y_score)
    y_true = y_true[finite]
    y_score = y_score[finite]
    if y_true.size == 0 or np.unique(y_true).size < 2:
        m = metric_fn(y_true, y_score, float(fallback_threshold))
        m.update({
            "threshold_selection_metric": str(metric or "fixed"),
            "min_specificity": float(min_specificity),
            "min_recall": float(min_recall),
            "constraint_satisfied": False,
            "constraint_reason": "single_class_or_empty",
        })
        return float(fallback_threshold), m

    best_any_th = float(fallback_threshold)
    best_any_metrics = metric_fn(y_true, y_score, best_any_th)
    best_any_score = _metric_value(best_any_metrics, metric)
    best_feasible_th: float | None = None
    best_feasible_metrics: dict[str, Any] | None = None
    best_feasible_score = -float("inf")

    for th in threshold_candidates(y_score, grid_size=grid_size):
        m = metric_fn(y_true, y_score, float(th))
        score = _metric_value(m, metric)
        if score > best_any_score:
            best_any_score = score
            best_any_th = float(th)
            best_any_metrics = m
        spec = m.get("specificity")
        rec = m.get("recall")
        feasible = True
        if min_specificity and (spec is None or float(spec) < float(min_specificity)):
            feasible = False
        if min_recall and (rec is None or float(rec) < float(min_recall)):
            feasible = False
        if feasible and score > best_feasible_score:
            best_feasible_score = score
            best_feasible_th = float(th)
            best_feasible_metrics = m

    if best_feasible_metrics is not None:
        best_feasible_metrics = dict(best_feasible_metrics)
        best_feasible_metrics.update({
            "threshold_selection_metric": str(metric or "mcc"),
            "min_specificity": float(min_specificity),
            "min_recall": float(min_recall),
            "constraint_satisfied": True,
        })
        return float(best_feasible_th), best_feasible_metrics

    best_any_metrics = dict(best_any_metrics)
    best_any_metrics.update({
        "threshold_selection_metric": str(metric or "mcc"),
        "min_specificity": float(min_specificity),
        "min_recall": float(min_recall),
        "constraint_satisfied": False,
        "constraint_reason": "no_threshold_met_constraints",
    })
    return float(best_any_th), best_any_metrics


def operating_point_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: MetricFn,
    *,
    min_specificity: float = 0.50,
    high_specificity: float = 0.80,
    min_recall: float = 0.50,
) -> dict[str, dict[str, Any]]:
    points: dict[str, dict[str, Any]] = {}
    for name, metric, spec, rec in [
        ("fixed_0_5", "fixed", 0.0, 0.0),
        ("max_mcc", "mcc", 0.0, 0.0),
        ("balanced_accuracy", "balanced_accuracy", 0.0, 0.0),
        ("screening_high_recall", "recall", 0.0, min_recall),
        ("specificity_constrained", "mcc", min_specificity, 0.0),
        ("high_specificity", "mcc", high_specificity, 0.0),
    ]:
        if metric == "fixed":
            m = metric_fn(y_true, y_score, 0.5)
            m["selected_threshold"] = 0.5
            m["operating_point"] = name
        else:
            th, m = choose_threshold_with_constraints(y_true, y_score, metric, metric_fn, min_specificity=spec, min_recall=rec)
            m["selected_threshold"] = float(th)
            m["operating_point"] = name
        points[name] = m
    return points


def class_distribution(y: Any) -> dict[str, Any]:
    arr = np.asarray(y).astype(float)
    arr = arr[np.isfinite(arr)]
    n_pos = int((arr == 1).sum())
    n_neg = int((arr == 0).sum())
    n = n_pos + n_neg
    return {
        "n": int(n),
        "positive": n_pos,
        "negative": n_neg,
        "positive_ratio": float(n_pos / n) if n else 0.0,
        "negative_ratio": float(n_neg / n) if n else 0.0,
    }


def infer_target_group_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "protein_node_ref", "target_node_ref", "protein_ref", "target_ref",
        "protein_entity_id", "target_entity_id", "protein_node_id", "target_node_id",
        "protein_id", "target_id", "accession", "uniprot", "tail", "target",
    ]
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    # Common pair exports may contain one of five CYP names in a text column.
    for col in df.columns:
        if df[col].dtype == object and any(tok in col.lower() for tok in ["protein", "target", "gene", "enzyme"]):
            return col
    return None


def per_group_binary_metrics(
    frame: pd.DataFrame,
    y_true: Any,
    y_score: Any,
    metric_fn: MetricFn,
    *,
    threshold: float,
    group_col: str | None = None,
    min_rows: int = 2,
) -> pd.DataFrame:
    if frame is None or len(frame) == 0:
        return pd.DataFrame()
    group_col = group_col or infer_target_group_column(frame)
    if not group_col or group_col not in frame.columns:
        return pd.DataFrame()
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score).astype(float)
    groups = frame[group_col].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for group in sorted(pd.unique(groups)):
        idx = np.where(groups == group)[0]
        if len(idx) < min_rows or np.unique(y[idx]).size < 2:
            continue
        m = metric_fn(y[idx], s[idx], threshold)
        m.update({"group_column": group_col, "group": str(group), "n": int(len(idx))})
        rows.append(m)
    return pd.DataFrame(rows)


def balanced_diagnostic_metrics(
    y_true: Any,
    y_score: Any,
    metric_fn: MetricFn,
    *,
    threshold: float,
    seed: int = 42,
    max_per_class: int = 0,
) -> dict[str, Any]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score).astype(float)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return {"status": "skipped", "reason": "single_class", **class_distribution(y)}
    n = min(len(pos), len(neg))
    if max_per_class and max_per_class > 0:
        n = min(n, int(max_per_class))
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    rng.shuffle(idx)
    m = metric_fn(y[idx], s[idx], threshold)
    m.update({"status": "computed", "n_per_class": int(n), "n": int(2 * n)})
    return m


def negative_source_summary(df: pd.DataFrame, label_col: str = "_label") -> dict[str, Any]:
    if df is None or df.empty or label_col not in df.columns:
        return {}
    out: dict[str, Any] = {"overall": class_distribution(df[label_col].to_numpy())}
    for col in ["negative_source", "candidate_sampling_method", "label_rule", "stage_use", "split", "split_group"]:
        if col not in df.columns:
            continue
        rows = []
        for value, group in df.groupby(col, dropna=False):
            dist = class_distribution(group[label_col].to_numpy())
            dist[col] = str(value)
            rows.append(dist)
        out[col] = rows
    return out


def write_dataframe_if_not_empty(df: pd.DataFrame, path: str | Path) -> str | None:
    if df is None or df.empty:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)
