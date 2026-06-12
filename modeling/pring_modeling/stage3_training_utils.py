from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except Exception:  # pragma: no cover
    accuracy_score = average_precision_score = balanced_accuracy_score = confusion_matrix = None
    f1_score = matthews_corrcoef = precision_score = recall_score = roc_auc_score = None


def label_stats(y: torch.Tensor | np.ndarray) -> dict[str, int | float]:
    arr = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    arr = arr.astype(float)
    n_pos = int((arr == 1).sum())
    n_neg = int((arr == 0).sum())
    n = int(n_pos + n_neg)
    return {
        "n": n,
        "positive": n_pos,
        "negative": n_neg,
        "positive_ratio": float(n_pos / n) if n else 0.0,
        "negative_ratio": float(n_neg / n) if n else 0.0,
    }


def maybe_balance_training_tensors(
    c_idx: torch.Tensor,
    p_idx: torch.Tensor,
    y: torch.Tensor,
    args: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Optionally oversample the minority class for LinkNeighborLoader supervision edges.

    This does not change graph message-passing edges. It only changes how often
    labeled Compound-Protein supervision pairs are sampled during training.
    """
    before = label_stats(y)
    if not bool(getattr(args, "balanced_batches", False)):
        return c_idx, p_idx, y, {"enabled": False, "before": before, "after": before}

    pos = torch.where(y.long() == 1)[0]
    neg = torch.where(y.long() == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return c_idx, p_idx, y, {"enabled": False, "reason": "single_class", "before": before, "after": before}

    # Default ratio = 1.0 gives one negative supervision pair per positive pair.
    ratio = float(getattr(args, "balance_ratio", 1.0))
    target_neg = max(len(neg), int(round(len(pos) * ratio)))
    if target_neg > len(neg):
        extra = neg[torch.randint(0, len(neg), (target_neg - len(neg),))]
        neg_bal = torch.cat([neg, extra], dim=0)
    else:
        perm = torch.randperm(len(neg))[:target_neg]
        neg_bal = neg[perm]

    idx = torch.cat([pos, neg_bal], dim=0)
    idx = idx[torch.randperm(len(idx))]
    c2, p2, y2 = c_idx[idx], p_idx[idx], y[idx]
    after = label_stats(y2)
    return c2, p2, y2, {"enabled": True, "before": before, "after": after, "balance_ratio": ratio}


def resolve_class_weights(y: torch.Tensor, args: Any) -> dict[int, float]:
    """Return per-class weights for labels 0 and 1.

    For this CYP450 case study, the minority class is often label 0
    inactive/weak. `balanced` gives n/(2*n_class). `negative_ratio` gives
    w0=n_pos/n_neg, w1=1. Manual values override the automatic weights.
    """
    stats = label_stats(y)
    n_pos = int(stats["positive"])
    n_neg = int(stats["negative"])
    n = max(1, int(stats["n"]))
    mode = str(getattr(args, "class_weighting", "balanced") or "none").lower()

    w0 = 1.0
    w1 = 1.0
    if mode == "balanced" and n_pos > 0 and n_neg > 0:
        w0 = n / (2.0 * n_neg)
        w1 = n / (2.0 * n_pos)
    elif mode == "negative_ratio" and n_pos > 0 and n_neg > 0:
        w0 = n_pos / n_neg
        w1 = 1.0

    manual_neg = getattr(args, "negative_class_weight", None)
    manual_pos = getattr(args, "positive_class_weight", None)
    if manual_neg is not None:
        w0 = float(manual_neg)
    if manual_pos is not None:
        w1 = float(manual_pos)
    return {0: float(w0), 1: float(w1)}


def weighted_or_focal_loss(logits: torch.Tensor, y: torch.Tensor, args: Any, class_weights: dict[int, float]) -> torch.Tensor:
    loss_name = str(getattr(args, "loss", "weighted_bce") or "weighted_bce").lower()
    y = y.float()
    if loss_name == "bce":
        return F.binary_cross_entropy_with_logits(logits, y)

    w0 = float(class_weights.get(0, 1.0))
    w1 = float(class_weights.get(1, 1.0))
    weights = torch.where(y >= 0.5, torch.full_like(y, w1), torch.full_like(y, w0))

    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    if loss_name == "weighted_bce":
        return (weights * bce).mean()

    if loss_name == "focal":
        gamma = float(getattr(args, "focal_gamma", 2.0))
        alpha = float(getattr(args, "focal_alpha", -1.0))
        probs = torch.sigmoid(logits)
        p_t = probs * y + (1.0 - probs) * (1.0 - y)
        focal = (1.0 - p_t).clamp(min=1e-6).pow(gamma)
        if alpha >= 0:
            alpha_t = alpha * y + (1.0 - alpha) * (1.0 - y)
        else:
            alpha_t = torch.ones_like(y)
        return (weights * alpha_t * focal * bce).mean()

    raise ValueError(f"Unsupported loss: {loss_name}. Use bce, weighted_bce, or focal.")


def amp_autocast(device: torch.device | str, enabled: bool):
    use_amp = bool(enabled) and str(device).startswith("cuda")
    if use_amp and hasattr(torch, "amp"):
        return torch.amp.autocast("cuda", enabled=True)
    if use_amp:
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def make_grad_scaler(device: torch.device | str, enabled: bool):
    use_amp = bool(enabled) and str(device).startswith("cuda")
    if hasattr(torch, "amp"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def extended_binary_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float | int | None]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    out: dict[str, float | int | None] = {}
    if len(y_true) == 0:
        return {
            "roc_auc": None,
            "average_precision": None,
            "accuracy": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "balanced_accuracy": None,
            "mcc": None,
            "specificity": None,
            "threshold": float(threshold),
        }

    if roc_auc_score is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    else:
        out["roc_auc"] = None
    out["average_precision"] = float(average_precision_score(y_true, y_score)) if average_precision_score is not None else None

    y_pred = (y_score >= threshold).astype(int)
    out["threshold"] = float(threshold)
    if accuracy_score is not None:
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
        out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) == 2 else None
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) == 2 else None
        if confusion_matrix is not None and len(np.unique(y_true)) == 2:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            out.update({
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
                "negative_precision": float(tn / (tn + fn)) if (tn + fn) else None,
                "positive_ratio": float(np.mean(y_true == 1)),
                "negative_ratio": float(np.mean(y_true == 0)),
            })
    return out


def _metric_value(metrics: dict[str, Any], metric: str) -> float:
    v = metrics.get(metric)
    if v is None:
        return float("-inf")
    return float(v)


def optimize_threshold(y_true, y_score, metric: str = "mcc") -> tuple[float, dict[str, Any]]:
    metric = str(metric or "mcc").lower()
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if metric in {"fixed", "none"} or len(np.unique(y_true)) < 2:
        m = extended_binary_metrics(y_true, y_score, 0.5)
        return 0.5, m
    best_threshold = 0.5
    best_metrics = extended_binary_metrics(y_true, y_score, 0.5)
    best_score = _metric_value(best_metrics, metric)
    for th in np.linspace(0.01, 0.99, 99):
        m = extended_binary_metrics(y_true, y_score, float(th))
        score = _metric_value(m, metric)
        if score > best_score:
            best_score = score
            best_threshold = float(th)
            best_metrics = m
    return best_threshold, best_metrics


def metric_for_checkpoint(metrics: dict[str, Any], metric: str) -> float:
    metric = str(metric or "mcc").lower()
    aliases = {
        "ap": "average_precision",
        "aupr": "average_precision",
        "auc": "roc_auc",
        "auroc": "roc_auc",
        "bal_acc": "balanced_accuracy",
    }
    metric = aliases.get(metric, metric)
    return _metric_value(metrics, metric)
