from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from pring_modeling.final_validation import NumpyPlattCalibrator, expected_calibration_error

SCORE_COLUMNS = ["score__stage3_rgcn_sampled", "score__stage3_hgt_sampled"]


def _optimize_mcc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(np.concatenate(([0.0], y_prob, [1.0])))
    best_threshold, best_mcc = 0.5, -2.0
    for threshold in candidates:
        value = float(matthews_corrcoef(y_true, y_prob >= threshold))
        if value > best_mcc:
            best_threshold, best_mcc = float(threshold), value
    return best_threshold, best_mcc


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    predicted = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    ece, _ = expected_calibration_error(y_true, y_prob, n_bins=10)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "mcc": float(matthews_corrcoef(y_true, predicted)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "expected_calibration_error": float(ece),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def build(frame_path: Path, output_dir: Path, seed: int = 5) -> dict:
    frame = pd.read_csv(frame_path, low_memory=False)
    required = [*SCORE_COLUMNS, "label", "final_split"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Finalized frame is missing columns: {missing}")
    frame = frame.dropna(subset=required).copy()
    frame["label"] = frame["label"].astype(int)
    split = frame["final_split"].astype(str).str.lower()
    train = split.eq("train")
    valid = split.isin(["valid", "validation", "val"])
    test = split.eq("test")
    if not train.any() or not valid.any() or not test.any():
        raise ValueError("Explicit train, valid and test rows are required.")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", ExtraTreesClassifier(
            n_estimators=800,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )),
    ])
    model.fit(frame.loc[train, SCORE_COLUMNS], frame.loc[train, "label"])
    valid_raw = model.predict_proba(frame.loc[valid, SCORE_COLUMNS])[:, 1]
    calibrator = NumpyPlattCalibrator().fit(valid_raw, frame.loc[valid, "label"].to_numpy())
    valid_probability = calibrator.predict(valid_raw)
    threshold, valid_mcc = _optimize_mcc(frame.loc[valid, "label"].to_numpy(), valid_probability)
    test_raw = model.predict_proba(frame.loc[test, SCORE_COLUMNS])[:, 1]
    test_probability = calibrator.predict(test_raw)

    bundle = {
        "model": model,
        "calibrator": calibrator,
        "score_columns": SCORE_COLUMNS,
        "threshold": threshold,
        "background_medians": {
            column: float(frame.loc[train, column].median()) for column in SCORE_COLUMNS
        },
        "model_name": "PRING Stage 3 deployable fallback ensemble",
        "model_version": "stage3-fallback-v1",
        "variant": "stage3_fallback",
        "seed": seed,
    }
    manifest = {
        "status": "ready",
        "model_name": bundle["model_name"],
        "model_version": bundle["model_version"],
        "variant": bundle["variant"],
        "selection_basis": "MCC on the explicit validation partition",
        "calibration": "Platt scaling fitted on validation predictions",
        "score_columns": SCORE_COLUMNS,
        "threshold": threshold,
        "metrics": _metrics(frame.loc[test, "label"].to_numpy(), test_probability, threshold),
        "validation_metrics": _metrics(frame.loc[valid, "label"].to_numpy(), valid_probability, threshold),
        "validation_mcc_at_selected_threshold": valid_mcc,
        "training_rows": int(train.sum()),
        "validation_rows": int(valid.sum()),
        "test_rows": int(test.sum()),
        "note": (
            "Used only when R-GCN and HGT reproduce validated scores but the live Stage 1 "
            "feature source cannot reproduce the training-time Stage 1 score."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_dir / "production_stage3_fallback.joblib")
    (output_dir / "stage3_fallback_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-frame", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(build(args.training_frame, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
