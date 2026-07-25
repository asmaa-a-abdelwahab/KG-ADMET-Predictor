from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

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

from .final_validation import NumpyPlattCalibrator, expected_calibration_error

DEPLOYABLE_SCORE_COLUMNS = [
    "score__stage1_tabular_extra_trees",
    "score__stage3_rgcn_sampled",
    "score__stage3_hgt_sampled",
]


def _safe_float(value: Any) -> float | None:
    try:
        v = float(value)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ece, _ = expected_calibration_error(y_true, y_prob, n_bins=10)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "negative_precision": float(tn / (tn + fn)) if (tn + fn) else None,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "expected_calibration_error": float(ece),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _optimize_mcc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    candidates = np.unique(np.concatenate(([0.0], y_prob, [1.0])))
    best_threshold = 0.5
    best_mcc = -2.0
    for threshold in candidates:
        score = matthews_corrcoef(y_true, y_prob >= threshold)
        if score > best_mcc:
            best_mcc = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_mcc


def build_production_bundle(
    training_frame: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 5,
    score_columns: list[str] | None = None,
    source_metrics: str | Path | None = None,
    stage1_feature_importance: str | Path | None = None,
    per_target_metrics: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deployable ensemble using only component scores reproducible at inference.

    The research Finalized V2 ensemble includes Stage 1 holdout/CV score columns.
    Those columns are evaluation products rather than separately serialized models,
    so they cannot be generated for a new compound-target pair. This builder creates
    the production counterpart from three reproducible scorers: Stage 1 Extra Trees,
    Stage 3 R-GCN and Stage 3 HGT.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = Path(training_frame)
    frame = pd.read_csv(frame_path)
    score_columns = score_columns or list(DEPLOYABLE_SCORE_COLUMNS)

    required = [*score_columns, "label", "final_split"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise KeyError(f"Training frame is missing required columns: {missing}")

    frame = frame.dropna(subset=required).copy()
    frame["label"] = frame["label"].astype(int)
    split = frame["final_split"].astype(str).str.lower()
    train = split.eq("train")
    valid = split.isin(["valid", "validation", "val"])
    test = split.eq("test")
    if not train.any() or not valid.any() or not test.any():
        raise ValueError("Production bundle requires explicit train, valid and test rows.")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            ExtraTreesClassifier(
                n_estimators=800,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
    ])
    model.fit(frame.loc[train, score_columns], frame.loc[train, "label"])

    valid_raw = model.predict_proba(frame.loc[valid, score_columns])[:, 1]
    calibrator = NumpyPlattCalibrator().fit(valid_raw, frame.loc[valid, "label"].to_numpy())
    valid_prob = calibrator.predict(valid_raw)
    threshold, valid_mcc = _optimize_mcc(frame.loc[valid, "label"].to_numpy(), valid_prob)

    test_raw = model.predict_proba(frame.loc[test, score_columns])[:, 1]
    test_prob = calibrator.predict(test_raw)
    metrics = _metrics(frame.loc[test, "label"].to_numpy(), test_prob, threshold)
    valid_metrics = _metrics(frame.loc[valid, "label"].to_numpy(), valid_prob, threshold)

    classifier = model.named_steps["classifier"]
    importances = getattr(classifier, "feature_importances_", np.zeros(len(score_columns)))
    feature_importance = pd.DataFrame({
        "component_score": score_columns,
        "importance": np.asarray(importances, dtype=float),
    }).sort_values("importance", ascending=False)
    feature_importance.to_csv(output_dir / "component_feature_importance.csv", index=False)

    background = frame.loc[train, score_columns].sample(
        n=min(500, int(train.sum())), random_state=seed
    )
    background.to_csv(output_dir / "explainability_background.csv", index=False)
    medians = {c: float(frame.loc[train, c].median()) for c in score_columns}

    source_summary: dict[str, Any] = {}
    if source_metrics and Path(source_metrics).exists():
        source_summary = json.loads(Path(source_metrics).read_text(encoding="utf-8"))

    bundle = {
        "model": model,
        "calibrator": calibrator,
        "score_columns": score_columns,
        "threshold": float(threshold),
        "background_medians": medians,
        "model_name": "PRING deployable finalized ensemble",
        "model_version": "production-v1",
        "seed": int(seed),
    }
    model_file = output_dir / "production_ensemble.joblib"
    joblib.dump(bundle, model_file)

    copied_files: dict[str, str | None] = {
        "stage1_feature_importance": None,
        "per_target_metrics": None,
    }
    if stage1_feature_importance and Path(stage1_feature_importance).exists():
        target = output_dir / "stage1_feature_importance.csv"
        target.write_bytes(Path(stage1_feature_importance).read_bytes())
        copied_files["stage1_feature_importance"] = target.name
    if per_target_metrics and Path(per_target_metrics).exists():
        target = output_dir / "per_target_metrics.csv"
        target.write_bytes(Path(per_target_metrics).read_bytes())
        copied_files["per_target_metrics"] = target.name

    manifest = {
        "status": "ready",
        "model_name": bundle["model_name"],
        "model_version": bundle["model_version"],
        "selection_basis": "MCC on the explicit validation partition",
        "research_best_model": {
            "name": source_summary.get("model", "finalized_ensemble_extra_trees"),
            "metrics": source_summary.get("metrics", {}),
            "note": (
                "The research ensemble includes Stage 1 holdout/CV score columns that are not "
                "reproducible for a new pair. The production ensemble uses only deployable components."
            ),
        },
        "production_components": [
            {"score_column": score_columns[0], "display_name": "Stage 1 Extra Trees", "artifact_role": "structural baseline"},
            {"score_column": score_columns[1], "display_name": "Stage 3 R-GCN", "artifact_role": "primary heterogeneous graph scorer"},
            {"score_column": score_columns[2], "display_name": "Stage 3 HGT", "artifact_role": "complementary attention-based scorer"},
        ],
        "score_columns": score_columns,
        "threshold": float(threshold),
        "calibration": "Platt scaling fitted on validation predictions",
        "metrics": metrics,
        "validation_metrics": valid_metrics,
        "validation_mcc_at_selected_threshold": float(valid_mcc),
        "training_rows": int(train.sum()),
        "validation_rows": int(valid.sum()),
        "test_rows": int(test.sum()),
        "model_file": model_file.name,
        "background_file": "explainability_background.csv",
        "component_feature_importance_file": "component_feature_importance.csv",
        **copied_files,
        "required_live_artifacts": {
            "stage1": [
                "stage1_tabular_extra_trees.joblib",
                "feature_columns.json",
                "Neo4j node embedding property used during training (default: pringFastRP)",
            ],
            "stage3_rgcn": ["best_model.pt", "rgcn_sampled_metadata.json"],
            "stage3_hgt": ["best_model.pt", "hgt_sampled_metadata.json"],
            "stage3_prepared_data": ["HeteroData export", "node_mapping.csv"],
        },
        "fallback_mode": (
            "When live component checkpoints are not mounted, the predictor can return exact precomputed "
            "scores for pairs present in the finalized training/prediction frame."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a deployable PRING ensemble from reproducible component scores.")
    p.add_argument("--training-frame", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--source-metrics", default=None)
    p.add_argument("--stage1-feature-importance", default=None)
    p.add_argument("--per-target-metrics", default=None)
    p.add_argument("--score-columns", nargs="*", default=None)
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = build_production_bundle(
        args.training_frame,
        args.output_dir,
        seed=args.seed,
        score_columns=args.score_columns,
        source_metrics=args.source_metrics,
        stage1_feature_importance=args.stage1_feature_importance,
        per_target_metrics=args.per_target_metrics,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
