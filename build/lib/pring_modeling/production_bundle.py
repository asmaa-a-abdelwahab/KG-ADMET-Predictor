from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    allow_diagnostic_split: bool = False,
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
    training_frame_sha256 = _sha256_file(frame_path)
    score_columns = score_columns or list(DEPLOYABLE_SCORE_COLUMNS)

    component_split_columns = [c for c in frame.columns if c.startswith("split__")]
    held_out_names = {"test", "holdout", "heldout", "held_out"}
    all_components_held_out = bool(component_split_columns) and all(
        set(frame[c].dropna().astype(str).str.lower().str.strip().unique()).issubset(held_out_names)
        and frame[c].notna().any()
        for c in component_split_columns
    )
    diagnostic_marker = (
        "split_is_diagnostic" in frame.columns
        and frame["split_is_diagnostic"].astype(str).str.lower().isin({"true", "1", "yes"}).any()
    )
    diagnostic_origin = (
        "split_origin" in frame.columns
        and frame["split_origin"].astype(str).str.contains(
            "generated_from_already_scored_predictions", case=False, na=False
        ).any()
    )
    diagnostic_split = bool(all_components_held_out or diagnostic_marker or diagnostic_origin)
    if diagnostic_split and not allow_diagnostic_split:
        raise ValueError(
            "Refusing to build a production bundle from a diagnostic re-split of "
            "already-held-out component predictions. Generate registered train/validation/test "
            "or nested out-of-fold component scores, or use --allow-diagnostic-split only for "
            "non-production legacy reproduction."
        )

    required = [*score_columns, "label", "final_split", "compound_key", "target_key"]
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

    normalized_final_split = split.replace({"validation": "valid", "val": "valid"})
    if not normalized_final_split.isin({"train", "valid", "test"}).all():
        invalid = sorted(normalized_final_split[~normalized_final_split.isin(
            {"train", "valid", "test"}
        )].unique())
        raise ValueError(f"Unsupported final_split values: {invalid}")

    duplicate_pairs = frame.duplicated(["compound_key", "target_key"], keep=False)
    if duplicate_pairs.any():
        duplicate_count = int(duplicate_pairs.sum())
        raise ValueError(
            f"Production bundle requires one row per compound-target pair; "
            f"found {duplicate_count} duplicate rows."
        )

    compound_sets = {
        name: set(frame.loc[mask, "compound_key"].astype(str))
        for name, mask in {"train": train, "valid": valid, "test": test}.items()
    }
    compound_overlap = {
        "train_valid": len(compound_sets["train"] & compound_sets["valid"]),
        "train_test": len(compound_sets["train"] & compound_sets["test"]),
        "valid_test": len(compound_sets["valid"] & compound_sets["test"]),
    }
    if any(compound_overlap.values()):
        raise ValueError(
            "Compound groups cross final partitions; use a compound/similarity-group-aware "
            f"split registry. Overlap counts: {compound_overlap}"
        )

    component_split_mismatches: dict[str, int] = {}
    split_aliases = {
        "validation": "valid",
        "val": "valid",
        "training": "train",
        "oof": "train",
        "train_oof": "train",
        "holdout": "test",
        "heldout": "test",
        "held_out": "test",
    }
    for column in component_split_columns:
        if frame[column].isna().any():
            component_split_mismatches[column] = int(frame[column].isna().sum())
            continue
        component_split = (
            frame[column].astype(str).str.lower().str.strip().replace(split_aliases)
        )
        component_split_mismatches[column] = int(
            component_split.ne(normalized_final_split).sum()
        )
    component_split_mismatches = {
        column: count for column, count in component_split_mismatches.items() if count
    }
    if component_split_mismatches and not (diagnostic_split and allow_diagnostic_split):
        raise ValueError(
            "Component predictions do not share the registered final split: "
            f"{component_split_mismatches}"
        )

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

    split_identity_columns = [
        column for column in ("pair_key", "compound_key", "target_key", "final_split")
        if column in frame.columns
    ]
    split_registry_id = hashlib.sha256(
        frame[split_identity_columns]
        .sort_values(split_identity_columns)
        .to_json(orient="records")
        .encode("utf-8")
    ).hexdigest()
    score_schema_id = hashlib.sha256(
        json.dumps(score_columns, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    model_version = "production-" + hashlib.sha256(
        f"{training_frame_sha256}|{split_registry_id}|{score_schema_id}|{seed}".encode("utf-8")
    ).hexdigest()[:12]
    runtime_versions = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }

    bundle = {
        "model": model,
        "calibrator": calibrator,
        "score_columns": score_columns,
        "threshold": float(threshold),
        "background_medians": medians,
        "model_name": "PRING deployable finalized ensemble",
        "model_version": model_version,
        "training_frame_sha256": training_frame_sha256,
        "dataset_id": source_summary.get("dataset_id", training_frame_sha256),
        "split_registry_id": source_summary.get("split_registry_id", split_registry_id),
        "feature_schema_id": score_schema_id,
        "label_policy_id": source_summary.get("label_policy_id", "aggregated_pring_interaction_label"),
        "runtime_versions": runtime_versions,
        "seed": int(seed),
    }
    model_file = output_dir / "production_ensemble.joblib"
    joblib.dump(bundle, model_file)
    model_artifact_sha256 = hashlib.sha256(model_file.read_bytes()).hexdigest()

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
        "status": "diagnostic_only" if diagnostic_split else "ready",
        "publishable": not diagnostic_split,
        "model_name": bundle["model_name"],
        "model_version": bundle["model_version"],
        "training_frame": str(frame_path),
        "training_frame_sha256": training_frame_sha256,
        "dataset_id": bundle["dataset_id"],
        "split_registry_id": bundle["split_registry_id"],
        "feature_schema_id": bundle["feature_schema_id"],
        "label_policy_id": bundle["label_policy_id"],
        "runtime_versions": runtime_versions,
        "model_artifact_sha256": model_artifact_sha256,
        "selection_basis": (
            "MCC on a diagnostic validation re-split; not production-valid"
            if diagnostic_split
            else "MCC on the registered validation partition"
        ),
        "split_provenance": {
            "diagnostic_split": diagnostic_split,
            "component_split_columns": component_split_columns,
            "all_component_rows_held_out": all_components_held_out,
            "override_used": bool(allow_diagnostic_split),
            "pair_duplicates": 0,
            "compound_partition_overlap": compound_overlap,
            "component_split_mismatches": component_split_mismatches,
            "partition_counts": {
                "train": int(train.sum()),
                "valid": int(valid.sum()),
                "test": int(test.sum()),
            },
        },
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
    p.add_argument(
        "--allow-diagnostic-split",
        action="store_true",
        help="Permit a non-publishable legacy bundle from re-split held-out predictions.",
    )
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
        allow_diagnostic_split=args.allow_diagnostic_split,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
