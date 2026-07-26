from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

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


def build(
    frame_path: Path,
    output_dir: Path,
    seed: int = 5,
    provenance_manifest: Path | None = None,
) -> dict:
    frame = pd.read_csv(frame_path, low_memory=False)
    required = [
        *SCORE_COLUMNS,
        "label",
        "final_split",
        "compound_key",
        "target_key",
    ]
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
    component_split_columns = [f"split__{column}" for column in SCORE_COLUMNS]
    missing_component_splits = [
        column for column in component_split_columns if column not in frame.columns
    ]
    if missing_component_splits:
        raise ValueError(
            "Stage 3 fallback requires registered component splits: "
            + ", ".join(missing_component_splits)
        )
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
    normalized_final = split.replace(split_aliases)
    mismatches = {}
    non_oof_train_rows = {}
    for column in component_split_columns:
        normalized_component = (
            frame[column].astype(str).str.lower().str.strip().replace(split_aliases)
        )
        mismatch_count = int(normalized_component.ne(normalized_final).sum())
        if mismatch_count:
            mismatches[column] = mismatch_count
        raw_train = frame.loc[train, column].astype(str).str.lower().str.strip()
        non_oof_count = int(
            (
                ~raw_train.str.contains(
                    r"oof|out[_ -]?of[_ -]?fold",
                    regex=True,
                )
            ).sum()
        )
        if non_oof_count:
            non_oof_train_rows[column] = non_oof_count
    if mismatches:
        raise ValueError(
            "Stage 3 component scores do not share the registered final split: "
            f"{mismatches}"
        )
    if non_oof_train_rows:
        raise ValueError(
            "Stage 3 fallback meta-training requires out-of-fold component scores: "
            f"{non_oof_train_rows}"
        )
    duplicate_rows = int(
        frame.duplicated(["compound_key", "target_key"], keep=False).sum()
    )
    if duplicate_rows:
        raise ValueError(
            f"Stage 3 fallback requires unique compound-target pairs; found {duplicate_rows} duplicate rows."
        )
    compound_sets = {
        name: set(frame.loc[mask, "compound_key"].astype(str))
        for name, mask in {"train": train, "valid": valid, "test": test}.items()
    }
    overlap = {
        "train_valid": len(compound_sets["train"] & compound_sets["valid"]),
        "train_test": len(compound_sets["train"] & compound_sets["test"]),
        "valid_test": len(compound_sets["valid"] & compound_sets["test"]),
    }
    if any(overlap.values()):
        raise ValueError(
            "Compound groups cross Stage 3 fallback partitions: "
            f"{overlap}"
        )

    source_provenance = {}
    if provenance_manifest:
        source_provenance = json.loads(
            Path(provenance_manifest).read_text(encoding="utf-8")
        )
    required_provenance = ("dataset_id", "split_registry_id", "label_policy_id")
    missing_provenance = [
        key for key in required_provenance
        if not str(source_provenance.get(key) or "").strip()
    ]
    frame_digest = hashlib.sha256(Path(frame_path).read_bytes()).hexdigest()
    score_schema_id = hashlib.sha256(
        json.dumps(SCORE_COLUMNS, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    split_registry_id = hashlib.sha256(
        frame[["compound_key", "target_key", "final_split"]]
        .sort_values(["compound_key", "target_key", "final_split"])
        .to_json(orient="records")
        .encode("utf-8")
    ).hexdigest()
    runtime_versions = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }

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
        "model_version": "stage3-fallback-" + hashlib.sha256(
            f"{frame_digest}|{split_registry_id}|{score_schema_id}|{seed}".encode("utf-8")
        ).hexdigest()[:12],
        "variant": "stage3_fallback",
        "training_frame_sha256": frame_digest,
        "dataset_id": source_provenance.get("dataset_id", frame_digest),
        "split_registry_id": source_provenance.get(
            "split_registry_id", split_registry_id
        ),
        "feature_schema_id": score_schema_id,
        "label_policy_id": source_provenance.get(
            "label_policy_id",
            "aggregated_pring_interaction_label-unverified-source",
        ),
        "runtime_versions": runtime_versions,
        "seed": seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "production_stage3_fallback.joblib"
    joblib.dump(bundle, model_path)
    artifact_digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    publishable = not missing_provenance
    manifest = {
        "status": "ready" if publishable else "diagnostic_only",
        "publishable": publishable,
        "model_name": bundle["model_name"],
        "model_version": bundle["model_version"],
        "variant": bundle["variant"],
        "training_frame_sha256": frame_digest,
        "dataset_id": bundle["dataset_id"],
        "split_registry_id": bundle["split_registry_id"],
        "feature_schema_id": bundle["feature_schema_id"],
        "label_policy_id": bundle["label_policy_id"],
        "runtime_versions": runtime_versions,
        "model_artifact_sha256": artifact_digest,
        "source_provenance_verified": not missing_provenance,
        "scientific_release_blockers": (
            ["missing source provenance: " + ", ".join(missing_provenance)]
            if missing_provenance else []
        ),
        "split_provenance": {
            "component_split_columns": component_split_columns,
            "component_split_mismatches": mismatches,
            "component_train_non_oof": non_oof_train_rows,
            "pair_duplicates": duplicate_rows,
            "compound_partition_overlap": overlap,
        },
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
    (output_dir / "stage3_fallback_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-frame", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument(
        "--provenance-manifest",
        type=Path,
        help="Validated upstream manifest containing dataset, split, and label-policy IDs.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.training_frame,
                args.output_dir,
                args.seed,
                args.provenance_manifest,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
