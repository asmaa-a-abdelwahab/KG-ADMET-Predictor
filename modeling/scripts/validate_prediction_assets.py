from __future__ import annotations

import json
import hashlib
import os
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn


def _runtime_check(manifest: dict) -> dict:
    expected = manifest.get("runtime_versions") or {}
    actual = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }
    mismatches = {}
    for key, value in expected.items():
        if key not in actual:
            continue
        expected_value = str(value)
        actual_value = str(actual[key])
        if key == "python":
            expected_value = ".".join(expected_value.split(".")[:2])
            actual_value = ".".join(actual_value.split(".")[:2])
        if expected_value != actual_value:
            mismatches[key] = {
                "expected": str(value),
                "actual": str(actual[key]),
            }
    return {
        "status": (
            "unverified_legacy_artifact"
            if not expected
            else ("compatible" if not mismatches else "incompatible")
        ),
        "expected": expected,
        "actual": actual,
        "mismatches": mismatches,
    }


def main() -> int:
    model_dir = Path(os.getenv("PRODUCTION_MODEL_DIR", "/models/production"))
    score_file = Path(
        os.getenv(
            "PRECOMPUTED_SCORE_FRAME",
            "/results/production/finalized_training_frame.csv",
        )
    )

    model_file = model_dir / "production_ensemble.joblib"
    manifest_file = model_dir / "manifest.json"
    allow_diagnostic = os.getenv("ALLOW_DIAGNOSTIC_PRODUCTION_ASSETS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    allow_runtime_mismatch = os.getenv("PREDICTION_ALLOW_RUNTIME_MISMATCH", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    allow_unverified_legacy = os.getenv("ALLOW_UNVERIFIED_LEGACY_ASSETS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }

    report = {
        "ready": False,
        "model_dir": str(model_dir),
        "score_file": str(score_file),
        "checks": {},
    }

    required = {
        "production_ensemble": model_file,
        "manifest": manifest_file,
        "precomputed_score_frame": score_file,
    }

    missing = []
    for name, path in required.items():
        exists = path.is_file()
        report["checks"][name] = {
            "path": str(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists else None,
        }
        if not exists:
            missing.append(str(path))

    if missing:
        report["error"] = "Missing required production assets"
        report["missing"] = missing
        print(json.dumps(report, indent=2))
        return 1

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        runtime_check = _runtime_check(manifest)
        report["checks"]["runtime_compatibility"] = {
            **runtime_check,
            "override_allowed": allow_runtime_mismatch,
            "legacy_override_allowed": allow_unverified_legacy,
        }

        expected_digest = manifest.get("model_artifact_sha256")
        actual_digest = hashlib.sha256(model_file.read_bytes()).hexdigest()
        report["checks"]["model_artifact_integrity"] = {
            "status": "verified" if expected_digest == actual_digest else (
                "unverified_legacy_artifact" if not expected_digest else "mismatch"
            ),
            "expected": expected_digest,
            "actual": actual_digest,
            "legacy_override_allowed": allow_unverified_legacy,
        }
        score_columns = list(manifest.get("score_columns") or [])
        if not score_columns:
            raise ValueError("The production manifest must define score_columns.")

        frame = pd.read_csv(score_file, low_memory=False)
        required_columns = ["compound_key", "target_key", *score_columns]
        absent = [column for column in required_columns if column not in frame.columns]
        if absent:
            raise ValueError(
                "Precomputed score frame is missing columns: " + ", ".join(absent)
            )

        component_split_columns = [
            column for column in frame.columns if column.startswith("split__")
        ]
        held_out_names = {"test", "holdout", "heldout", "held_out"}
        all_components_held_out = bool(component_split_columns) and all(
            set(frame[column].dropna().astype(str).str.lower().str.strip().unique()).issubset(held_out_names)
            and frame[column].notna().any()
            for column in component_split_columns
        )
        cache_rows = (
            int(frame["record_type"].astype(str).str.lower().eq("production_prediction_cache").sum())
            if "record_type" in frame.columns else 0
        )
        diagnostic = bool(
            all_components_held_out
            or manifest.get("publishable") is False
            or manifest.get("status") == "diagnostic_only"
        )
        required_provenance_ids = [
            "dataset_id",
            "split_registry_id",
            "feature_schema_id",
            "label_policy_id",
        ]
        missing_provenance_ids = [
            key for key in required_provenance_ids
            if not str(manifest.get(key) or "").strip()
        ]
        scientific_release_approved = bool(
            str(manifest.get("status") or "").strip().lower()
            in {"ready", "production_ready"}
            and manifest.get("publishable") is True
            and not missing_provenance_ids
            and not diagnostic
        )
        report["checks"]["scientific_provenance"] = {
            "diagnostic": diagnostic,
            "component_split_columns": component_split_columns,
            "all_component_rows_held_out": all_components_held_out,
            "manifest_publishable": manifest.get("publishable"),
            "manifest_status": manifest.get("status"),
            "required_provenance_ids": required_provenance_ids,
            "missing_provenance_ids": missing_provenance_ids,
            "release_approved": scientific_release_approved,
            "override_allowed": allow_diagnostic,
        }
        report["checks"]["cache_separation"] = {
            "production_cache_rows_in_reference": cache_rows,
            "passed": cache_rows == 0,
        }

        blockers = []
        if runtime_check["status"] == "incompatible" and not allow_runtime_mismatch:
            blockers.append(
                "serialized model runtime is incompatible with this validation environment: "
                f"{runtime_check['mismatches']}"
            )
        if runtime_check["status"] == "unverified_legacy_artifact" and not allow_unverified_legacy:
            blockers.append("manifest does not record the serialized model runtime")
        if expected_digest and expected_digest != actual_digest:
            blockers.append("production model digest does not match the manifest")
        if not expected_digest and not allow_unverified_legacy:
            blockers.append("manifest does not record the production model digest")
        if cache_rows:
            blockers.append("production cache rows contaminate the immutable reference frame")
        if diagnostic and not allow_diagnostic:
            blockers.append(
                "component predictions were already held out before the final "
                "train/validation/test re-split"
            )
        if not scientific_release_approved and not allow_diagnostic:
            blockers.append(
                "manifest is not an approved scientific release "
                f"(status={manifest.get('status')!r}, publishable={manifest.get('publishable')!r}, "
                f"missing provenance IDs={missing_provenance_ids})"
            )
        expected_frame_digest = str(manifest.get("training_frame_sha256") or "").strip()
        actual_frame_digest = hashlib.sha256(score_file.read_bytes()).hexdigest()
        report["checks"]["reference_frame_integrity"] = {
            "expected": expected_frame_digest or None,
            "actual": actual_frame_digest,
            "verified": bool(expected_frame_digest and expected_frame_digest == actual_frame_digest),
        }
        if not expected_frame_digest:
            blockers.append("manifest does not record the immutable reference-frame digest")
        elif expected_frame_digest != actual_frame_digest:
            blockers.append("immutable reference-frame digest does not match the manifest")
        duplicate_pairs = int(frame.duplicated(["compound_key", "target_key"], keep=False).sum())
        report["checks"]["pair_uniqueness"] = {
            "duplicate_rows": duplicate_pairs,
            "passed": duplicate_pairs == 0,
        }
        if duplicate_pairs:
            blockers.append(f"reference frame contains {duplicate_pairs} duplicate pair rows")
        if blockers:
            raise ValueError("Production asset validation failed: " + "; ".join(blockers) + ".")

        modeling_root = str(Path(__file__).resolve().parents[1])
        if modeling_root not in sys.path:
            sys.path.insert(0, modeling_root)
        bundle = joblib.load(model_file)
        bundle_score_columns = list(bundle.get("score_columns") or [])
        required_bundle_keys = {"model", "calibrator", "threshold", "score_columns"}
        missing_bundle_keys = sorted(required_bundle_keys.difference(bundle))
        report["checks"]["bundle_manifest_consistency"] = {
            "missing_bundle_keys": missing_bundle_keys,
            "score_columns_match": bundle_score_columns == score_columns,
        }
        if missing_bundle_keys:
            raise ValueError(
                "Production bundle is missing keys: " + ", ".join(missing_bundle_keys)
            )
        if bundle_score_columns != score_columns:
            raise ValueError("Bundle score_columns do not match the manifest.")

        report["ready"] = True
        report["model_name"] = manifest.get("model_name")
        report["score_columns"] = score_columns
        report["rows_read"] = len(frame)
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        print(json.dumps(report, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
