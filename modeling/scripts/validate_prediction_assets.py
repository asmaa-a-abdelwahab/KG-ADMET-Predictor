from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd


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
        bundle = joblib.load(model_file)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        score_columns = list(
            bundle.get("score_columns")
            or manifest.get("score_columns")
            or []
        )
        if not score_columns:
            raise ValueError("No score_columns were defined by the bundle or manifest.")

        frame = pd.read_csv(score_file, nrows=10, low_memory=False)
        required_columns = ["compound_key", "target_key", *score_columns]
        absent = [column for column in required_columns if column not in frame.columns]
        if absent:
            raise ValueError(
                "Precomputed score frame is missing columns: " + ", ".join(absent)
            )

        report["ready"] = True
        report["model_name"] = manifest.get("model_name")
        report["score_columns"] = score_columns
        report["sample_rows_read"] = len(frame)
        print(json.dumps(report, indent=2))
        return 0
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        print(json.dumps(report, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
