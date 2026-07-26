from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts/models/production"
SCORE_FRAME = ROOT / "artifacts/results/production/finalized_training_frame.csv"
pytestmark = [
    pytest.mark.production_assets,
    pytest.mark.skipif(
        not (MODEL_DIR / "production_ensemble.joblib").is_file()
        or not SCORE_FRAME.is_file(),
        reason="local production model and result artifacts are not available",
    ),
]

os.environ.setdefault("PRODUCTION_MODEL_DIR", str(MODEL_DIR))
os.environ.setdefault("PRECOMPUTED_SCORE_FRAME", str(SCORE_FRAME))
os.environ.setdefault("PREDICTION_SCORE_MODE", "precomputed")

from pring_modeling.prediction_service import PRINGPredictionService
from utils.prediction_report import generate_prediction_report_html, prediction_summary_dataframe


def test_precomputed_active_and_inactive_predictions() -> None:
    frame = pd.read_csv(os.environ["PRECOMPUTED_SCORE_FRAME"])
    active = frame[frame["label"].eq(1)].iloc[0]
    inactive = frame[frame["label"].eq(0)].iloc[0]
    service = PRINGPredictionService()
    try:
        output = service.predict_many(
            [active["compound_key"], inactive["compound_key"]],
            [active["target_key"], inactive["target_key"]],
        )
    finally:
        service.close()
    assert output["successful_pairs"] >= 2
    assert output["model_status"]["validated_reference_store"]["available"] is True
    assert any(p["prediction"]["predicted_label"] == 1 for p in output["predictions"])
    assert any(p["prediction"]["predicted_label"] == 0 for p in output["predictions"])
    for prediction in output["predictions"]:
        assert 0.0 <= prediction["prediction"]["calibrated_probability"] <= 1.0
        assert prediction["explainability"]["local_component_contributions"]
        assert prediction["explainability"]["tree_shap"]["status"] in {
            "computed",
            "unavailable",
            "disabled",
        }


def test_unknown_pair_returns_explicit_error() -> None:
    service = PRINGPredictionService()
    try:
        output = service.predict_many(
            ["Compound|cid=999999999"],
            ["Protein|protein_id=P08684"],
        )
    finally:
        service.close()
    assert output["successful_pairs"] == 0
    assert "absent from both the validated reference frame" in output["errors"][0]["error"]
    assert "live inference is disabled" in output["errors"][0]["error"]


def test_report_exports() -> None:
    frame = pd.read_csv(os.environ["PRECOMPUTED_SCORE_FRAME"], nrows=1)
    row = frame.iloc[0]
    service = PRINGPredictionService()
    try:
        output = service.predict_many([row["compound_key"]], [row["target_key"]])
    finally:
        service.close()
    report = generate_prediction_report_html(output)
    summary = prediction_summary_dataframe(output)
    assert "PRING compound-target prediction report" in report
    assert "Local calibrated-probability explanation" in report
    assert not summary.empty
    assert "calibrated_probability" in summary.columns
