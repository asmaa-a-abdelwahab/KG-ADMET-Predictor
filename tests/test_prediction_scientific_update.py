from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_reference_and_cache_are_separate(tmp_path: Path):
    from pring_modeling.prediction_store import PredictionCacheStore, ReferenceScoreStore

    score_columns = ["s1", "s2", "s3"]
    reference_path = tmp_path / "finalized_training_frame.csv"
    pd.DataFrame(
        [
            {"compound_key": "C1", "target_key": "T1", "s1": 0.1, "s2": 0.2, "s3": 0.3, "final_split": "test"},
            {"compound_key": "C2", "target_key": "T2", "s1": 0.4, "s2": 0.5, "s3": 0.6, "final_split": "production_inference", "record_type": "production_prediction_cache"},
        ]
    ).to_csv(reference_path, index=False)

    reference = ReferenceScoreStore(reference_path, score_columns)
    assert reference.row_count == 1
    assert reference.get("C1", "T1") is not None
    assert reference.get("C2", "T2") is None

    cache_path = tmp_path / "production_prediction_cache.csv"
    cache = PredictionCacheStore(cache_path, score_columns)
    result = cache.upsert_rows([
        {"compound_key": "C2", "target_key": "T2", "s1": 0.4, "s2": 0.5, "s3": 0.6}
    ])
    assert result["added_rows"] == 1
    assert cache.get("C2", "T2") is not None
    assert len(pd.read_csv(reference_path)) == 2  # source file was not changed


def test_report_contains_scientific_sections():
    from utils.prediction_report import generate_prediction_report_html

    payload = {
        "requested_pairs": 1,
        "successful_pairs": 1,
        "report_context": {
            "model_provenance": {"model_name": "PRING", "model_version": "v1", "score_columns": []},
            "global_validation_metrics": {"mcc": 0.9},
            "result_status_counts": {"novel_predicted_interaction": 1},
            "score_source_counts": {"live_component_inference": 1},
            "live_inference_parity": {"status": "passed", "sample_size": 5, "decision_agreement": 1.0},
        },
        "predictions": [{
            "pair": {"compound_name": "Compound A", "target_name": "CYP3A4", "compound_key": "C1", "target_key": "T1"},
            "prediction": {"calibrated_probability": 0.8, "threshold": 0.2, "result_status": "novel_predicted_interaction"},
            "model": {"score_source": "live_component_inference", "graph_version": "g1"},
            "component_score_details": [],
            "explainability": {
                "uncertainty_metrics": {"model_certainty": "high"},
                "applicability_domain": {"status": "in_domain", "components": []},
                "local_component_explanation": {"components": []},
                "tree_shap": {"status": "disabled"},
            },
            "evidence": {"tier": "Tier 3", "evidence_support": "low", "known_direct_interaction": False},
            "interpretation": {"statement": "Prediction statement", "recommended_action": "Validate", "scope_warning": "Boundary"},
        }],
        "errors": [],
    }
    report = generate_prediction_report_html(payload)
    assert "Executive summary" in report
    assert "Model and data provenance" in report
    assert "Frozen-test validation" in report
    assert "Scientific limitations" in report
    assert "Novel predicted interaction" in report
