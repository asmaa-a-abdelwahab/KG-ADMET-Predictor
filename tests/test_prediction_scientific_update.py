from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest


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
        {
            "compound_key": "C2",
            "target_key": "T2",
            "s1": 0.4,
            "s2": 0.5,
            "s3": 0.6,
            "model_variant": "primary",
            "model_version": "v1",
            "graph_version": "g1",
        }
    ])
    assert result["added_rows"] == 1
    assert cache.get("C2", "T2", model_versions={"primary": "v1"}, graph_version="g1") is not None
    assert cache.get("C2", "T2", model_versions={"primary": "v2"}, graph_version="g1") is None
    assert cache.get("C2", "T2", model_versions={"primary": "v1"}, graph_version="g2") is None
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


def test_production_bundle_rejects_resplit_heldout_component_scores(tmp_path: Path):
    from pring_modeling.production_bundle import (
        DEPLOYABLE_SCORE_COLUMNS,
        build_production_bundle,
    )

    frame = pd.DataFrame(
        [
            {
                **{column: 0.1 + (row_index * 0.1) for column in DEPLOYABLE_SCORE_COLUMNS},
                "label": row_index % 2,
                "final_split": ("train", "valid", "test")[row_index % 3],
                **{f"split__{column}": "test" for column in DEPLOYABLE_SCORE_COLUMNS},
            }
            for row_index in range(12)
        ]
    )
    frame_path = tmp_path / "diagnostic_frame.csv"
    frame.to_csv(frame_path, index=False)

    with pytest.raises(ValueError, match="already-held-out"):
        build_production_bundle(frame_path, tmp_path / "production")


def test_production_bundle_rejects_compound_overlap_across_partitions(tmp_path: Path):
    from pring_modeling.production_bundle import (
        DEPLOYABLE_SCORE_COLUMNS,
        build_production_bundle,
    )

    rows = []
    for row_index in range(12):
        final_split = ("train", "valid", "test")[row_index % 3]
        rows.append(
            {
                **{column: 0.05 + (row_index * 0.05) for column in DEPLOYABLE_SCORE_COLUMNS},
                "compound_key": f"C{row_index // 3}",
                "target_key": f"T{row_index}",
                "label": row_index % 2,
                "final_split": final_split,
                **{
                    f"split__{column}": final_split
                    for column in DEPLOYABLE_SCORE_COLUMNS
                },
            }
        )
    frame_path = tmp_path / "compound_overlap.csv"
    pd.DataFrame(rows).to_csv(frame_path, index=False)

    with pytest.raises(ValueError, match="Compound groups cross final partitions"):
        build_production_bundle(frame_path, tmp_path / "production")


def test_model_digest_verification_rejects_tampering(tmp_path: Path):
    from pring_modeling.prediction_service import _verify_artifact_digest

    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"original")
    manifest = {"model_artifact_sha256": hashlib.sha256(b"original").hexdigest()}
    assert _verify_artifact_digest(model_file, manifest)["status"] == "verified"

    model_file.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _verify_artifact_digest(model_file, manifest)


def test_prediction_api_key_is_optional_in_development_and_enforced_when_set(monkeypatch):
    from fastapi import HTTPException
    from pring_modeling.prediction_api import require_api_key

    monkeypatch.delenv("PREDICTION_API_KEY", raising=False)
    require_api_key(None)
    monkeypatch.setenv("PREDICTION_API_KEY", "secret")
    require_api_key("secret")
    with pytest.raises(HTTPException) as error:
        require_api_key("wrong")
    assert error.value.status_code == 401


def test_stage1_identifier_filter_does_not_drop_scientific_acidic_descriptor():
    from pring_modeling.stage1_tabular import _is_identifier_like

    assert _is_identifier_like("molgraph_cid")
    assert _is_identifier_like("missing_molgraph_cid")
    assert _is_identifier_like("compound_node_id")
    assert not _is_identifier_like("acidic_group_count")


def test_final_validation_rejects_conflicting_component_split_registry():
    from pring_modeling.final_validation import _infer_supplied_split

    frame = pd.DataFrame(
        {
            "split__model_a": ["train", "test"],
            "split__model_b": ["test", "test"],
        }
    )
    split, audit = _infer_supplied_split(frame)
    assert split is not None
    assert split.iloc[0] == "unknown"
    assert split.iloc[1] == "test"
    assert audit["conflicting_rows"] == 1
