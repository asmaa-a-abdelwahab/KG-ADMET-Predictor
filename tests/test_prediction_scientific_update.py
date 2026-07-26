from __future__ import annotations

import hashlib
import json
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
    assert "Heuristic confidence" in report


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


def test_fixed_mean_production_bundle_preserves_validated_score_contract(tmp_path: Path):
    import joblib
    import numpy as np

    from pring_modeling.production_bundle import (
        DEPLOYABLE_SCORE_COLUMNS,
        build_production_bundle,
    )

    rows = []
    for row_index in range(18):
        partition = ("train", "valid", "test")[row_index // 6]
        label = row_index % 2
        rows.append(
            {
                "compound_key": f"C{row_index}",
                "target_key": f"T{row_index}",
                "label": label,
                "final_split": partition,
                **{
                    column: (0.75 if label else 0.25) + component_index * 0.02
                    for component_index, column in enumerate(DEPLOYABLE_SCORE_COLUMNS)
                },
                **{
                    f"split__{column}": (
                        "train_oof" if partition == "train" else partition
                    )
                    for column in DEPLOYABLE_SCORE_COLUMNS
                },
            }
        )
    frame_path = tmp_path / "registered_scores.csv"
    pd.DataFrame(rows).to_csv(frame_path, index=False)
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "dataset_id": "dataset-1",
                "split_registry_id": "split-1",
                "label_policy_id": "labels-1",
            }
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "production"
    manifest = build_production_bundle(
        frame_path,
        output_dir,
        source_metrics=provenance_path,
    )
    assert manifest["status"] == "ready"
    assert manifest["publishable"] is True
    assert manifest["combiner_protocol"] == "fixed_equal_weight_combiner_no_meta_training"
    assert manifest["meta_training_rows"] == 0
    assert manifest["score_columns"] == DEPLOYABLE_SCORE_COLUMNS

    bundle = joblib.load(output_dir / "production_ensemble.joblib")
    component_scores = np.array([[0.2, 0.5, 0.8]])
    assert bundle["score_columns"] == DEPLOYABLE_SCORE_COLUMNS
    assert bundle["model"].predict_proba(component_scores)[0, 1] == pytest.approx(0.5)


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
    from pring_modeling.final_validation import _infer_supplied_split, build_parser

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
    help_text = build_parser().format_help()
    assert "Compound-group bootstrap resamples for 95%" in help_text


def test_prediction_release_gate_rejects_diagnostic_and_incomplete_manifests():
    from pring_modeling.prediction_service import (
        _require_scientific_release,
        _scientific_release_status,
    )

    diagnostic = {
        "status": "diagnostic_only",
        "publishable": False,
    }
    with pytest.raises(RuntimeError, match="not approved for production serving"):
        _require_scientific_release(diagnostic)

    incomplete = {"status": "ready", "publishable": True}
    assert _scientific_release_status(incomplete)["ready"] is False

    ready = {
        "status": "ready",
        "publishable": True,
        "dataset_id": "dataset",
        "split_registry_id": "split",
        "feature_schema_id": "features",
        "label_policy_id": "labels",
    }
    assert _require_scientific_release(ready)["ready"] is True


def test_shared_split_prefers_registered_groups_and_fails_closed():
    from pring_modeling.shared_splits import _make_split, _scaffold_or_group_key

    frame = pd.DataFrame(
        {
            "compound_key": [f"C{i}" for i in range(6)],
            "split_group": ["G1", "G1", "G2", "G2", "G3", "G3"],
        }
    )
    compound = frame["compound_key"]
    groups = _scaffold_or_group_key(frame, compound, "registered")
    assert groups.tolist() == frame["split_group"].tolist()

    split = _make_split(
        pd.Series([0, 1, 0, 1, 0, 1]),
        groups,
        seed=11,
        test_size=0.2,
        valid_size=0.2,
    )
    assert set(split) == {"train", "valid", "test"}
    assert (
        pd.DataFrame({"group": groups, "split": split})
        .groupby("group")["split"]
        .nunique()
        .max()
        == 1
    )

    with pytest.raises(ValueError, match="row-level fallback is prohibited"):
        _make_split(
            pd.Series([0, 1, 0, 1]),
            pd.Series(["one", "one", "two", "two"]),
            seed=11,
            test_size=0.2,
            valid_size=0.2,
        )
