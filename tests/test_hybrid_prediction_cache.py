from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class IdentityCalibrator:
    def predict(self, values):
        return np.asarray(values, dtype=float)


class FakeLiveScorer:
    def __init__(self):
        self.calls = 0

    def status(self):
        return {"ready": True, "components": {}, "shared_graph": {"loaded": False}}

    def score_many(self, pairs, _provider):
        self.calls += 1
        output = {}
        for pair in pairs:
            output[(pair.compound_key, pair.target_key)] = {
                "scores": {
                    "score__stage1_tabular_extra_trees": 0.8,
                    "score__stage3_rgcn_sampled": 0.7,
                    "score__stage3_hgt_sampled": 0.9,
                },
                "details": {
                    "score_source": "live_component_inference",
                    "stage1_pair_features": {"gds_pringfastrp_dot": 1.0},
                },
            }
        return output


def _make_bundle(model_dir: Path) -> None:
    columns = [
        "score__stage1_tabular_extra_trees",
        "score__stage3_rgcn_sampled",
        "score__stage3_hgt_sampled",
    ]
    x = pd.DataFrame(
        [
            [0.1, 0.2, 0.1],
            [0.2, 0.1, 0.2],
            [0.8, 0.7, 0.9],
            [0.9, 0.8, 0.7],
        ],
        columns=columns,
    )
    y = np.asarray([0, 0, 1, 1])
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", ExtraTreesClassifier(n_estimators=20, random_state=3)),
        ]
    )
    model.fit(x, y)
    bundle = {
        "model": model,
        "calibrator": IdentityCalibrator(),
        "score_columns": columns,
        "threshold": 0.5,
        "background_medians": {column: 0.5 for column in columns},
        "model_name": "test ensemble",
        "model_version": "test-v1",
    }
    model_dir.mkdir(parents=True)
    joblib.dump(bundle, model_dir / "production_ensemble.joblib")
    (model_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_name": "test ensemble",
                "model_version": "test-v1",
                "score_columns": columns,
                "production_components": [
                    {"score_column": column, "display_name": column}
                    for column in columns
                ],
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )


def test_precomputed_first_live_fallback_and_persistence(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "production"
    score_file = tmp_path / "results" / "production" / "finalized_training_frame.csv"
    _make_bundle(model_dir)
    score_file.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair_key": "Compound|cid=1||Protein|protein_id=P1",
                "compound_key": "Compound|cid=1",
                "target_key": "Protein|protein_id=P1",
                "label": 1,
                "score__stage1_tabular_extra_trees": 0.75,
                "score__stage3_rgcn_sampled": 0.70,
                "score__stage3_hgt_sampled": 0.80,
                "final_split": "test",
            }
        ]
    ).to_csv(score_file, index=False)

    monkeypatch.setenv("PRODUCTION_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("PRECOMPUTED_SCORE_FRAME", str(score_file))
    monkeypatch.setenv("PREDICTION_SCORE_MODE", "auto")
    monkeypatch.setenv("PREDICTION_PERSIST_NEW_SCORES", "true")
    monkeypatch.setenv("PREDICTION_REQUIRE_CACHE_WRITE", "true")
    monkeypatch.setenv("PREDICTION_CACHE_BACKUP", "false")
    monkeypatch.setenv("PREDICTION_MAX_PAIRS", "9")

    import pring_modeling.prediction_service as module

    monkeypatch.setattr(module, "GraphDatabase", None)
    service = module.PRINGPredictionService()
    fake_live = FakeLiveScorer()
    service.live = fake_live

    payload = service.predict_many(["CID 1", "CID 2"], ["P1"])
    assert payload["successful_pairs"] == 2
    assert payload["live_predictions_generated"] == 1
    assert fake_live.calls == 1

    by_cid = {item["pair"]["cid"]: item for item in payload["predictions"]}
    assert by_cid["1"]["model"]["score_source"] == "precomputed_validated_model_outputs"
    assert by_cid["2"]["model"]["score_source"] == "live_component_inference"
    assert by_cid["2"]["prediction_cache"]["status"] == "written"

    updated = pd.read_csv(score_file)
    assert len(updated) == 2
    cached = updated.loc[updated["compound_key"] == "Compound|cid=2"].iloc[0]
    assert cached["record_type"] == "production_prediction_cache"
    assert str(cached["exclude_from_training"]).lower() == "true"
    assert cached["final_split"] == "production_inference"
    assert pd.isna(cached["label"])

    second = service.predict_many(["CID 2"], ["P1"])
    assert second["successful_pairs"] == 1
    assert second["live_predictions_generated"] == 0
    assert fake_live.calls == 1
    assert second["predictions"][0]["model"]["score_source"] == "live_component_inference_cached"
