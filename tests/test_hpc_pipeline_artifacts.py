from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "examples"
    / "hpc"
    / "validate_pipeline_artifacts.py"
)
SPEC = importlib.util.spec_from_file_location("validate_pipeline_artifacts", SCRIPT)
assert SPEC and SPEC.loader
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


TARGETS = ["P08684", "P05177", "P33261", "P11712", "P10635"]


def test_local_and_hpc_launchers_share_complete_scientific_contract() -> None:
    app_root = Path(__file__).parents[1]
    hpc = (
        app_root / "examples" / "hpc" / "04_full_cyp450_pipeline.sbatch"
    ).read_text(encoding="utf-8")
    local = (
        app_root / "examples" / "local" / "05_full_cyp450_pipeline.sh"
    ).read_text(encoding="utf-8")

    assert 'bash "$APP_DIR/examples/hpc/04_full_cyp450_pipeline.sbatch"' in local
    assert 'LOCAL_NEO4J_MODE="${LOCAL_NEO4J_MODE:-managed}"' in local
    assert '--case-study-mode final-cyp450' in hpc
    assert '--weak-activity-as-negative "${PRING_WEAK_ACTIVITY_AS_NEGATIVE:-false}"' in hpc
    for uncapped_option in (
        "--max-compounds-per-target none",
        "--max-targets-per-compound none",
        "--max-substances-per-compound none",
        "--max-measuregroups-per-target none",
        "--max-measuregroups-per-compound none",
        "--max-endpoints-per-pair none",
        "--max-similar-compounds-per-compound none",
        "--max-textmine-records none",
        "--max-textmine-records-per-target none",
        "--max-textmine-references-per-pair none",
        "--max-enrichment-records-per-entity none",
        "--max-candidate-missing-pairs none",
    ):
        assert uncapped_option in hpc
    for model in (
        "stage1_gds_${MODEL_STAGE1_CLASSIFIER}",
        "stage2_${kge_model}_supervised",
        "stage3_rgcn_sampled",
        "stage3_hgt_sampled",
        "finalized_v2",
    ):
        assert model in hpc


def _write(path: Path, value: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _write_json(path: Path, value: dict) -> Path:
    return _write(path, json.dumps(value))


def _prepared_run(tmp_path: Path, *, overlap: bool = False) -> Path:
    run_dir = tmp_path / "run"
    modeling = run_dir / "graph" / "ml" / "modeling"
    stage1 = modeling / "stage1_neo4j_gds_baselines"
    stage2 = modeling / "stage2_kg_embedding_baselines"
    stage3 = modeling / "stage3_heterogeneous_gnn"

    _write_json(run_dir / "manifest.json", {"manifest_schema": "test"})
    _write_json(
        run_dir / "graph" / "run_quality_report.json",
        {
            "cap_completeness_report": {
                "data_completeness_status": (
                    "uncapped_or_no_internal_caps_detected"
                )
            },
            "cyp450_gcn_readiness_report": {
                "pipeline_validation_ready": True,
                "blockers": [],
                "warnings": [],
                "ml_pair_summary": {"candidate_missing_pair_mode": "all"},
            },
            "similarity_report": {"dangling_similarity_edges": 0},
        },
    )
    _write_json(
        run_dir / "graph" / "ml" / "modeling_readiness_manifest.json",
        {"status": "gcn_modeling_ready", "gcn_ready": True, "blockers": []},
    )
    _write_json(
        modeling / "modeling_stage_manifest.json",
        {
            "format": "pring_modeling_stage_exports_v2",
            "repository": "PRING-PACKAGE",
            "dataset_id": "dataset",
            "split_registry_id": "split",
            "feature_schema_id": "features",
            "label_policy_id": "labels",
            "graph_scope": "cold_compound_inductive_train_only",
            "prediction_contamination_control": {"exclude_from_training": True},
        },
    )
    _write_json(
        stage1 / "stage1_outcome_safe_gds_summary.json",
        {
            "dataset_id": "dataset",
            "split_registry_id": "split",
            "graph_scope": "transductive_node_set_outcome_safe",
            "included_relationship_types": [
                "PRING_TRAIN_POSITIVE",
                "SIMILAR_TO",
            ],
        },
    )

    pair_file = stage1 / "compound_target_training_pairs_for_gds.csv"
    pair_file.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "compound_node_ref",
        "protein_node_ref",
        "label",
        "split",
        "split_group",
    ]
    with pair_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, target in enumerate(TARGETS):
            split = ("train", "validation", "test", "train", "validation")[index]
            writer.writerow(
                {
                    "compound_node_ref": f"Compound|cid={index}",
                    "protein_node_ref": f"Protein|protein_id={target}",
                    "label": str(index % 2),
                    "split": split,
                    "split_group": "shared" if overlap and index < 2 else f"group-{index}",
                }
            )

    for path in (
        stage1 / "candidate_pairs_for_gds_scoring.csv",
        stage1 / "cypher" / "02_fastrp_embeddings.cypher",
        stage2 / "train_graph_triples_leakage_safe.tsv",
        stage2 / "target_relation_valid.tsv",
        stage2 / "target_relation_test.tsv",
        stage3 / "edge_index_train_only.csv",
        stage3 / "compound_target_training_pairs.csv",
        stage3 / "pyg_export" / "heterodata.pt",
    ):
        _write(path)
    return run_dir


def _eda(tmp_path: Path) -> Path:
    eda = tmp_path / "eda"
    for filename in ("eda_report.html", "eda_report.md", "eda_summary.json"):
        _write(eda / filename)
    _write(eda / "figures" / "plot.png")
    _write(eda / "tables" / "summary.csv")
    return eda


def _models(tmp_path: Path) -> tuple[Path, Path]:
    models = tmp_path / "models"
    reports = tmp_path / "reports"
    for path in (
        models / "stage1_gds_extra_trees" / "metrics.json",
        models / "stage2_complex_supervised" / "metrics.json",
        models / "stage2_distmult_supervised" / "metrics.json",
        models / "stage2_rotate_supervised" / "metrics.json",
        models / "stage3_rgcn_sampled" / "metrics.json",
        models / "stage3_hgt_sampled" / "metrics.json",
        models / "finalized_v2" / "seed_metric_summary.csv",
        models / "finalized_v2" / "merged_base_prediction_frame.csv",
        reports / "comparison" / "model_comparison.csv",
        reports / "comparison" / "model_comparison.md",
        reports / "comparison" / "model_comparison_report.html",
    ):
        _write(path)
    _write(
        models / "finalized_v2" / "seed_metrics.csv",
        "seed,mcc\n1,0.1\n2,0.2\n3,0.3\n4,0.4\n5,0.5\n",
    )
    for seed in range(1, 6):
        seed_dir = models / "finalized_v2" / f"seed_{seed}"
        for filename in (
            "metrics.json",
            "predictions.csv",
            "finalized_training_frame.csv",
            "calibration_bins.csv",
            "common_test_model_metrics.csv",
            "per_target_metrics.csv",
            "top_k_by_target.csv",
            "most_uncertain_predictions.csv",
            "finalized_ensemble.joblib",
        ):
            _write(seed_dir / filename)
    _write_json(
        models / "finalized_v2" / "metrics.json",
        {
            "status": "trained",
            "publishable": True,
            "implementation": "improved_v2",
            "base_score_protocol": (
                "fixed_equal_weight_combiner_selected_before_test_no_meta_training"
            ),
            "dataset_id": "dataset",
            "split_registry_id": "split",
            "label_policy_id": "labels",
            "scientific_release_blockers": [],
        },
    )
    return models, reports


def test_strict_final_validation_accepts_complete_artifacts(tmp_path: Path) -> None:
    run_dir = _prepared_run(tmp_path)
    eda = _eda(tmp_path)
    models, reports = _models(tmp_path)
    result = validator.main(
        [
            "--run-dir",
            str(run_dir),
            "--phase",
            "final",
            "--eda-dir",
            str(eda),
            "--model-output-dir",
            str(models),
            "--model-report-dir",
            str(reports),
            "--output-json",
            str(tmp_path / "readiness.json"),
            "--output-markdown",
            str(tmp_path / "readiness.md"),
            "--strict",
        ]
    )
    assert result == 0
    payload = json.loads((tmp_path / "readiness.json").read_text(encoding="utf-8"))
    assert payload["status"] == "pass"


def test_strict_final_validation_requires_all_three_kge_models(
    tmp_path: Path,
) -> None:
    run_dir = _prepared_run(tmp_path)
    eda = _eda(tmp_path)
    models, reports = _models(tmp_path)
    (models / "stage2_rotate_supervised" / "metrics.json").unlink()
    result = validator.main(
        [
            "--run-dir",
            str(run_dir),
            "--phase",
            "final",
            "--eda-dir",
            str(eda),
            "--model-output-dir",
            str(models),
            "--model-report-dir",
            str(reports),
            "--output-json",
            str(tmp_path / "readiness.json"),
            "--output-markdown",
            str(tmp_path / "readiness.md"),
            "--strict",
        ]
    )
    assert result == 2
    payload = json.loads((tmp_path / "readiness.json").read_text(encoding="utf-8"))
    rotate = next(
        check for check in payload["checks"]
        if check["name"] == "Stage 2 rotate metrics"
    )
    assert rotate["status"] == "fail"


def test_strict_prepared_validation_rejects_split_group_overlap(
    tmp_path: Path,
) -> None:
    run_dir = _prepared_run(tmp_path, overlap=True)
    result = validator.main(
        [
            "--run-dir",
            str(run_dir),
            "--phase",
            "prepared",
            "--output-json",
            str(tmp_path / "readiness.json"),
            "--output-markdown",
            str(tmp_path / "readiness.md"),
            "--strict",
        ]
    )
    assert result == 2
    payload = json.loads((tmp_path / "readiness.json").read_text(encoding="utf-8"))
    split_check = next(
        check for check in payload["checks"] if check["name"] == "split-group isolation"
    )
    assert split_check["status"] == "fail"
