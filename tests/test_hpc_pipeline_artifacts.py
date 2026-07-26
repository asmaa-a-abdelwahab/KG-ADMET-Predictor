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
        models / "stage2_distmult_supervised" / "metrics.json",
        models / "stage3_rgcn_sampled" / "metrics.json",
        models / "stage3_hgt_sampled" / "metrics.json",
        models / "finalized_v2" / "seed_metrics.csv",
        models / "finalized_v2" / "seed_metric_summary.csv",
        reports / "comparison" / "model_comparison.csv",
    ):
        _write(path)
    _write_json(
        models / "finalized_v2" / "metrics.json",
        {
            "status": "trained",
            "publishable": True,
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
