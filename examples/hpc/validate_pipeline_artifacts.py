#!/usr/bin/env python3
"""Validate artifacts produced by the end-to-end PRING HPC workflow.

The validator deliberately reads only immutable run/model outputs.  It does not
repair files, infer missing splits, or copy predictions into training data.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FIVE_CYP_ACCESSIONS = {
    "P08684": "CYP3A4",
    "P05177": "CYP1A2",
    "P33261": "CYP2C19",
    "P11712": "CYP2C9",
    "P10635": "CYP2D6",
}


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    critical: bool = True


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _add_file_check(
    checks: list[Check], name: str, path: Path, *, critical: bool = True
) -> None:
    checks.append(
        Check(
            name=name,
            status="pass" if _nonempty(path) else "fail",
            detail=str(path),
            critical=critical,
        )
    )


def _normalise_split(value: str) -> str:
    value = value.strip().lower()
    return "validation" if value in {"valid", "val", "validation"} else value


def _scan_supervised_pairs(path: Path) -> dict[str, Any]:
    targets: set[str] = set()
    splits: set[str] = set()
    labels: set[str] = set()
    group_to_split: dict[str, str] = {}
    pairs: set[tuple[str, str]] = set()
    duplicate_pairs = 0
    group_overlap = 0
    rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "compound_node_ref",
            "protein_node_ref",
            "label",
            "split",
            "split_group",
        }
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing columns in {path}: {', '.join(missing)}")

        for row in reader:
            rows += 1
            compound = (row.get("compound_node_ref") or "").strip()
            protein = (row.get("protein_node_ref") or "").strip()
            label = (row.get("label") or "").strip()
            split = _normalise_split(row.get("split") or "")
            group = (row.get("split_group") or compound).strip()

            for accession in FIVE_CYP_ACCESSIONS:
                if accession in protein:
                    targets.add(accession)
            labels.add(label)
            splits.add(split)

            pair = (compound, protein)
            if pair in pairs:
                duplicate_pairs += 1
            else:
                pairs.add(pair)

            previous = group_to_split.setdefault(group, split)
            if previous != split:
                group_overlap += 1

    return {
        "rows": rows,
        "targets": sorted(targets),
        "splits": sorted(splits),
        "labels": sorted(labels),
        "duplicate_pairs": duplicate_pairs,
        "split_group_overlap_rows": group_overlap,
    }


def _check_prepared_run(
    run_dir: Path, checks: list[Check], *, require_gds_summary: bool = False
) -> dict[str, Any]:
    modeling_dir = run_dir / "graph" / "ml" / "modeling"
    manifest_path = run_dir / "manifest.json"
    model_manifest_path = modeling_dir / "modeling_stage_manifest.json"

    _add_file_check(checks, "run manifest", manifest_path)
    _add_file_check(checks, "modeling-stage manifest", model_manifest_path)

    required_files = {
        "Stage 1 supervised pairs": (
            modeling_dir
            / "stage1_neo4j_gds_baselines"
            / "compound_target_training_pairs_for_gds.csv"
        ),
        "Stage 1 candidate pairs": (
            modeling_dir
            / "stage1_neo4j_gds_baselines"
            / "candidate_pairs_for_gds_scoring.csv"
        ),
        "Stage 1 FastRP script": (
            modeling_dir
            / "stage1_neo4j_gds_baselines"
            / "cypher"
            / "02_fastrp_embeddings.cypher"
        ),
        "Stage 2 leakage-safe triples": (
            modeling_dir
            / "stage2_kg_embedding_baselines"
            / "train_graph_triples_leakage_safe.tsv"
        ),
        "Stage 2 validation triples": (
            modeling_dir
            / "stage2_kg_embedding_baselines"
            / "target_relation_valid.tsv"
        ),
        "Stage 2 locked-test triples": (
            modeling_dir
            / "stage2_kg_embedding_baselines"
            / "target_relation_test.tsv"
        ),
        "Stage 3 train-only graph": (
            modeling_dir
            / "stage3_heterogeneous_gnn"
            / "edge_index_train_only.csv"
        ),
        "Stage 3 supervised pairs": (
            modeling_dir
            / "stage3_heterogeneous_gnn"
            / "compound_target_training_pairs.csv"
        ),
        "Stage 3 PyG graph": (
            modeling_dir
            / "stage3_heterogeneous_gnn"
            / "pyg_export"
            / "heterodata.pt"
        ),
    }
    for name, path in required_files.items():
        _add_file_check(checks, name, path)

    summary: dict[str, Any] = {"modeling_dir": str(modeling_dir)}
    manifest: dict[str, Any] = {}
    if _nonempty(model_manifest_path):
        manifest = _load_json(model_manifest_path)
        summary["modeling_manifest"] = {
            key: manifest.get(key)
            for key in (
                "format",
                "repository",
                "dataset_id",
                "split_registry_id",
                "feature_schema_id",
                "label_policy_id",
                "graph_scope",
            )
        }
        for key in (
            "dataset_id",
            "split_registry_id",
            "feature_schema_id",
            "label_policy_id",
        ):
            value = str(manifest.get(key) or "")
            checks.append(
                Check(
                    name=f"provenance field: {key}",
                    status="pass" if value else "fail",
                    detail=value or "missing",
                )
            )
        graph_scope = str(manifest.get("graph_scope") or "")
        checks.append(
            Check(
                name="train-only graph scope",
                status=(
                    "pass"
                    if graph_scope == "cold_compound_inductive_train_only"
                    else "fail"
                ),
                detail=graph_scope or "missing",
            )
        )
        contamination = manifest.get("prediction_contamination_control") or {}
        prediction_excluded = (
            isinstance(contamination, dict)
            and contamination.get("exclude_from_training") is True
        )
        checks.append(
            Check(
                name="prediction contamination control",
                status="pass" if prediction_excluded else "fail",
                detail=json.dumps(contamination, sort_keys=True),
            )
        )

    if require_gds_summary:
        gds_summary_path = (
            modeling_dir
            / "stage1_neo4j_gds_baselines"
            / "stage1_outcome_safe_gds_summary.json"
        )
        _add_file_check(
            checks, "Stage 1 outcome-safe GDS summary", gds_summary_path
        )
        if _nonempty(gds_summary_path):
            try:
                gds_summary = _load_json(gds_summary_path)
                summary["stage1_gds"] = gds_summary
                graph_scope = str(gds_summary.get("graph_scope") or "")
                checks.append(
                    Check(
                        name="Stage 1 outcome-safe graph scope",
                        status=(
                            "pass"
                            if graph_scope
                            == "transductive_node_set_outcome_safe"
                            else "fail"
                        ),
                        detail=graph_scope or "missing",
                    )
                )
                included = set(gds_summary.get("included_relationship_types") or [])
                unsafe = included.intersection(
                    {
                        "ASSERTS_CHEMICAL",
                        "ASSERTS_TARGET",
                        "SUPPORTED_BY_ASSAY",
                        "SUPPORTED_BY_ENDPOINT",
                        "OBSERVED_INTERACTS_WITH",
                    }
                )
                checks.append(
                    Check(
                        name="Stage 1 held-out outcome exclusion",
                        status="pass" if not unsafe else "fail",
                        detail=(
                            "no held-out outcome relationship types projected"
                            if not unsafe
                            else f"unsafe relationships: {', '.join(sorted(unsafe))}"
                        ),
                    )
                )
                for key in ("dataset_id", "split_registry_id"):
                    expected = str(manifest.get(key) or "")
                    actual = str(gds_summary.get(key) or "")
                    checks.append(
                        Check(
                            name=f"Stage 1/source provenance match: {key}",
                            status=(
                                "pass"
                                if expected and actual == expected
                                else "fail"
                            ),
                            detail=(
                                f"expected={expected or 'missing'}; "
                                f"actual={actual or 'missing'}"
                            ),
                        )
                    )
            except Exception as exc:
                checks.append(
                    Check(
                        name="Stage 1 GDS summary JSON",
                        status="fail",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )

    pair_path = required_files["Stage 1 supervised pairs"]
    if _nonempty(pair_path):
        try:
            pair_summary = _scan_supervised_pairs(pair_path)
            summary["supervised_pairs"] = pair_summary
            missing_targets = sorted(
                set(FIVE_CYP_ACCESSIONS).difference(pair_summary["targets"])
            )
            checks.append(
                Check(
                    name="five-CYP target coverage",
                    status="pass" if not missing_targets else "fail",
                    detail=(
                        "all five accessions present"
                        if not missing_targets
                        else f"missing: {', '.join(missing_targets)}"
                    ),
                )
            )
            missing_splits = sorted(
                {"train", "validation", "test"}.difference(pair_summary["splits"])
            )
            checks.append(
                Check(
                    name="registered train/validation/test partitions",
                    status="pass" if not missing_splits else "fail",
                    detail=(
                        ", ".join(pair_summary["splits"])
                        if not missing_splits
                        else f"missing: {', '.join(missing_splits)}"
                    ),
                )
            )
            checks.append(
                Check(
                    name="binary supervised labels",
                    status=(
                        "pass"
                        if set(pair_summary["labels"]).issubset({"0", "1"})
                        else "fail"
                    ),
                    detail=", ".join(pair_summary["labels"]),
                )
            )
            checks.append(
                Check(
                    name="unique supervised compound-target pairs",
                    status=(
                        "pass"
                        if pair_summary["duplicate_pairs"] == 0
                        else "fail"
                    ),
                    detail=f"duplicates={pair_summary['duplicate_pairs']}",
                )
            )
            checks.append(
                Check(
                    name="split-group isolation",
                    status=(
                        "pass"
                        if pair_summary["split_group_overlap_rows"] == 0
                        else "fail"
                    ),
                    detail=(
                        "overlap_rows="
                        f"{pair_summary['split_group_overlap_rows']}"
                    ),
                )
            )
        except Exception as exc:  # pragma: no cover - defensive reporting
            checks.append(
                Check(
                    name="supervised-pair audit",
                    status="fail",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
    return summary


def _check_eda(eda_dir: Path | None, checks: list[Check]) -> None:
    if eda_dir is None:
        return
    for filename in ("eda_report.html", "eda_report.md", "eda_summary.json"):
        _add_file_check(checks, f"EDA output: {filename}", eda_dir / filename)
    checks.append(
        Check(
            name="EDA figures",
            status=(
                "pass"
                if (eda_dir / "figures").is_dir()
                and any((eda_dir / "figures").glob("*.png"))
                else "fail"
            ),
            detail=str(eda_dir / "figures"),
        )
    )
    checks.append(
        Check(
            name="EDA tables",
            status=(
                "pass"
                if (eda_dir / "tables").is_dir()
                and any((eda_dir / "tables").glob("*.csv"))
                else "fail"
            ),
            detail=str(eda_dir / "tables"),
        )
    )


def _check_final_models(
    model_output_dir: Path | None,
    model_report_dir: Path | None,
    checks: list[Check],
    expected_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if model_output_dir is None:
        checks.append(
            Check("model output directory", "fail", "not provided")
        )
        return summary

    final_dir = model_output_dir / "finalized_v2"
    metrics_path = final_dir / "metrics.json"
    required = {
        "final validation metrics": metrics_path,
        "multi-seed metrics": final_dir / "seed_metrics.csv",
        "multi-seed metric summary": final_dir / "seed_metric_summary.csv",
        "R-GCN metrics": (
            model_output_dir / "stage3_rgcn_sampled" / "metrics.json"
        ),
        "HGT metrics": model_output_dir / "stage3_hgt_sampled" / "metrics.json",
    }
    for name, path in required.items():
        _add_file_check(checks, name, path)

    stage1_metrics = list(model_output_dir.glob("stage1_gds_*/metrics.json"))
    checks.append(
        Check(
            name="Stage 1 metrics",
            status="pass" if stage1_metrics else "fail",
            detail=f"files={len(stage1_metrics)}",
        )
    )
    kge_metrics = list(model_output_dir.glob("stage2_*_supervised/metrics.json"))
    checks.append(
        Check(
            name="knowledge-graph embedding metrics",
            status="pass" if kge_metrics else "fail",
            detail=f"files={len(kge_metrics)}",
        )
    )

    if _nonempty(metrics_path):
        try:
            metrics = _load_json(metrics_path)
            summary["final_validation"] = {
                key: metrics.get(key)
                for key in (
                    "status",
                    "publishable",
                    "dataset_id",
                    "split_registry_id",
                    "label_policy_id",
                    "scientific_release_blockers",
                )
            }
            checks.append(
                Check(
                    name="strict final-validation release gate",
                    status="pass" if metrics.get("publishable") is True else "fail",
                    detail=(
                        f"status={metrics.get('status')}; "
                        f"blockers={metrics.get('scientific_release_blockers')}"
                    ),
                )
            )
            for key in ("dataset_id", "split_registry_id", "label_policy_id"):
                expected = str((expected_provenance or {}).get(key) or "")
                actual = str(metrics.get(key) or "")
                checks.append(
                    Check(
                        name=f"final/source provenance match: {key}",
                        status=(
                            "pass"
                            if expected and actual == expected
                            else "fail"
                        ),
                        detail=f"expected={expected or 'missing'}; actual={actual or 'missing'}",
                    )
                )
        except Exception as exc:
            checks.append(
                Check(
                    name="final metrics JSON",
                    status="fail",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )

    if model_report_dir is not None:
        comparison_dir = model_report_dir / "comparison"
        checks.append(
            Check(
                name="model-comparison report",
                status=(
                    "pass"
                    if comparison_dir.is_dir()
                    and any(path.is_file() for path in comparison_dir.rglob("*"))
                    else "fail"
                ),
                detail=str(comparison_dir),
            )
        )
    return summary


def _markdown_report(payload: dict[str, Any]) -> str:
    status = payload["status"].upper()
    lines = [
        "# PRING HPC pipeline readiness report",
        "",
        f"- Status: **{status}**",
        f"- Phase: `{payload['phase']}`",
        f"- Generated (UTC): `{payload['generated_at']}`",
        f"- Run directory: `{payload['run_dir']}`",
        "",
        "| Check | Status | Critical | Detail |",
        "|---|---:|---:|---|",
    ]
    for check in payload["checks"]:
        detail = str(check["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {check['name']} | {check['status'].upper()} | "
            f"{'yes' if check['critical'] else 'no'} | {detail} |"
        )
    lines.extend(
        [
            "",
            (
                "The Stage 1 FastRP component is outcome-safe but transductive "
                "with respect to node presence and the compound-similarity graph. "
                "An ensemble containing that component must be reported as "
                "transductive, even when its supervised pairs use the registered "
                "compound-group split."
            ),
            "",
            "A passing workflow-readiness report confirms artifact and provenance "
            "contracts. It does not itself establish biological or clinical validity.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate prepared or final artifacts from the PRING HPC pipeline."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("prepared", "final"), default="prepared"
    )
    parser.add_argument("--eda-dir", type=Path)
    parser.add_argument("--model-output-dir", type=Path)
    parser.add_argument("--model-report-dir", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when any critical check fails.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    checks: list[Check] = []
    details = _check_prepared_run(
        run_dir, checks, require_gds_summary=args.phase == "final"
    )
    _check_eda(
        args.eda_dir.expanduser().resolve() if args.eda_dir else None, checks
    )
    if args.phase == "final":
        details.update(
            _check_final_models(
                (
                    args.model_output_dir.expanduser().resolve()
                    if args.model_output_dir
                    else None
                ),
                (
                    args.model_report_dir.expanduser().resolve()
                    if args.model_report_dir
                    else None
                ),
                checks,
                details.get("modeling_manifest") or {},
            )
        )

    failed_critical = [
        check for check in checks if check.critical and check.status != "pass"
    ]
    payload = {
        "format": "pring-hpc-pipeline-readiness-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "status": "pass" if not failed_critical else "fail",
        "run_dir": str(run_dir),
        "checks": [asdict(check) for check in checks],
        "details": details,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output_markdown.write_text(
        _markdown_report(payload), encoding="utf-8"
    )

    print(json.dumps(payload, indent=2))
    if args.strict and failed_critical:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
