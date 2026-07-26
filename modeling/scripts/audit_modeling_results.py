#!/usr/bin/env python3
"""Read-only scientific provenance audit for PRING-APP modeling results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HELD_OUT_NAMES = {"test", "holdout", "heldout", "held_out"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_training_frame(path: Path) -> dict[str, Any]:
    split_values: dict[str, set[str]] = defaultdict(set)
    final_split_counts: Counter[str] = Counter()
    record_type_counts: Counter[str] = Counter()
    rows = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        component_split_columns = [name for name in fieldnames if name.startswith("split__")]
        for row in reader:
            rows += 1
            final_split_counts[str(row.get("final_split", "")).strip().lower()] += 1
            record_type_counts[str(row.get("record_type", "")).strip().lower()] += 1
            for column in component_split_columns:
                value = str(row.get(column, "")).strip().lower()
                if value:
                    split_values[column].add(value)

    all_components_held_out = bool(component_split_columns) and all(
        bool(split_values[column])
        and split_values[column].issubset(HELD_OUT_NAMES)
        for column in component_split_columns
    )
    cache_rows = record_type_counts.get("production_prediction_cache", 0)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": rows,
        "component_split_columns": component_split_columns,
        "component_split_values": {
            column: sorted(values) for column, values in split_values.items()
        },
        "all_component_rows_held_out": all_components_held_out,
        "final_split_counts": dict(final_split_counts),
        "production_cache_rows_in_reference": cache_rows,
        "scientific_status": "diagnostic_only" if all_components_held_out else "requires_manifest_verification",
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        return {"_load_error": f"{type(exc).__name__}: {exc}"}


def run(results_dir: Path, model_dir: Path | None = None) -> dict[str, Any]:
    production_dir = results_dir / "production"
    frame_path = production_dir / "finalized_training_frame.csv"
    manifest_path = (model_dir or production_dir) / "manifest.json"
    findings: list[dict[str, str]] = []

    frame_audit: dict[str, Any] = {"status": "missing", "path": str(frame_path)}
    if frame_path.exists():
        frame_audit = audit_training_frame(frame_path)
        if frame_audit["all_component_rows_held_out"]:
            findings.append(
                {
                    "severity": "critical",
                    "code": "RESPLIT_HELDOUT_COMPONENT_SCORES",
                    "message": (
                        "Every component split column is held out while final_split contains "
                        "new train/validation/test assignments. Metrics and calibration are diagnostic only."
                    ),
                }
            )
        if frame_audit["production_cache_rows_in_reference"]:
            findings.append(
                {
                    "severity": "critical",
                    "code": "CACHE_CONTAMINATION",
                    "message": "Production cache rows were found in the immutable reference frame.",
                }
            )
    else:
        findings.append(
            {
                "severity": "critical",
                "code": "MISSING_PRODUCTION_FRAME",
                "message": f"Missing production frame: {frame_path}",
            }
        )

    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    if not manifest_path.exists():
        findings.append(
            {
                "severity": "major",
                "code": "MISSING_PRODUCTION_MANIFEST",
                "message": f"Missing production manifest: {manifest_path}",
            }
        )
    elif frame_audit.get("all_component_rows_held_out") and (
        manifest.get("status") == "ready" or manifest.get("publishable") is not False
    ):
        findings.append(
            {
                "severity": "critical",
                "code": "MANIFEST_PROVENANCE_CONTRADICTION",
                "message": (
                    "The manifest presents the bundle as ready/publishable while the component "
                    "scores are a diagnostic re-split of held-out predictions."
                ),
            }
        )

    metric_manifests = []
    if results_dir.exists():
        for path in sorted(results_dir.rglob("metrics.json")):
            data = read_json(path)
            metric_manifests.append(
                {
                    "path": str(path),
                    "status": data.get("status"),
                    "publishable": data.get("publishable"),
                    "seed": data.get("seed"),
                    "split_audit": data.get("split_audit"),
                }
            )

    severity_counts = Counter(item["severity"] for item in findings)
    return {
        "audit_schema": "pring-app-modeling-results-audit-v1",
        "results_dir": str(results_dir),
        "model_dir": str(model_dir) if model_dir else None,
        "read_only": True,
        "production_frame": frame_audit,
        "production_manifest": {
            "path": str(manifest_path),
            "status": manifest.get("status"),
            "publishable": manifest.get("publishable"),
            "model_name": manifest.get("model_name"),
            "model_version": manifest.get("model_version"),
        },
        "metric_manifests": metric_manifests,
        "findings": findings,
        "severity_counts": dict(severity_counts),
        "production_ready": severity_counts["critical"] == 0,
        "publication_ready": severity_counts["critical"] == 0 and severity_counts["major"] == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="artifacts/results")
    parser.add_argument("--model-dir", default="artifacts/models/production")
    parser.add_argument("--output", default=None, help="Optional JSON output path. The results directory is never modified.")
    parser.add_argument(
        "--allow-diagnostic",
        action="store_true",
        help="Return zero for legacy reproduction even when critical scientific findings are present.",
    )
    args = parser.parse_args()
    report = run(Path(args.results_dir), Path(args.model_dir))
    payload = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if not report["production_ready"] and not args.allow_diagnostic:
        sys.exit(2)


if __name__ == "__main__":
    main()
