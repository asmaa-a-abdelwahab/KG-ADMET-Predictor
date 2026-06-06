"""Helpers for reading PRING run artifacts and model outputs in Streamlit."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def run_artifact_overview(run_dir: Path) -> dict:
    graph = run_dir / "graph"
    ml = graph / "ml"
    return {
        "run_dir": str(run_dir),
        "exists": run_dir.exists(),
        "manifest": read_json(run_dir / "manifest.json"),
        "quality_report": read_json(graph / "run_quality_report.json"),
        "csv_summary": read_json(graph / "csv_export_summary.json"),
        "ml_manifest": read_json(ml / "modeling_readiness_manifest.json"),
        "ml_files": sorted([p.name for p in ml.glob("*.csv")]) if ml.exists() else [],
    }


def load_predictions(model_dir: Path) -> pd.DataFrame:
    path = model_dir / "predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_model_metrics(model_dir: Path) -> dict:
    return read_json(model_dir / "metrics.json")


def run_pring_eda(run_dir: Path, output_dir: Path) -> tuple[bool, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pring",
        "eda",
        "--run-path",
        str(run_dir),
        "--output-dir",
        str(output_dir),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    message = completed.stdout + "\n" + completed.stderr
    return completed.returncode == 0, message
