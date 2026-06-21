"""Aggregate model metrics across PRING implementations run on one shared split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

METRICS = [
    "mcc", "balanced_accuracy", "youden_j", "roc_auc", "average_precision", "specificity",
    "negative_precision", "accuracy", "f1", "precision", "recall", "positive_ratio", "negative_ratio",
]


def _flatten_metrics(path: Path, impl: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"implementation": impl, "metrics_file": str(path), "status": "read_error", "error": str(exc)}
    row = {
        "implementation": impl,
        "stage": data.get("stage"),
        "model": data.get("model") or path.parent.name,
        "run_dir": str(path.parent),
        "metrics_file": str(path),
        "status": data.get("status", "unknown"),
    }
    nested = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    for k in METRICS:
        row[k] = data.get(k, nested.get(k))
    return row


def collect(root: Path, implementations: list[str]) -> pd.DataFrame:
    rows = []
    for impl in implementations:
        impl_root = root / impl
        if not impl_root.exists():
            continue
        for path in sorted(impl_root.rglob("metrics.json")):
            rows.append(_flatten_metrics(path, impl))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for k in METRICS:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    return df


def render_markdown(df: pd.DataFrame, primary_metric: str) -> str:
    lines = ["# PRING cross-implementation comparison", "", f"Models are ranked by `{primary_metric}` across implementations run on the same materialized split.", ""]
    if df.empty:
        return "\n".join(lines + ["No metrics were found.\n"])
    cols = ["rank", "implementation", "stage", "model", "mcc", "balanced_accuracy", "roc_auc", "average_precision", "specificity", "accuracy", "f1", "precision", "recall", "positive_ratio", "negative_ratio"]
    show = df.copy().sort_values(primary_metric, ascending=False, na_position="last").reset_index(drop=True)
    show.insert(0, "rank", range(1, len(show) + 1))
    for c in cols:
        if c not in show.columns:
            show[c] = None
    lines.append(show[cols].to_markdown(index=False))
    lines.extend(["", "## Best model per implementation", ""])
    best = show.dropna(subset=[primary_metric]).sort_values(primary_metric, ascending=False).groupby("implementation", as_index=False).first()
    if not best.empty:
        lines.append(best[["implementation", "stage", "model", primary_metric, "balanced_accuracy", "specificity", "recall"]].to_markdown(index=False))
    lines.extend(["", "## Notes", "", "- This report is only reliable when all implementations were run with the same `PRING_RUN_DIR` prepared by `pring_modeling.shared_splits`.", "- Check `shared_splits/split_summary.json` for the exact manifest, split strategy, seed, and row counts."])
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare legacy/improved/improved_v2 metrics from shared-split runs.")
    p.add_argument("--outputs-root", type=Path, required=True, help="Root containing implementation subfolders, e.g. models_same_splits/<run_id>")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--implementations", nargs="+", default=["legacy", "improved", "improved_v2"])
    p.add_argument("--primary-metric", default="mcc")
    args = p.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = collect(args.outputs_root, args.implementations)
    if not df.empty and args.primary_metric in df.columns:
        df = df.sort_values(args.primary_metric, ascending=False, na_position="last")
    df.to_csv(args.output_dir / "cross_implementation_metrics.csv", index=False)
    (args.output_dir / "cross_implementation_comparison.md").write_text(render_markdown(df, args.primary_metric), encoding="utf-8")
    print(json.dumps({"rows": int(len(df)), "csv": str(args.output_dir / "cross_implementation_metrics.csv"), "markdown": str(args.output_dir / "cross_implementation_comparison.md")}, indent=2))


if __name__ == "__main__":
    main()
