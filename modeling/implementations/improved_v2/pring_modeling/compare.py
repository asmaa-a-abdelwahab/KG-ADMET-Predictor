from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .common import ensure_dir, read_table

PREFERRED_METRICS = ["mcc", "balanced_accuracy", "youden_j", "roc_auc", "average_precision", "specificity", "negative_precision", "accuracy", "f1", "precision", "recall", "positive_ratio", "negative_ratio", "hits_at_1", "hits_at_5", "hits_at_10", "mrr"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_metrics(obj: Any, prefix: str = "") -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}_{key}" if prefix else str(key)
            if isinstance(value, dict):
                out.update(flatten_metrics(value, prefix=name if key != "metrics" else prefix))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                out[name] = float(value)
            elif key in {"model", "model_name", "stage", "encoder", "decoder", "status"}:
                out[name] = str(value)
    return out


def infer_stage_and_model(run_dir: Path, metrics: dict[str, Any]) -> tuple[str, str]:
    model = str(metrics.get("model") or metrics.get("model_name") or run_dir.name)
    stage = str(metrics.get("stage") or "")
    if stage:
        return stage, model
    lower = run_dir.name.lower()
    if "stage1" in lower or "gds" in lower or "fastrp" in lower or "graphsage" in lower:
        return "Stage 1 — Neo4j GDS baseline", model
    if "stage2" in lower or "rotate" in lower or "distmult" in lower or "complex" in lower or "kge" in lower:
        return "Stage 2 — KG embedding baseline", model
    if "stage3" in lower or "hgt" in lower or "rgcn" in lower:
        return "Stage 3 — Heterogeneous GNN", model
    return "Unspecified", model


def find_metric_files(paths: list[str], output_root: str | None) -> list[Path]:
    metric_files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.name == "metrics.json":
            metric_files.append(path)
        elif path.is_dir():
            direct = path / "metrics.json"
            if direct.exists():
                metric_files.append(direct)
            else:
                metric_files.extend(path.rglob("metrics.json"))
    if output_root:
        metric_files.extend(Path(output_root).rglob("metrics.json"))
    seen: set[Path] = set(); unique: list[Path] = []
    for p in metric_files:
        rp = p.resolve()
        if rp not in seen:
            unique.append(p); seen.add(rp)
    return unique


def compare_metrics(runs: list[str], outputs_root: str | None, output_dir: str, primary_metric: str = "mcc") -> pd.DataFrame:
    out_dir = ensure_dir(output_dir)
    metric_files = find_metric_files(runs, outputs_root)
    if not metric_files:
        raise FileNotFoundError("No metrics.json files found. Pass --runs or --outputs-root.")
    rows: list[dict[str, Any]] = []
    for metric_file in metric_files:
        obj = load_json(metric_file)
        flat = flatten_metrics(obj)
        stage, model = infer_stage_and_model(metric_file.parent, flat)
        row = {"stage": stage, "model": model, "run_dir": str(metric_file.parent), "metrics_file": str(metric_file)}
        row.update(flat)
        rows.append(row)
    df = pd.DataFrame(rows)
    metric_cols = [c for c in PREFERRED_METRICS if c in df.columns]
    sort_metric = primary_metric if primary_metric in df.columns else (metric_cols[0] if metric_cols else None)
    if sort_metric:
        df = df.sort_values(sort_metric, ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.to_csv(out_dir / "model_comparison.csv", index=False)
    md_cols = [c for c in ["rank", "stage", "model", "run_dir", *metric_cols] if c in df.columns]
    md = "# PRING model comparison\n\n"
    if sort_metric:
        md += f"Models are ranked by `{sort_metric}` in descending order.\n\n"
    md += df[md_cols].to_markdown(index=False) + "\n\n## Included metric files\n" + "\n".join(f"- `{p}`" for p in metric_files) + "\n"
    (out_dir / "model_comparison.md").write_text(md, encoding="utf-8")
    return df


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return
    plot_df["display_name"] = plot_df.apply(lambda r: f"{r.get('model', 'model')}\n{r.get('stage', '')}"[:90], axis=1)
    fig_height = max(4.0, 0.55 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    ax.barh(plot_df["display_name"], plot_df[metric])
    ax.invert_yaxis(); ax.set_xlabel(metric.replace("_", " ").title()); ax.set_ylabel("Model")
    ax.set_title(f"Model comparison — {metric.replace('_', ' ').title()}")
    ax.grid(axis="x", alpha=0.25)
    max_val = plot_df[metric].max()
    for idx, value in enumerate(plot_df[metric]):
        ax.text(value + max(0.01, max_val * 0.01), idx, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=180, bbox_inches="tight"); plt.close(fig)


def make_html_report(df: pd.DataFrame, image_files: list[Path], out_path: Path) -> None:
    rows = []
    for _, row in df.iterrows():
        metrics = []
        for col in PREFERRED_METRICS:
            if col in df.columns and pd.notna(row.get(col)):
                metrics.append(f"<span><b>{html.escape(col)}</b>: {float(row[col]):.4f}</span>")
        rows.append(f"<tr><td>{row.get('rank','')}</td><td>{html.escape(str(row.get('stage','')))}</td><td>{html.escape(str(row.get('model','')))}</td><td>{' · '.join(metrics)}</td><td><code>{html.escape(str(row.get('run_dir','')))}</code></td></tr>")
    images = "\n".join(f'<section><h2>{html.escape(img.stem.replace("_", " ").title())}</h2><img src="{html.escape(img.name)}" alt="{html.escape(img.stem)}"></section>' for img in image_files)
    doc = f"""<!doctype html><html><head><meta charset='utf-8'><title>PRING model comparison</title><style>
body {{ font-family: Arial, sans-serif; margin: 32px; background:#f8fafc; color:#111827; }}
header, section {{ background:#fff; border:1px solid #e5e7eb; border-radius:16px; padding:20px; margin-bottom:18px; box-shadow:0 10px 25px rgba(17,24,39,.06); }}
img {{ max-width:100%; height:auto; border:1px solid #e5e7eb; border-radius:12px; }} table {{ width:100%; border-collapse:collapse; }}
th, td {{ border-bottom:1px solid #e5e7eb; padding:10px; text-align:left; vertical-align:top; }} th {{ background:#f3f4f6; }} span {{ display:inline-block; margin-right:10px; margin-bottom:5px; }}
</style></head><body><header><h1>PRING model comparison</h1><p>Comparison of Stage 1, Stage 2, and Stage 3 CYP450 link-prediction models.</p></header>{images}<section><h2>Ranked results</h2><table><thead><tr><th>Rank</th><th>Stage</th><th>Model</th><th>Metrics</th><th>Run directory</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section></body></html>"""
    out_path.write_text(doc, encoding="utf-8")


def visualize(comparison_csv: str, output_dir: str, metrics: list[str] | None = None) -> list[Path]:
    out_dir = ensure_dir(output_dir)
    df = pd.read_csv(comparison_csv)
    image_files: list[Path] = []
    for metric in (metrics or ["mcc", "balanced_accuracy", "roc_auc", "average_precision", "specificity", "f1"]):
        if metric not in df.columns:
            continue
        out_path = out_dir / f"comparison_{metric}.png"
        plot_metric(df, metric, out_path)
        if out_path.exists():
            image_files.append(out_path)
    make_html_report(df, image_files, out_dir / "model_comparison_report.html")
    return image_files


def compare_prediction_scores(predictions: list[str], names: list[str] | None, output_dir: str, top_k: int = 100) -> None:
    out_dir = ensure_dir(output_dir)
    frames = []
    for i, p in enumerate(predictions):
        path = Path(p); name = names[i] if names and i < len(names) else path.parent.name
        df = read_table(path)
        c_col = next((c for c in ["compound_node_ref", "compound_entity_id", "compound_node_id", "head"] if c in df.columns), None)
        p_col = next((c for c in ["protein_node_ref", "protein_entity_id", "protein_node_id", "tail"] if c in df.columns), None)
        s_col = next((c for c in ["score", "prediction", "probability", "raw_score"] if c in df.columns), None)
        if not c_col or not p_col or not s_col:
            continue
        tmp = df[[c_col, p_col, s_col]].copy(); tmp.columns = ["compound_key", "protein_key", name]
        tmp["pair_key"] = tmp["compound_key"].astype(str) + "||" + tmp["protein_key"].astype(str)
        frames.append(tmp[["pair_key", "compound_key", "protein_key", name]])
    if not frames:
        raise ValueError("No usable prediction files were found.")
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["pair_key", "compound_key", "protein_key"], how="outer")
    model_cols = [c for c in merged.columns if c not in {"pair_key", "compound_key", "protein_key"}]
    merged.to_csv(out_dir / "combined_prediction_scores.csv", index=False)
    merged[model_cols].corr(method="spearman", min_periods=5).to_csv(out_dir / "prediction_score_spearman_correlation.csv")
    rows = []
    top_sets = {m: set(merged.nlargest(min(top_k, len(merged)), m)["pair_key"]) for m in model_cols if merged[m].notna().sum()}
    keys = list(top_sets)
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            inter = len(top_sets[a] & top_sets[b]); union = len(top_sets[a] | top_sets[b])
            rows.append({"model_a": a, "model_b": b, "k": top_k, "overlap_count": inter, "overlap_fraction_of_k": inter / max(1, top_k), "jaccard": inter / union if union else 0.0})
    pd.DataFrame(rows).to_csv(out_dir / "topk_prediction_overlap.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare and visualize PRING model results.")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("metrics"); c.add_argument("--runs", nargs="*", default=[]); c.add_argument("--outputs-root"); c.add_argument("--output-dir", required=True); c.add_argument("--primary-metric", default="mcc")
    v = sub.add_parser("visualize"); v.add_argument("--comparison-csv", required=True); v.add_argument("--output-dir", required=True); v.add_argument("--metrics", nargs="*")
    ps = sub.add_parser("predictions"); ps.add_argument("--predictions", nargs="+", required=True); ps.add_argument("--names", nargs="*"); ps.add_argument("--output-dir", required=True); ps.add_argument("--top-k", type=int, default=100)
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "metrics":
        compare_metrics(args.runs, args.outputs_root, args.output_dir, args.primary_metric)
    elif args.cmd == "visualize":
        visualize(args.comparison_csv, args.output_dir, args.metrics)
    else:
        compare_prediction_scores(args.predictions, args.names, args.output_dir, args.top_k)


if __name__ == "__main__":
    main()
