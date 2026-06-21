from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from neo4j import GraphDatabase

from .common import ensure_dir, parse_node_ref, read_table, save_json

EVIDENCE_QUERY = """
MATCH (c:Compound)
WHERE ($cid <> '' AND toString(c.cid) = $cid) OR ($compound_ref <> '' AND c.node_ref = $compound_ref)
MATCH (p:Protein)
WHERE ($protein_id <> '' AND (toString(p.protein_id) = $protein_id OR toString(p.accession) = $protein_id)) OR ($protein_ref <> '' AND p.node_ref = $protein_ref)
OPTIONAL MATCH (c)<-[:ASSERTS_CHEMICAL]-(i:Interaction)-[:ASSERTS_TARGET]->(p)
OPTIONAL MATCH (i)-[:SUPPORTED_BY|HAS_MEASURE_GROUP|MEASURED_IN|DESCRIBED_BY*1..2]-(e)
WITH c, p, collect(DISTINCT i)[0..20] AS interactions, collect(DISTINCT e)[0..50] AS evidence_nodes
OPTIONAL MATCH (c)-[:SIMILAR_TO]-(known:Compound)<-[:ASSERTS_CHEMICAL]-(ki:Interaction)-[:ASSERTS_TARGET]->(p)
RETURN c{.*} AS compound, p{.*} AS protein,
       size(interactions) AS direct_interaction_count,
       size(evidence_nodes) AS evidence_node_count,
       collect(DISTINCT known{.*})[0..10] AS similar_known_compounds,
       interactions[0..10] AS direct_interactions
"""


def prediction_row_to_params(row: pd.Series) -> dict[str, str]:
    c_ref = str(row.get("compound_node_ref") or "").strip()
    p_ref = str(row.get("protein_node_ref") or "").strip()
    c = parse_node_ref(c_ref); p = parse_node_ref(p_ref)
    return {
        "compound_ref": c_ref,
        "protein_ref": p_ref,
        "cid": str(c.get("cid") or row.get("cid") or row.get("compound_id") or "").strip(),
        "protein_id": str(p.get("protein_id") or p.get("accession") or row.get("protein_id") or row.get("target_id") or "").strip(),
    }


def explain_record(record: dict[str, Any], row: pd.Series, *, stage_name: str | None = None) -> dict[str, Any]:
    compound = record.get("compound") or {}
    protein = record.get("protein") or {}
    score = row.get("score")
    direct = int(record.get("direct_interaction_count") or 0)
    similar = len(record.get("similar_known_compounds") or [])
    evidence_nodes = int(record.get("evidence_node_count") or 0)
    if direct:
        level = "direct evidence already exists"
        plain = "This pair has direct interaction evidence in the knowledge graph, so it should be treated as known evidence rather than a purely novel prediction."
    elif similar:
        level = "similar-compound support"
        plain = "No direct interaction edge was found for this exact pair, but similar compounds in the graph have evidence connected to the same CYP450 target."
    else:
        level = "model-only hypothesis"
        plain = "No direct or similar-compound evidence path was found in the queried graph; interpret this as a model-generated hypothesis requiring validation."
    return {
        "stage": stage_name or str(row.get("stage") or ""),
        "model": str(row.get("model") or ""),
        "compound": compound.get("name") or compound.get("cid") or row.get("compound_node_ref"),
        "target": protein.get("name") or protein.get("protein_id") or protein.get("accession") or row.get("protein_node_ref"),
        "score": None if pd.isna(score) else float(score),
        "evidence_level": level,
        "plain_language_explanation": plain,
        "evidence_counts": {"direct_interactions": direct, "similar_known_compounds": similar, "evidence_nodes": evidence_nodes},
        "compound_properties": compound,
        "protein_properties": protein,
        "similar_known_compounds": record.get("similar_known_compounds") or [],
    }


def summarize_feature_explanations(model_output_dir: Path | None) -> dict[str, Any]:
    """Collect stage-suitable explanation artifacts without making training fail.

    - Stage 1: SHAP can be applied offline to the saved sklearn model, but the
      reliable default artifact is feature importance/coefficient ranking.
    - Stage 2: KGE models are explained by rank score, relation type, and graph
      evidence paths because embedding dimensions are latent.
    - Stage 3: Captum/PyG Explainer/GNNExplainer can be run from the saved GNN
      checkpoint. The orchestrator reports whether the needed files exist so a
      heavy explainer job can be run separately if desired.
    """
    if model_output_dir is None:
        return {}
    out: dict[str, Any] = {"model_output_dir": str(model_output_dir)}
    fi = model_output_dir / "feature_importance.csv"
    if fi.exists():
        top = read_table(fi).head(25)
        out["feature_explanation_method"] = "sklearn feature_importances_ or absolute coefficients"
        out["feature_importance_file"] = str(fi)
        out["top_features"] = top.to_dict(orient="records")
    history = model_output_dir / "training_history.csv"
    if history.exists():
        out["training_history_file"] = str(history)
    ckpt = model_output_dir / "best_model.pt"
    if ckpt.exists():
        out["gnn_or_kge_checkpoint"] = str(ckpt)
        out["deep_explainer_note"] = "Checkpoint available. For Stage 3, PyG Explainer/GNNExplainer/PGExplainer can be run as a heavier follow-up job using this checkpoint and the HeteroData export."
    return out


def render_html(explanations: list[dict[str, Any]], extra: dict[str, Any]) -> str:
    cards = []
    for ex in explanations:
        counts = "".join(f"<li><b>{html.escape(str(k))}</b>: {v}</li>" for k, v in ex["evidence_counts"].items())
        score = ex.get("score")
        score_text = "" if score is None else f"{float(score):.4f}"
        cards.append(f"""
        <section class="card"><h2>{html.escape(str(ex['compound']))} → {html.escape(str(ex['target']))}</h2>
        <div class="badge">{html.escape(str(ex['evidence_level']))}</div>
        <p><b>Stage:</b> {html.escape(str(ex.get('stage') or ''))} · <b>Model:</b> {html.escape(str(ex.get('model') or ''))}</p>
        <p><b>Model score:</b> {score_text}</p>
        <p>{html.escape(str(ex['plain_language_explanation']))}</p><h3>Evidence counts</h3><ul>{counts}</ul></section>
        """)
    extra_html = ""
    if extra:
        extra_html = "<section class='card'><h2>Model-level explanation artifacts</h2><pre>" + html.escape(json.dumps(extra, indent=2, ensure_ascii=False)[:12000]) + "</pre></section>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>PRING prediction explanations</title><style>
body {{ font-family: Arial, sans-serif; margin:32px; background:#f8fafc; color:#111827; }} .card {{ background:#fff; border:1px solid #e5e7eb; border-radius:16px; padding:22px; margin-bottom:18px; box-shadow:0 10px 28px rgba(17,24,39,.06); }} .badge {{ display:inline-block; background:#fee2e2; color:#991b1b; padding:6px 10px; border-radius:999px; font-weight:800; font-size:12px; }} pre {{ white-space: pre-wrap; background:#f3f4f6; padding:16px; border-radius:12px; }}</style></head><body><h1>PRING CYP450 prediction explanation report</h1>{extra_html}{''.join(cards)}</body></html>"""


def render_markdown(explanations: list[dict[str, Any]], extra: dict[str, Any]) -> str:
    lines = ["# PRING CYP450 prediction explanation report", ""]
    if extra:
        lines += ["## Model-level explanation artifacts", "", "```json", json.dumps(extra, indent=2, ensure_ascii=False), "```", ""]
    lines.append("## Prediction-level evidence explanations")
    for ex in explanations:
        lines += [
            "",
            f"### {ex.get('compound')} → {ex.get('target')}",
            f"- **Stage**: {ex.get('stage')}",
            f"- **Model**: {ex.get('model')}",
            f"- **Score**: {ex.get('score')}",
            f"- **Evidence level**: {ex.get('evidence_level')}",
            f"- **Plain-language explanation**: {ex.get('plain_language_explanation')}",
        ]
        for k, v in ex.get("evidence_counts", {}).items():
            lines.append(f"- **{k}**: {v}")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = ensure_dir(args.output_dir)
    preds = read_table(args.predictions).head(args.limit)
    extra = summarize_feature_explanations(Path(args.model_output_dir) if args.model_output_dir else None)
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    explanations: list[dict[str, Any]] = []
    try:
        with driver.session(database=args.database) as session:
            for _, row in preds.iterrows():
                params = prediction_row_to_params(row)
                if not any(params.values()):
                    continue
                recs = session.run(EVIDENCE_QUERY, params).data()
                if recs:
                    explanations.append(explain_record(recs[0], row, stage_name=args.stage_name))
    finally:
        driver.close()
    save_json(explanations, out_dir / "prediction_explanations.json")
    save_json(extra, out_dir / "model_level_explanation_artifacts.json")
    (out_dir / "prediction_explanations.html").write_text(render_html(explanations, extra), encoding="utf-8")
    (out_dir / "prediction_explanations.md").write_text(render_markdown(explanations, extra), encoding="utf-8")
    summary = {"stage": "Stage 4 — Explainability", "status": "written", "stage_name": args.stage_name, "explanations_written": len(explanations), "output_dir": str(out_dir)}
    save_json(summary, out_dir / "metrics.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate model-level and Neo4j evidence-path explanations for predicted compound-CYP450 interactions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--neo4j-uri", default="bolt://neo4j:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="cyp450kg")
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model-output-dir", default=None)
    parser.add_argument("--stage-name", default="")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
