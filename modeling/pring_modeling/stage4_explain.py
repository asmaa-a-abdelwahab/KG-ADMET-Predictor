from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from neo4j import GraphDatabase

from .common import ensure_dir, parse_node_ref, pick_col, read_table, save_json

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


def explain_record(record: dict[str, Any], row: pd.Series) -> dict[str, Any]:
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


def render_html(explanations: list[dict[str, Any]]) -> str:
    cards = []
    for ex in explanations:
        counts = "".join(f"<li><b>{html.escape(k)}</b>: {v}</li>" for k, v in ex["evidence_counts"].items())
        score_text = "" if ex.get("score") is None else f"{ex.get("score"):.4f}"
        cards.append(f"""
        <section class="card"><h2>{html.escape(str(ex['compound']))} → {html.escape(str(ex['target']))}</h2>
        <div class="badge">{html.escape(ex['evidence_level'])}</div>
        <p><b>Model score:</b> {score_text}</p>
        <p>{html.escape(ex['plain_language_explanation'])}</p><h3>Evidence counts</h3><ul>{counts}</ul></section>
        """)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>PRING prediction explanations</title><style>
body {{ font-family: Arial, sans-serif; margin:32px; background:#f8fafc; color:#111827; }} .card {{ background:#fff; border:1px solid #e5e7eb; border-radius:16px; padding:22px; margin-bottom:18px; box-shadow:0 10px 28px rgba(17,24,39,.06); }} .badge {{ display:inline-block; background:#fee2e2; color:#991b1b; padding:6px 10px; border-radius:999px; font-weight:800; font-size:12px; }}</style></head><body><h1>PRING CYP450 prediction explanation report</h1>{''.join(cards)}</body></html>"""


def run(args: argparse.Namespace) -> dict:
    out_dir = ensure_dir(args.output_dir)
    preds = read_table(args.predictions).head(args.limit)
    driver = GraphDatabase.driver(args.neo4j_uri, auth=(args.neo4j_user, args.neo4j_password))
    explanations = []
    try:
        with driver.session(database=args.database) as session:
            for _, row in preds.iterrows():
                params = prediction_row_to_params(row)
                if not any(params.values()):
                    continue
                recs = session.run(EVIDENCE_QUERY, params).data()
                if recs:
                    explanations.append(explain_record(recs[0], row))
    finally:
        driver.close()
    save_json(explanations, out_dir / "prediction_explanations.json")
    (out_dir / "prediction_explanations.html").write_text(render_html(explanations), encoding="utf-8")
    summary = {"stage": "Stage 4 — Explainability", "explanations_written": len(explanations), "output_dir": str(out_dir)}
    save_json(summary, out_dir / "metrics.json")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate evidence-path explanations for predicted compound-CYP450 interactions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--neo4j-uri", default="bolt://neo4j:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="cyp450kg")
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=100)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    print(json.dumps(run(build_parser().parse_args(argv)), indent=2))


if __name__ == "__main__":
    main()
