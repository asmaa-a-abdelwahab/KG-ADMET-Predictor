from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from .common import parse_node_ref, read_table


def _row_to_payload(row: pd.Series, model_name: str) -> dict[str, Any] | None:
    compound_ref = str(row.get("compound_node_ref") or "").strip()
    protein_ref = str(row.get("protein_node_ref") or "").strip()
    if not compound_ref or not protein_ref or compound_ref.lower() == "nan" or protein_ref.lower() == "nan":
        return None
    c = parse_node_ref(compound_ref)
    p = parse_node_ref(protein_ref)
    raw_score = row.get("raw_score", row.get("score_raw", None))
    score = row.get("score", row.get("prediction", row.get("probability", None)))
    if pd.isna(score):
        return None
    return {
        "compound_node_ref": compound_ref,
        "protein_node_ref": protein_ref,
        "cid": str(c.get("cid") or c.get("id") or row.get("cid") or row.get("compound_id") or "").strip(),
        "protein_id": str(p.get("protein_id") or p.get("accession") or p.get("id") or row.get("protein_id") or row.get("target_id") or "").strip(),
        "score": float(score),
        "raw_score": None if raw_score is None or pd.isna(raw_score) else float(raw_score),
        "predicted_label": int(row.get("predicted_label", float(score) >= float(os.getenv("MODEL_THRESHOLD", "0.5")))),
        "model": str(row.get("model", model_name)),
        "stage": str(row.get("stage", "")),
        "source": str(row.get("source", "modeling")),
    }


def export_predictions_dataframe(predictions: pd.DataFrame, *, model_name: str, max_rows: int = 50000) -> dict[str, Any]:
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "cyp450kg")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    rows = []
    for _, row in predictions.head(max_rows).iterrows():
        payload = _row_to_payload(row, model_name)
        if payload and (payload["cid"] or payload["protein_id"]):
            rows.append(payload)
    if not rows:
        return {"exported": 0, "reason": "no valid rows with compound_node_ref/protein_node_ref and score"}

    cypher = """
    UNWIND $rows AS row
    MATCH (c:Compound)
    WHERE (row.cid <> '' AND toString(c.cid) = row.cid)
       OR (exists(c.node_ref) AND c.node_ref = row.compound_node_ref)
    MATCH (p:Protein)
    WHERE (row.protein_id <> '' AND (toString(p.protein_id) = row.protein_id OR toString(p.accession) = row.protein_id))
       OR (exists(p.node_ref) AND p.node_ref = row.protein_node_ref)
    MERGE (c)-[r:PREDICTED_INTERACTION {model: row.model, compound_node_ref: row.compound_node_ref, protein_node_ref: row.protein_node_ref}]->(p)
    SET r.score = row.score,
        r.raw_score = row.raw_score,
        r.predicted_label = row.predicted_label,
        r.stage = row.stage,
        r.source = row.source,
        r.updated_at = datetime()
    """
    # Neo4j 5 supports exists() on properties for compatibility, but the modern syntax is p.prop IS NOT NULL.
    cypher = cypher.replace("exists(c.node_ref)", "c.node_ref IS NOT NULL").replace("exists(p.node_ref)", "p.node_ref IS NOT NULL")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            summary = session.execute_write(lambda tx: tx.run(cypher, rows=rows).consume())
    finally:
        driver.close()
    return {
        "exported_attempted": len(rows),
        "neo4j_uri": uri,
        "database": database,
        "relationships_created": summary.counters.relationships_created,
        "properties_set": summary.counters.properties_set,
    }


def export_predictions_file(path: str | Path, *, model_name: str | None = None, max_rows: int = 50000) -> dict[str, Any]:
    p = Path(path)
    df = read_table(p)
    return export_predictions_dataframe(df, model_name=model_name or p.parent.name, max_rows=max_rows)
