from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from .common import parse_node_ref, read_table


def _row_to_payload(row: pd.Series, model_name: str) -> dict[str, Any] | None:
    compound_ref = str(row.get("compound_node_ref") or "").strip()
    protein_ref = str(row.get("protein_node_ref") or "").strip()
    c = parse_node_ref(compound_ref)
    p = parse_node_ref(protein_ref)
    cid = str(c.get("cid") or c.get("id") or row.get("cid") or row.get("compound_id") or row.get("compound_entity_id") or "").strip()
    protein_id = str(p.get("protein_id") or p.get("accession") or p.get("id") or row.get("protein_id") or row.get("accession") or row.get("target_id") or row.get("protein_entity_id") or "").strip()
    score = row.get("score", row.get("prediction", row.get("probability", None)))
    if score is None or pd.isna(score):
        return None
    raw_score = row.get("raw_score", row.get("score_raw", None))
    try:
        score_f = float(score)
    except Exception:
        return None
    return {
        "compound_node_ref": compound_ref,
        "protein_node_ref": protein_ref,
        "cid": cid,
        "protein_id": protein_id,
        "score": score_f,
        "raw_score": None if raw_score is None or pd.isna(raw_score) else float(raw_score),
        "predicted_label": int(row.get("predicted_label", score_f >= float(os.getenv("MODEL_THRESHOLD", "0.5")))),
        "model": str(row.get("model", model_name)),
        "stage": str(row.get("stage", "")),
        "source": str(row.get("source", "pring_modeling")),
    }


def _chunks(rows: list[dict[str, Any]], size: int):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def export_predictions_dataframe(predictions: pd.DataFrame, *, model_name: str, max_rows: int = 50000, batch_size: int | None = None) -> dict[str, Any]:
    """Export prediction rows back to Neo4j as PREDICTED_INTERACTION relationships.

    The export is idempotent for the same model/compound/protein reference and
    writes only rows that can be linked through cid/protein_id or PRING node_ref.
    """
    from neo4j import GraphDatabase

    uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "cyp450kg")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    batch_size = batch_size or int(os.getenv("MODEL_NEO4J_EXPORT_BATCH_SIZE", "1000"))

    rows: list[dict[str, Any]] = []
    limit_df = predictions.head(max_rows) if max_rows and max_rows > 0 else predictions
    for _, row in limit_df.iterrows():
        payload = _row_to_payload(row, model_name)
        if payload and (payload["cid"] or payload["protein_id"] or payload["compound_node_ref"] or payload["protein_node_ref"]):
            rows.append(payload)
    if not rows:
        return {"exported_attempted": 0, "relationships_created": 0, "relationships_matched": 0, "reason": "no valid rows with compound/protein identifiers and score"}

    cypher = """
    UNWIND $rows AS row
    MATCH (c:Compound)
    WHERE (row.cid <> '' AND toString(c.cid) = row.cid)
       OR (row.compound_node_ref <> '' AND c.node_ref IS NOT NULL AND c.node_ref = row.compound_node_ref)
    MATCH (p:Protein)
    WHERE (row.protein_id <> '' AND (toString(p.protein_id) = row.protein_id OR toString(p.accession) = row.protein_id))
       OR (row.protein_node_ref <> '' AND p.node_ref IS NOT NULL AND p.node_ref = row.protein_node_ref)
    MERGE (c)-[r:PREDICTED_INTERACTION {model: row.model, compound_node_ref: row.compound_node_ref, protein_node_ref: row.protein_node_ref}]->(p)
    SET r.score = row.score,
        r.raw_score = row.raw_score,
        r.predicted_label = row.predicted_label,
        r.stage = row.stage,
        r.source = row.source,
        r.updated_at = datetime()
    RETURN count(r) AS written
    """

    total_written = 0
    created = 0
    properties_set = 0
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            for batch in _chunks(rows, batch_size):
                result = session.run(cypher, rows=batch)
                data = result.single()
                total_written += int(data["written"] if data else 0)
                summary = result.consume()
                created += summary.counters.relationships_created
                properties_set += summary.counters.properties_set
    finally:
        driver.close()
    return {
        "exported_attempted": len(rows),
        "relationships_written_or_updated": total_written,
        "relationships_created": created,
        "relationships_matched_or_updated": max(0, total_written - created),
        "properties_set": properties_set,
        "neo4j_uri": uri,
        "database": database,
    }


def export_predictions_file(path: str | Path, *, model_name: str | None = None, max_rows: int = 50000) -> dict[str, Any]:
    p = Path(path)
    df = read_table(p)
    return export_predictions_dataframe(df, model_name=model_name or p.parent.name, max_rows=max_rows)
