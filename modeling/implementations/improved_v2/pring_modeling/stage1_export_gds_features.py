from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable, Any

import numpy as np
import pandas as pd
from neo4j import GraphDatabase

from .common import STAGE1, ensure_dir, read_table, resolve_stage_dir, save_json
from .stage1_tabular import find_training_file, find_candidate_file
from .logging_utils import get_logger

logger = get_logger(__name__)

STRUCTURAL_NAME_HINTS = (
    "fastrp", "graphsage", "embedding", "emb", "pagerank", "page_rank", "degree",
    "centrality", "louvain", "community", "wcc", "triangle", "article_rank",
)

DEFAULT_METADATA_COLUMNS = [
    "compound_node_id", "compound_node_ref", "protein_node_id", "protein_node_ref",
    "label", "split", "split_group", "split_strategy", "stage_use", "target_relation",
    "negative_source", "candidate_sampling_method",
]


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) and math.isfinite(float(x))


def _is_numeric_vector(x: Any, min_len: int = 4) -> bool:
    if not isinstance(x, (list, tuple)) or len(x) < min_len:
        return False
    sample = x[: min(10, len(x))]
    return all(_is_number(v) for v in sample)


def _has_structural_hint(name: str) -> bool:
    lower = name.lower()
    return any(h in lower for h in STRUCTURAL_NAME_HINTS)


def _split_csv_arg(value: str | None) -> list[str]:
    if not value or value.lower() == "auto":
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _neo4j_config(args: argparse.Namespace) -> tuple[str, str, str, str]:
    uri = args.neo4j_uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = args.neo4j_user or os.environ.get("NEO4J_USER", "neo4j")
    password = args.neo4j_password or os.environ.get("NEO4J_PASSWORD", "cyp450kg")
    database = args.neo4j_database or os.environ.get("NEO4J_DATABASE", "neo4j")
    return uri, user, password, database


def _fetch_node_props(driver, database: str, node_ids: list[int], batch_size: int = 5000) -> dict[int, dict]:
    out: dict[int, dict] = {}
    if not node_ids:
        return out
    node_ids = sorted(set(int(x) for x in node_ids if pd.notna(x)))
    query = """
    UNWIND $ids AS node_id
    MATCH (n)
    WHERE id(n) = node_id
    RETURN node_id AS node_id, labels(n) AS labels, properties(n) AS props
    """
    with driver.session(database=database) as session:
        for start in range(0, len(node_ids), batch_size):
            ids = node_ids[start : start + batch_size]
            records = session.run(query, ids=ids)
            for rec in records:
                props = dict(rec["props"] or {})
                props["__labels__"] = list(rec["labels"] or [])
                out[int(rec["node_id"])] = props
            logger.info("Fetched Neo4j properties for {} / {} nodes", min(start + batch_size, len(node_ids)), len(node_ids))
    return out


def _infer_properties(prop_maps: Iterable[dict], user_embedding_props: list[str], user_scalar_props: list[str]) -> tuple[list[str], list[str], dict]:
    if user_embedding_props or user_scalar_props:
        return user_embedding_props, user_scalar_props, {"mode": "user_supplied"}
    embedding_props: set[str] = set()
    scalar_props: set[str] = set()
    inspected = 0
    for props in prop_maps:
        inspected += 1
        for k, v in props.items():
            if k.startswith("__"):
                continue
            if _is_numeric_vector(v) and _has_structural_hint(k):
                embedding_props.add(k)
            elif _is_number(v) and _has_structural_hint(k):
                scalar_props.add(k)
    return sorted(embedding_props), sorted(scalar_props), {
        "mode": "auto",
        "inspected_nodes": inspected,
        "embedding_props": sorted(embedding_props),
        "scalar_props": sorted(scalar_props),
    }


def _vector_matrix(ids: pd.Series, prop_maps: dict[int, dict], prop: str, dim: int) -> np.ndarray:
    mat = np.full((len(ids), dim), np.nan, dtype=np.float32)
    for i, node_id in enumerate(ids.astype("int64", copy=False).to_numpy()):
        vec = prop_maps.get(int(node_id), {}).get(prop)
        if isinstance(vec, (list, tuple)) and len(vec) == dim:
            try:
                mat[i, :] = np.asarray(vec, dtype=np.float32)
            except Exception:
                pass
    return mat


def _safe_nan_stats(arr: np.ndarray, axis: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.errstate(all="ignore"):
        mean = np.nanmean(arr, axis=axis)
        maxv = np.nanmax(arr, axis=axis)
        minv = np.nanmin(arr, axis=axis)
    return mean.astype("float32"), maxv.astype("float32"), minv.astype("float32")


def _build_features_chunk(
    df: pd.DataFrame,
    compound_props: dict[int, dict],
    protein_props: dict[int, dict],
    embedding_props: list[str],
    scalar_props: list[str],
    compound_id_col: str,
    protein_id_col: str,
) -> pd.DataFrame:
    out_cols = [c for c in DEFAULT_METADATA_COLUMNS if c in df.columns]
    out = df[out_cols].copy()

    cids = pd.to_numeric(df[compound_id_col], errors="coerce").fillna(-1).astype("int64")
    pids = pd.to_numeric(df[protein_id_col], errors="coerce").fillna(-1).astype("int64")

    # Structural scalar node properties, if available.
    for prop in scalar_props:
        safe = prop.lower().replace(" ", "_").replace("-", "_")
        cvals = np.array([compound_props.get(int(i), {}).get(prop, np.nan) for i in cids], dtype="float32")
        pvals = np.array([protein_props.get(int(i), {}).get(prop, np.nan) for i in pids], dtype="float32")
        out[f"gds_compound_{safe}"] = cvals
        out[f"gds_protein_{safe}"] = pvals
        out[f"gds_pair_{safe}_sum"] = cvals + pvals
        out[f"gds_pair_{safe}_absdiff"] = np.abs(cvals - pvals)

    # Same-space structural embedding pair features.
    for prop in embedding_props:
        # Determine dimension from first valid vector in either map.
        dim = None
        for props in list(compound_props.values())[:1000] + list(protein_props.values())[:1000]:
            v = props.get(prop)
            if _is_numeric_vector(v):
                dim = len(v)
                break
        if not dim:
            continue
        safe = prop.lower().replace(" ", "_").replace("-", "_")
        C = _vector_matrix(cids, compound_props, prop, dim)
        P = _vector_matrix(pids, protein_props, prop, dim)
        valid = np.isfinite(C).all(axis=1) & np.isfinite(P).all(axis=1)
        dot = np.full(len(df), np.nan, dtype="float32")
        cosine = np.full(len(df), np.nan, dtype="float32")
        l2 = np.full(len(df), np.nan, dtype="float32")
        abs_mean = np.full(len(df), np.nan, dtype="float32")
        abs_max = np.full(len(df), np.nan, dtype="float32")
        if valid.any():
            Cv = C[valid]
            Pv = P[valid]
            dot_v = np.sum(Cv * Pv, axis=1)
            denom = np.linalg.norm(Cv, axis=1) * np.linalg.norm(Pv, axis=1)
            cosine_v = np.divide(dot_v, denom, out=np.zeros_like(dot_v), where=denom != 0)
            diff = Cv - Pv
            absdiff = np.abs(diff)
            dot[valid] = dot_v.astype("float32")
            cosine[valid] = cosine_v.astype("float32")
            l2[valid] = np.linalg.norm(diff, axis=1).astype("float32")
            abs_mean[valid], abs_max[valid], _ = _safe_nan_stats(absdiff, axis=1)
        out[f"gds_{safe}_dot"] = dot
        out[f"gds_{safe}_cosine"] = cosine
        out[f"gds_{safe}_l2"] = l2
        out[f"gds_{safe}_absdiff_mean"] = abs_mean
        out[f"gds_{safe}_absdiff_max"] = abs_max
    return out


def _write_features_csv(
    input_file: Path,
    output_file: Path,
    compound_props: dict[int, dict],
    protein_props: dict[int, dict],
    embedding_props: list[str],
    scalar_props: list[str],
    args: argparse.Namespace,
    max_rows: int = 0,
) -> dict:
    ensure_dir(output_file.parent)
    first = True
    rows = 0
    chunksize = max(1000, int(args.chunk_size))
    reader = pd.read_csv(input_file, chunksize=chunksize, nrows=max_rows if max_rows and max_rows > 0 else None)
    for chunk in reader:
        if args.compound_id_column not in chunk.columns or args.protein_id_column not in chunk.columns:
            raise KeyError(f"Expected columns {args.compound_id_column!r} and {args.protein_id_column!r} in {input_file}")
        feat = _build_features_chunk(
            chunk,
            compound_props=compound_props,
            protein_props=protein_props,
            embedding_props=embedding_props,
            scalar_props=scalar_props,
            compound_id_col=args.compound_id_column,
            protein_id_col=args.protein_id_column,
        )
        feat.to_csv(output_file, index=False, mode="w" if first else "a", header=first)
        first = False
        rows += len(feat)
        logger.info("Wrote {} rows to {}", rows, output_file)
    return {"file": str(output_file), "rows": int(rows), "feature_columns": [c for c in pd.read_csv(output_file, nrows=0).columns if c not in DEFAULT_METADATA_COLUMNS] if output_file.exists() else []}


def _collect_unique_ids(files: list[Path], args: argparse.Namespace, max_candidate_rows: int) -> tuple[list[int], list[int]]:
    compound_ids: set[int] = set()
    protein_ids: set[int] = set()
    for path in files:
        if not path or not path.exists():
            continue
        nrows = max_candidate_rows if ("candidate" in path.name.lower() and max_candidate_rows and max_candidate_rows > 0) else None
        for chunk in pd.read_csv(path, usecols=[args.compound_id_column, args.protein_id_column], chunksize=max(1000, args.chunk_size), nrows=nrows):
            compound_ids.update(pd.to_numeric(chunk[args.compound_id_column], errors="coerce").dropna().astype("int64").tolist())
            protein_ids.update(pd.to_numeric(chunk[args.protein_id_column], errors="coerce").dropna().astype("int64").tolist())
            if args.max_unique_nodes and len(compound_ids) + len(protein_ids) >= args.max_unique_nodes:
                logger.warning("Reached --max-unique-nodes; feature export will be partial.")
                return sorted(compound_ids), sorted(protein_ids)
    return sorted(compound_ids), sorted(protein_ids)


def run(args: argparse.Namespace) -> dict:
    resolved = resolve_stage_dir(args.modeling_dir, STAGE1)
    stage_dir = resolved.root
    try:
        train_in = Path(args.training_input) if args.training_input else find_training_file(stage_dir)
        cand_in = Path(args.candidate_input) if args.candidate_input else find_candidate_file(stage_dir)
        train_out = Path(args.training_output) if args.training_output else stage_dir / "compound_target_training_pairs_gds_features.csv"
        cand_out = Path(args.candidate_output) if args.candidate_output else stage_dir / "candidate_pairs_gds_features.csv"

        uri, user, password, database = _neo4j_config(args)
        logger.info("Connecting to Neo4j {} database {}", uri, database)
        driver = GraphDatabase.driver(uri, auth=(user, password))

        files_for_ids = [train_in]
        if args.include_candidates and cand_in is not None:
            files_for_ids.append(cand_in)
        compound_ids, protein_ids = _collect_unique_ids(files_for_ids, args, args.max_candidate_rows)
        logger.info("Unique compound nodes: {}", len(compound_ids))
        logger.info("Unique protein nodes: {}", len(protein_ids))
        compound_props = _fetch_node_props(driver, database, compound_ids, batch_size=args.neo4j_batch_size)
        protein_props = _fetch_node_props(driver, database, protein_ids, batch_size=args.neo4j_batch_size)
        driver.close()

        embedding_props, scalar_props, inference = _infer_properties(
            list(compound_props.values()) + list(protein_props.values()),
            _split_csv_arg(args.embedding_props),
            _split_csv_arg(args.scalar_props),
        )
        logger.info("Embedding properties selected: {}", embedding_props)
        logger.info("Scalar properties selected: {}", scalar_props)

        train_summary = _write_features_csv(train_in, train_out, compound_props, protein_props, embedding_props, scalar_props, args, max_rows=args.max_training_rows)
        cand_summary = None
        if args.include_candidates and cand_in is not None:
            cand_summary = _write_features_csv(cand_in, cand_out, compound_props, protein_props, embedding_props, scalar_props, args, max_rows=args.max_candidate_rows)

        summary = {
            "stage": "Stage 1 — GDS structural feature export",
            "status": "exported",
            "stage_dir": str(stage_dir),
            "training_input": str(train_in),
            "candidate_input": str(cand_in) if cand_in else None,
            "training_output": str(train_out),
            "candidate_output": str(cand_out) if cand_summary else None,
            "embedding_props": embedding_props,
            "scalar_props": scalar_props,
            "property_inference": inference,
            "training_export": train_summary,
            "candidate_export": cand_summary,
            "note": "Use the generated *_gds_features.csv files with stage1_tabular --feature-policy leakage_safe.",
        }
        save_json(summary, train_out.parent / "stage1_gds_feature_export_summary.json")
        return summary
    finally:
        resolved.cleanup()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export leakage-safe Stage 1 pair features from Neo4j GDS node embeddings/properties.")
    p.add_argument("--modeling-dir", required=True, help="Full modeling folder, standalone Stage 1 folder, or zip.")
    p.add_argument("--training-input", default=None)
    p.add_argument("--candidate-input", default=None)
    p.add_argument("--training-output", default=None)
    p.add_argument("--candidate-output", default=None)
    p.add_argument("--include-candidates", action="store_true", help="Also export candidate pair features for scoring.")
    p.add_argument("--max-training-rows", type=int, default=0, help="0 means all rows.")
    p.add_argument("--max-candidate-rows", type=int, default=100000, help="0 means all rows; start with 100k for tests.")
    p.add_argument("--chunk-size", type=int, default=50000)
    p.add_argument("--neo4j-batch-size", type=int, default=5000)
    p.add_argument("--max-unique-nodes", type=int, default=0)
    p.add_argument("--compound-id-column", default="compound_node_id")
    p.add_argument("--protein-id-column", default="protein_node_id")
    p.add_argument("--embedding-props", default="auto", help="Comma-separated embedding node properties or auto.")
    p.add_argument("--scalar-props", default="auto", help="Comma-separated scalar node properties or auto.")
    p.add_argument("--neo4j-uri", default=None)
    p.add_argument("--neo4j-user", default=None)
    p.add_argument("--neo4j-password", default=None)
    p.add_argument("--neo4j-database", default=None)
    return p


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
