#!/usr/bin/env python3
"""Create the outcome-safe Neo4j GDS graph used by the HPC Stage 1 baseline.

The generated PRING Cypher examples are useful for exploratory GDS work, but a
final evaluation must not project held-out Interaction/BioAssay/Endpoint
evidence.  This helper projects only:

* positive compound-target relationships from the registered training split;
* label-independent ``SIMILAR_TO`` compound relationships; and
* Compound and Protein nodes.

The resulting FastRP representation is outcome-safe but transductive with
respect to node presence and the similarity graph.  Reports must not describe
it as a strictly inductive cold-compound representation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from neo4j import GraphDatabase


@dataclass(frozen=True)
class Pair:
    compound_id: int
    protein_id: str
    compound_node_id: int
    protein_node_id: int


def _parse_compound_ref(value: str) -> int:
    marker = "cid="
    if marker not in value:
        raise ValueError(f"Unsupported compound reference: {value!r}")
    return int(value.rsplit(marker, 1)[1])


def _parse_protein_ref(value: str) -> str:
    marker = "protein_id="
    if marker not in value:
        raise ValueError(f"Unsupported protein reference: {value!r}")
    result = value.rsplit(marker, 1)[1].strip()
    if not result:
        raise ValueError(f"Empty protein identifier in {value!r}")
    return result


def _normalise_split(value: str) -> str:
    value = value.strip().lower()
    return "validation" if value in {"valid", "val"} else value


def _read_pairs(path: Path) -> tuple[list[Pair], dict[tuple[int, str], Pair]]:
    training_positive: list[Pair] = []
    all_pairs: dict[tuple[int, str], Pair] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "compound_node_ref",
            "protein_node_ref",
            "compound_node_id",
            "protein_node_id",
            "label",
            "split",
        }
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing pair columns: {', '.join(missing)}")

        for row in reader:
            pair = Pair(
                compound_id=_parse_compound_ref(row["compound_node_ref"]),
                protein_id=_parse_protein_ref(row["protein_node_ref"]),
                compound_node_id=int(row["compound_node_id"]),
                protein_node_id=int(row["protein_node_id"]),
            )
            key = (pair.compound_id, pair.protein_id)
            previous = all_pairs.setdefault(key, pair)
            if previous != pair:
                raise ValueError(f"Inconsistent node mapping for pair {key}")
            if (
                _normalise_split(row["split"]) == "train"
                and str(row["label"]).strip() in {"1", "1.0", "true", "True"}
            ):
                training_positive.append(pair)
    if not training_positive:
        raise ValueError("No positive pairs were found in the registered training split.")
    return training_positive, all_pairs


def _chunks(values: list[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _single(session, query: str, **parameters: Any) -> dict[str, Any]:
    record = session.run(query, **parameters).single()
    if record is None:
        return {}
    return dict(record)


def _verify_node_id_parity(
    session, pairs: dict[tuple[int, str], Pair], batch_size: int
) -> dict[str, int]:
    rows = [
        {
            "cid": pair.compound_id,
            "protein_id": pair.protein_id,
            "expected_compound_node_id": pair.compound_node_id,
            "expected_protein_node_id": pair.protein_node_id,
        }
        for pair in pairs.values()
    ]
    found = 0
    mismatched = 0
    for batch in _chunks(rows, batch_size):
        result = session.run(
            """
            UNWIND $rows AS row
            MATCH (c:Compound {cid: row.cid})
            MATCH (p:Protein {protein_id: row.protein_id})
            RETURN row.expected_compound_node_id AS expected_c,
                   row.expected_protein_node_id AS expected_p,
                   id(c) AS actual_c,
                   id(p) AS actual_p
            """,
            rows=batch,
        )
        for record in result:
            found += 1
            if (
                int(record["expected_c"]) != int(record["actual_c"])
                or int(record["expected_p"]) != int(record["actual_p"])
            ):
                mismatched += 1
        result.consume()

    expected = len(rows)
    missing = expected - found
    if missing or mismatched:
        raise RuntimeError(
            "Neo4j/package node-ID parity failed: "
            f"expected={expected}, found={found}, missing={missing}, "
            f"mismatched={mismatched}. Use a fresh database loaded only from "
            "this exact prepared run."
        )
    return {
        "expected_pairs": expected,
        "found_pairs": found,
        "missing_pairs": missing,
        "mismatched_pairs": mismatched,
    }


def _write_training_relationships(
    session,
    pairs: list[Pair],
    *,
    batch_size: int,
    dataset_id: str,
) -> dict[str, int]:
    session.run("MATCH ()-[r:PRING_TRAIN_POSITIVE]->() DELETE r").consume()
    unique = sorted({(pair.compound_id, pair.protein_id) for pair in pairs})
    matched = 0
    for batch_values in _chunks(
        [{"cid": cid, "protein_id": protein_id} for cid, protein_id in unique],
        batch_size,
    ):
        record = _single(
            session,
            """
            UNWIND $rows AS row
            MATCH (c:Compound {cid: row.cid})
            MATCH (p:Protein {protein_id: row.protein_id})
            MERGE (c)-[r:PRING_TRAIN_POSITIVE]->(p)
            SET r.source = 'PRING registered training split',
                r.dataset_id = $dataset_id,
                r.label = 1
            RETURN count(*) AS matched
            """,
            rows=batch_values,
            dataset_id=dataset_id,
        )
        matched += int(record.get("matched", 0))
    if matched != len(unique):
        raise RuntimeError(
            f"Only {matched} of {len(unique)} training-positive pairs matched Neo4j."
        )
    return {"unique_training_positive_pairs": len(unique), "matched_pairs": matched}


def _relationship_count(session, relationship_type: str) -> int:
    record = _single(
        session,
        f"MATCH ()-[r:`{relationship_type}`]->() RETURN count(r) AS n",
    )
    return int(record.get("n", 0))


def run(args: argparse.Namespace) -> dict[str, Any]:
    pair_file = args.pair_file.expanduser().resolve()
    manifest_path = args.provenance_manifest.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    if not pair_file.is_file():
        raise FileNotFoundError(pair_file)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    dataset_id = str(manifest.get("dataset_id") or "")
    split_registry_id = str(manifest.get("split_registry_id") or "")
    if not dataset_id or not split_registry_id:
        raise ValueError(
            "The modeling manifest must contain dataset_id and split_registry_id."
        )

    training_positive, all_pairs = _read_pairs(pair_file)
    uri = args.uri or os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    user = args.user or os.environ.get("NEO4J_USER", "neo4j")
    password = args.password or os.environ.get("NEO4J_PASSWORD")
    database = args.database or os.environ.get("NEO4J_DATABASE", "neo4j")
    if not password:
        raise ValueError("NEO4J_PASSWORD is required.")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            component = _single(
                session,
                """
                CALL dbms.components()
                YIELD name, versions, edition
                RETURN name, versions, edition
                LIMIT 1
                """,
            )
            gds_version = _single(
                session, "RETURN gds.version() AS version"
            ).get("version")
            parity = _verify_node_id_parity(session, all_pairs, args.batch_size)
            training_relations = _write_training_relationships(
                session,
                training_positive,
                batch_size=args.batch_size,
                dataset_id=dataset_id,
            )

            exists = bool(
                _single(
                    session,
                    "CALL gds.graph.exists($name) YIELD exists RETURN exists",
                    name=args.graph_name,
                ).get("exists")
            )
            if exists:
                session.run(
                    "CALL gds.graph.drop($name) YIELD graphName RETURN graphName",
                    name=args.graph_name,
                ).consume()

            relationship_projection: dict[str, dict[str, str]] = {
                "PRING_TRAIN_POSITIVE": {"orientation": "UNDIRECTED"}
            }
            similarity_count = _relationship_count(session, "SIMILAR_TO")
            if similarity_count:
                relationship_projection["SIMILAR_TO"] = {
                    "orientation": "UNDIRECTED"
                }

            projection = _single(
                session,
                """
                CALL gds.graph.project(
                  $name,
                  ['Compound', 'Protein'],
                  $relationships
                )
                YIELD graphName, nodeCount, relationshipCount
                RETURN graphName, nodeCount, relationshipCount
                """,
                name=args.graph_name,
                relationships=relationship_projection,
            )
            embedding = _single(
                session,
                """
                CALL gds.fastRP.write(
                  $name,
                  {
                    embeddingDimension: $dimension,
                    iterationWeights: [0.0, 1.0, 1.0, 1.0],
                    randomSeed: $seed,
                    writeProperty: $property
                  }
                )
                YIELD nodePropertiesWritten, computeMillis
                RETURN nodePropertiesWritten, computeMillis
                """,
                name=args.graph_name,
                dimension=args.embedding_dimension,
                seed=args.seed,
                property=args.write_property,
            )
    finally:
        driver.close()

    report = {
        "format": "pring-stage1-outcome-safe-gds-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "split_registry_id": split_registry_id,
        "pair_file": str(pair_file),
        "neo4j": {
            "uri": uri,
            "database": database,
            "component": component,
            "gds_version": gds_version,
        },
        "node_id_parity": parity,
        "training_relationships": training_relations,
        "projection": projection,
        "embedding": embedding,
        "embedding_property": args.write_property,
        "embedding_dimension": args.embedding_dimension,
        "random_seed": args.seed,
        "graph_scope": "transductive_node_set_outcome_safe",
        "included_relationship_types": sorted(relationship_projection),
        "excluded_outcome_layers": [
            "Interaction",
            "BioAssay",
            "Endpoint",
            "MeasureGrp",
            "ASSERTS_CHEMICAL",
            "ASSERTS_TARGET",
            "SUPPORTED_BY_ASSAY",
            "SUPPORTED_BY_ENDPOINT",
        ],
        "scientific_note": (
            "FastRP used only registered training-positive outcomes plus "
            "label-independent compound similarity. Validation/test outcome "
            "evidence was not projected. Node presence remains transductive."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare an outcome-safe FastRP graph for PRING Stage 1."
    )
    parser.add_argument("--pair-file", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--graph-name", default="pring_stage1_outcome_safe")
    parser.add_argument("--write-property", default="pringFastRP")
    parser.add_argument("--embedding-dimension", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--uri", default=None)
    parser.add_argument("--user", default=None)
    parser.add_argument(
        "--password",
        default=None,
        help="Prefer NEO4J_PASSWORD so the secret is not exposed in process arguments.",
    )
    parser.add_argument("--database", default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
