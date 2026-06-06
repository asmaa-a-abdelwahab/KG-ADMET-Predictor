# neo4j_utils.py
from __future__ import annotations

import re
from typing import Any, List, Optional

from neo4j import Driver, GraphDatabase
from utils.config import APP_OPTION_LIMIT, APP_RESULT_LIMIT, NEO4J_DATABASE, logger


class Neo4jConnectionError(Exception):
    """Exception raised when there is a problem connecting to Neo4j."""


class Neo4jBase:
    """Base class for handling connections to a PRING Neo4j database."""

    def __init__(self, uri: str, user: str, password: str, database: str = NEO4J_DATABASE) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None

    def connect_to_neo4j(self) -> None:
        try:
            logger.info("Connecting to Neo4j at %s", self.uri)
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1 AS ok").consume()
            logger.info("Successfully connected to the Neo4j database.")
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", e)
            raise Neo4jConnectionError("Failed to connect to the Neo4j database.") from e

    def close(self) -> None:
        if self.driver:
            logger.info("Closing Neo4j connection")
            self.driver.close()
            logger.info("Neo4j connection closed successfully.")


def _session(driver: Driver):
    return driver.session(database=NEO4J_DATABASE)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _display_value(label: str, identifier: Any | None = None) -> str:
    """Create stable UI display strings while tolerating missing identifiers."""
    label = _as_text(label)
    if identifier in (None, "", [], {}):
        return label
    return f"{label} ({identifier})"


def _display_compound(record: dict[str, Any]) -> str:
    label = _as_text(record.get("label") or record.get("cid") or record.get("element_id"))
    cid = record.get("cid")
    if cid in (None, ""):
        return label
    return f"{label} (CID {cid})"


def _tokenize_selected(values: List[str]) -> List[str]:
    """Return robust matching tokens from Streamlit display strings."""
    tokens: set[str] = set()
    for value in values or []:
        text = _as_text(value)
        if not text:
            continue
        tokens.add(text)

        # Remove trailing display suffix: "Name (ID)" -> "Name" and "ID".
        paren = re.search(r"^(.*?)\s*\(([^()]+)\)\s*$", text)
        if paren:
            if paren.group(1).strip():
                tokens.add(paren.group(1).strip())
            if paren.group(2).strip():
                tokens.add(paren.group(2).strip())

        # PubChem compound display suffix: "Aspirin (CID 2244)" -> "2244".
        cid = re.search(r"\bCID\s*([0-9]+)\b", text, flags=re.IGNORECASE)
        if cid:
            tokens.add(cid.group(1))
    return sorted(tokens)


def _parse_cids(selected_compounds: List[str]) -> List[int]:
    """Extract numeric PubChem CIDs from UI display strings."""
    cids: List[int] = []
    for token in _tokenize_selected(selected_compounds):
        if token.isdigit():
            cids.append(int(token))
    return sorted(set(cids))


def _parse_target_ids(selected_targets: List[str]) -> List[str]:
    """Extract target IDs from UI display strings such as 'CYP3A4 (P08684)'."""
    return sorted(set(_tokenize_selected(selected_targets)))


def _filter_params(compound_names: List[str], gene_symbols: Optional[List[str]] = None) -> dict[str, Any]:
    selected_compounds = [str(x) for x in (compound_names or []) if str(x).strip()]
    selected_targets = [str(x) for x in (gene_symbols or []) if str(x).strip()]
    return {
        "compound_values": selected_compounds,
        "compound_tokens": _tokenize_selected(selected_compounds),
        "compound_cids": _parse_cids(selected_compounds),
        "target_values": selected_targets,
        "target_tokens": _parse_target_ids(selected_targets),
        "target_ids": _parse_target_ids(selected_targets),
        "limit": APP_RESULT_LIMIT,
    }


# Filters are intentionally broad because PRING runs have evolved over time and
# can use slightly different key names depending on the run/materialization step.
# The aliases are used by the query functions below.
COMPOUND_FILTER = """
(
  c.cid IN $compound_cids OR
  toString(c.cid) IN $compound_tokens OR
  coalesce(c.preferred_name, c.name, c.CompoundName, c.title, toString(c.cid)) IN $compound_tokens OR
  (coalesce(c.preferred_name, c.name, c.CompoundName, c.title, toString(c.cid)) + ' (CID ' + toString(c.cid) + ')') IN $compound_values
)
"""

TARGET_FILTER = """
(
  size($target_tokens) = 0 OR
  p.protein_id IN $target_tokens OR
  toString(p.protein_id) IN $target_tokens OR
  p.uniprot_id IN $target_tokens OR
  p.uniprot_accession IN $target_tokens OR
  p.accession IN $target_tokens OR
  p.id IN $target_tokens OR
  p.refseq_accession IN $target_tokens OR
  p.ProteinRefSeqAccession IN $target_tokens OR
  p.cyp_symbol IN $target_tokens OR
  p.gene_symbol IN $target_tokens OR
  p.symbol IN $target_tokens OR
  p.name IN $target_tokens OR
  toString(coalesce(p.cyp_symbol, p.gene_symbol, p.symbol, p.name, p.uniprot_id, p.uniprot_accession, p.accession, p.protein_id, p.id)) IN $target_tokens OR
  (toString(coalesce(p.cyp_symbol, p.gene_symbol, p.symbol, p.name, p.uniprot_id, p.uniprot_accession, p.accession, p.protein_id, p.id)) + ' (' + toString(coalesce(p.protein_id, p.uniprot_id, p.uniprot_accession, p.accession, p.id)) + ')') IN $target_values
)
"""


def create_indexes(driver: Driver) -> None:
    """Keep the Streamlit app read-only with respect to Neo4j schema objects.

    PRING's loader creates the required uniqueness constraints/indexes. Creating
    plain app-side indexes on the same label/property pairs can block PRING from
    creating constraints later, for example Compound(cid). Therefore, the app no
    longer creates Neo4j indexes automatically.
    """
    logger.info("Skipping Streamlit-side index creation; PRING loader owns Neo4j schema constraints/indexes.")


def execute_query(driver: Driver, query: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    """Execute a Cypher query and return fully materialized records."""
    try:
        logger.info("Executing query: %s", query)
        with _session(driver) as session:
            return session.run(query, params or {}).data()
    except Exception as e:
        logger.error("Error executing query: %s", e)
        return []


def get_compound_names(driver: Driver) -> List[str]:
    """Return compound display values, prioritizing compounds with PRING evidence.

    The old implementation listed arbitrary Compound nodes alphabetically. In the
    rematerialized CYP450 graph, many compounds only have similarity links; those
    work for the similarity view but return empty results for assay/interaction
    views. This query lists compounds with Interaction or BioAssay evidence first.
    """
    query = """
    CALL {
      MATCH (c:Compound)<-[:ASSERTS_CHEMICAL]-(:Interaction)
      RETURN c, 3 AS priority
      UNION
      MATCH (c:Compound)<-[:STANDARDIZED_TO]-(:Substance)<-[:ABOUT_SUBSTANCE]-(:Endpoint)
      RETURN c, 2 AS priority
      UNION
      MATCH (c:Compound)-[:SIMILAR_TO]-(:Compound)
      RETURN c, 1 AS priority
    }
    WITH c, max(priority) AS priority, count(*) AS evidence_count
    WITH c, priority, evidence_count,
         toString(coalesce(c.preferred_name, c.name, c.CompoundName, c.title, c.cid, elementId(c))) AS label
    RETURN label, c.cid AS cid, elementId(c) AS element_id, priority, evidence_count
    ORDER BY priority DESC, evidence_count DESC, label
    LIMIT $limit
    """
    with _session(driver) as session:
        result = session.run(query, limit=APP_OPTION_LIMIT).data()
        return [_display_compound(record) for record in result]


def get_gene_symbols(driver: Driver) -> List[str]:
    """Return CYP450 target display values from Protein nodes."""
    query = """
    CALL {
      MATCH (:Interaction)-[:ASSERTS_TARGET]->(p:Protein)
      RETURN p, 2 AS priority
      UNION
      MATCH (:MeasureGrp)-[:TESTED_ON]->(p:Protein)
      RETURN p, 1 AS priority
    }
    WITH p, max(priority) AS priority, count(*) AS evidence_count
    WITH p, priority, evidence_count,
         toString(coalesce(p.cyp_symbol, p.gene_symbol, p.symbol, p.name, p.uniprot_id, p.uniprot_accession, p.accession, p.protein_id, p.id, elementId(p))) AS label,
         coalesce(p.protein_id, p.uniprot_id, p.uniprot_accession, p.accession, p.id) AS identifier
    RETURN label, identifier, priority, evidence_count
    ORDER BY priority DESC, evidence_count DESC, label
    LIMIT $limit
    """
    with _session(driver) as session:
        result = session.run(query, limit=APP_OPTION_LIMIT).data()
        return [_display_value(record["label"], record.get("identifier")) for record in result]


def get_similar_compounds(driver: Driver, compound_names: List[str]):
    """Retrieve Compound-[:SIMILAR_TO]-Compound neighborhoods from the current schema."""
    if not compound_names:
        raise ValueError("Please select at least one compound.")
    query = f"""
    MATCH path=(c:Compound)-[r:SIMILAR_TO]-(c2:Compound)
    WHERE {COMPOUND_FILTER}
      AND elementId(c2) <> elementId(c)
    RETURN path
    ORDER BY coalesce(r.score, r.edge_weight, r.tanimoto, r.similarity, 0) DESC
    LIMIT $limit
    """
    params = _filter_params(compound_names)
    logger.info("Retrieving similar compounds for %s", compound_names)
    with _session(driver) as session:
        return session.run(query, **params).graph()


def show_bioassays(driver: Driver, compound_names: List[str]):
    """Retrieve BioAssay/MeasureGrp/Endpoint evidence linked to selected compounds."""
    if not compound_names:
        raise ValueError("Please select at least one compound.")
    query = f"""
    CALL {{
      MATCH path=(ba:BioAssay)-[:HAS_MEASURE_GROUP]->(mg:MeasureGrp)-[:HAS_ENDPOINT]->(e:Endpoint)-[:ABOUT_SUBSTANCE]->(s:Substance)-[:STANDARDIZED_TO]->(c:Compound)
      WHERE {COMPOUND_FILTER}
      RETURN path
      UNION
      MATCH path=(c:Compound)<-[:ASSERTS_CHEMICAL]-(i:Interaction)-[:SUPPORTED_BY_ENDPOINT]->(e:Endpoint)<-[:HAS_ENDPOINT]-(mg:MeasureGrp)<-[:HAS_MEASURE_GROUP]-(ba:BioAssay)
      WHERE {COMPOUND_FILTER}
      RETURN path
      UNION
      MATCH path=(c:Compound)<-[:ASSERTS_CHEMICAL]-(i:Interaction)-[:SUPPORTED_BY_ASSAY]->(ba:BioAssay)
      WHERE {COMPOUND_FILTER}
      RETURN path
    }}
    RETURN path
    LIMIT $limit
    """
    params = _filter_params(compound_names)
    logger.info("Retrieving BioAssay evidence for %s", compound_names)
    with _session(driver) as session:
        return session.run(query, **params).graph()


def show_cooccurrence_cpd_cpd(driver: Driver, compound_names: List[str], gene_symbols: List[str] = None):
    """Retrieve related compounds using shared PRING evidence objects."""
    if not compound_names:
        raise ValueError("Please select at least one compound.")
    query = f"""
    CALL {{
      MATCH path=(c:Compound)<-[:ASSERTS_CHEMICAL]-(i1:Interaction)-[:ASSERTS_TARGET]->(p:Protein)<-[:ASSERTS_TARGET]-(i2:Interaction)-[:ASSERTS_CHEMICAL]->(c2:Compound)
      WHERE {COMPOUND_FILTER}
        AND elementId(c2) <> elementId(c)
      RETURN path
      UNION
      MATCH path=(c:Compound)<-[:ASSERTS_CHEMICAL]-(i1:Interaction)-[:SUPPORTED_BY_ASSAY]->(ba:BioAssay)<-[:SUPPORTED_BY_ASSAY]-(i2:Interaction)-[:ASSERTS_CHEMICAL]->(c2:Compound)
      WHERE {COMPOUND_FILTER}
        AND elementId(c2) <> elementId(c)
      RETURN path
      UNION
      MATCH path=(c:Compound)<-[:ASSERTS_CHEMICAL]-(i1:Interaction)-[:SUPPORTED_BY_REFERENCE]->(r:Reference)<-[:SUPPORTED_BY_REFERENCE]-(i2:Interaction)-[:ASSERTS_CHEMICAL]->(c2:Compound)
      WHERE {COMPOUND_FILTER}
        AND elementId(c2) <> elementId(c)
      RETURN path
    }}
    RETURN path
    LIMIT $limit
    """
    params = _filter_params(compound_names, gene_symbols)
    logger.info("Retrieving compound-compound evidence relationships")
    with _session(driver) as session:
        return session.run(query, **params).graph()


def show_cooccurrence_cpd_gene(driver: Driver, compound_names: List[str], gene_symbols: List[str]):
    """Retrieve evidence-backed compound-target paths for selected pairs."""
    if not compound_names or not gene_symbols:
        raise ValueError("Please select at least one compound and one CYP450 target.")
    query = f"""
    CALL {{
      MATCH path=(c:Compound)<-[:ASSERTS_CHEMICAL]-(i:Interaction)-[:ASSERTS_TARGET]->(p:Protein)
      WHERE {COMPOUND_FILTER}
        AND {TARGET_FILTER}
      RETURN path
      UNION
      MATCH path=(c:Compound)<-[:STANDARDIZED_TO]-(s:Substance)<-[:ABOUT_SUBSTANCE]-(e:Endpoint)<-[:HAS_ENDPOINT]-(mg:MeasureGrp)-[:TESTED_ON]->(p:Protein)
      WHERE {COMPOUND_FILTER}
        AND {TARGET_FILTER}
      RETURN path
    }}
    RETURN path
    LIMIT $limit
    """
    params = _filter_params(compound_names, gene_symbols)
    logger.info("Retrieving compound-target evidence paths")
    with _session(driver) as session:
        return session.run(query, **params).graph()


def show_pubchem_interactions(driver: Driver, compound_names: List[str], gene_symbols: List[str]):
    """Retrieve PubChem-derived compound-protein Interaction evidence."""
    if not compound_names or not gene_symbols:
        raise ValueError("Please select at least one compound and one CYP450 target.")
    query = f"""
    MATCH path=(c:Compound)<-[:ASSERTS_CHEMICAL]-(i:Interaction)-[:ASSERTS_TARGET]->(p:Protein)
    WHERE {COMPOUND_FILTER}
      AND {TARGET_FILTER}
    WITH DISTINCT c, i, p, path
    LIMIT $limit
    OPTIONAL MATCH assay_path=(i)-[:SUPPORTED_BY_ASSAY]->(:BioAssay)
    OPTIONAL MATCH endpoint_path=(i)-[:SUPPORTED_BY_ENDPOINT]->(:Endpoint)<-[:HAS_ENDPOINT]-(:MeasureGrp)<-[:HAS_MEASURE_GROUP]-(:BioAssay)
    OPTIONAL MATCH reference_path=(i)-[:SUPPORTED_BY_REFERENCE]->(:Reference)
    OPTIONAL MATCH assay_reference_path=(i)-[:SUPPORTED_BY_ASSAY]->(:BioAssay)-[:DESCRIBED_BY]->(:Reference)
    OPTIONAL MATCH organism_path=(i)-[:SCOPED_TO_ORGANISM]->(:Organism)
    RETURN path, assay_path, endpoint_path, reference_path, assay_reference_path, organism_path
    """
    params = _filter_params(compound_names, gene_symbols)
    logger.info("Retrieving PubChem-derived interactions between %s and %s", compound_names, gene_symbols)
    with _session(driver) as session:
        return session.run(query, **params).graph()


def show_external_interactions(driver: Driver, compound_names: List[str], gene_symbols: List[str]):
    """Retrieve compound/protein enrichment context for selected PRING interactions."""
    if not compound_names or not gene_symbols:
        raise ValueError("Please select at least one compound and one CYP450 target.")
    query = f"""
    MATCH base=(c:Compound)<-[:ASSERTS_CHEMICAL]-(i:Interaction)-[:ASSERTS_TARGET]->(p:Protein)
    WHERE {COMPOUND_FILTER}
      AND {TARGET_FILTER}
    WITH DISTINCT c, p, i, base
    LIMIT $limit
    OPTIONAL MATCH compound_context=(c)-[:HAS_PROPERTIES|HAS_STRUCTURE|HAS_SYNONYMS|HAS_MOLECULAR_REPRESENTATION|HAS_NEIGHBOR_SET]->()
    OPTIONAL MATCH protein_context=(p)-[:ENCODED_BY|HAS_UNIPROT_RECORD|HAS_GO_ANNOTATION|MAPS_TO_REACTOME_PATHWAY|PARTICIPATES_IN|HAS_INTERPRO_DOMAIN|HAS_PDB_STRUCTURE|HAS_ALPHAFOLD_MODEL|HAS_PROTEIN_EMBEDDING]->()
    OPTIONAL MATCH reactome_alignment=(p)-[:MAPS_TO_REACTOME_PATHWAY]->(:Reactome)-[:ALIGNS_TO_PATHWAY]->(:Pathway)
    OPTIONAL MATCH uniprot_embedding=(:UniProt)<-[:HAS_UNIPROT_RECORD]-(p)-[:HAS_PROTEIN_EMBEDDING]->(:ProtEmbed)
    RETURN base, compound_context, protein_context, reactome_alignment, uniprot_embedding
    """
    params = _filter_params(compound_names, gene_symbols)
    logger.info("Retrieving current-schema enrichment context")
    with _session(driver) as session:
        return session.run(query, **params).graph()


def get_neo4j_statistics(driver: Driver) -> dict[str, int]:
    """Retrieve compact PRING KG statistics for the fixed bottom bar."""
    queries = {
        "compounds_count": "MATCH (c:Compound) RETURN count(c) AS count",
        "proteins_count": "MATCH (p:Protein) RETURN count(p) AS count",
        "bioassays_count": "MATCH (ba:BioAssay) RETURN count(ba) AS count",
        "interactions_count": "MATCH (i:Interaction) RETURN count(i) AS count",
        "relationships_count": "MATCH ()-[r]->() RETURN count(r) AS count",
    }
    statistics: dict[str, int] = {}
    with _session(driver) as session:
        for stat_name, query in queries.items():
            try:
                statistics[stat_name] = int(session.run(query).single()["count"])
            except Exception:
                statistics[stat_name] = 0
    return statistics
