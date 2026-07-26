"""Visualization, export, table, and report utilities for the PRING Streamlit app."""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import zipfile
from collections import Counter, OrderedDict
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import pandas as pd
from pyvis.network import Network

import streamlit as st
import streamlit.components.v1 as components
from utils.neo4j_utils import get_neo4j_statistics


APP_BRAND_KICKER = "PRING Knowledge Graph"
APP_BRAND_NAME = "CYP450-KG Explorer"
APP_BRAND_DOI = "10.5281/zenodo.15323478"
APP_BRAND_DOI_URL = f"https://doi.org/{APP_BRAND_DOI}"
APP_BRAND_TAGLINE = "Compound, assay, and target evidence from PRING-generated Neo4j graphs."
APP_LICENSE_NAME = "GNU General Public License v3.0 (GPL-3.0)"
APP_LICENSE_URL = "https://www.gnu.org/licenses/gpl-3.0.en.html"
APP_LICENSE_DISCLAIMER = (
    "CYP450-KG Explorer is distributed under the GNU General Public License v3.0 (GPL-3.0). "
    "You may use, modify, and redistribute this software under GPL-3.0 terms. "
    "This software and its exported graph, table, and report outputs are provided without warranty; "
    "all scientific, modelling, and regulatory interpretations should be independently reviewed."
)


def _brand_logo_data_uri() -> str:
    """Return the embedded application logo for standalone HTML exports."""
    candidates = [
        Path(__file__).resolve().parents[1] / "images" / "kg_icon.webp",
        Path("/opt/kg/streamlit/images/kg_icon.webp"),
    ]
    for icon_path in candidates:
        try:
            if icon_path.exists():
                mime = "image/webp" if icon_path.suffix.lower() == ".webp" else "image/png"
                encoded = base64.b64encode(icon_path.read_bytes()).decode("utf-8")
                return f"data:{mime};base64,{encoded}"
        except OSError:
            continue
    return ""


def _brand_logo_html(class_name: str = "export-brand-logo") -> str:
    """Render the logo or a compact fallback for standalone exports."""
    logo_uri = _brand_logo_data_uri()
    if logo_uri:
        return f'<img class="{class_name}" src="{logo_uri}" alt="{APP_BRAND_NAME} logo">'
    return f'<div class="{class_name} export-brand-logo-fallback">KG</div>'


# Fixed node colors for the current PRING schema. Keep this as the single source
# of truth for nodes and legend so colors never drift between graph and legend.
NODE_TYPE_COLORS: dict[str, str] = {
    "Compound": "#F59E0B",
    "Protein": "#2563EB",
    "Gene": "#38BDF8",
    "Interaction": "#8B5CF6",
    "BioAssay": "#EF4444",
    "MeasureGrp": "#14B8A6",
    "Endpoint": "#22C55E",
    "Substance": "#A16207",
    "Reference": "#64748B",
    "Source": "#EAB308",
    "Organism": "#84CC16",
    "GO": "#06B6D4",
    "Reactome": "#7C3AED",
    "Pathway": "#A855F7",
    "InterPro": "#EC4899",
    "PDB": "#0EA5E9",
    "AlphaFold": "#6366F1",
    "UniProt": "#0284C7",
    "ProtEmbed": "#475569",
    "Properties": "#10B981",
    "Structure": "#F97316",
    "Synonyms": "#FB7185",
    "MolGraph": "#D946EF",
    "Neighbors": "#F43F5E",
    "Node": "#94A3B8",
}

RELATIONSHIP_COLORS: dict[str, str] = {
    "SIMILAR_TO": "#F59E0B",
    "ASSERTS_CHEMICAL": "#8B5CF6",
    "ASSERTS_TARGET": "#2563EB",
    "SUPPORTED_BY_ASSAY": "#EF4444",
    "SUPPORTED_BY_ENDPOINT": "#22C55E",
    "SUPPORTED_BY_REFERENCE": "#64748B",
    "HAS_MEASURE_GROUP": "#14B8A6",
    "HAS_ENDPOINT": "#22C55E",
    "ABOUT_SUBSTANCE": "#A16207",
    "STANDARDIZED_TO": "#F59E0B",
    "TESTED_ON": "#2563EB",
    "HAS_GO_ANNOTATION": "#06B6D4",
    "MAPS_TO_REACTOME_PATHWAY": "#7C3AED",
    "PARTICIPATES_IN": "#A855F7",
    "HAS_INTERPRO_DOMAIN": "#EC4899",
    "HAS_PDB_STRUCTURE": "#0EA5E9",
    "HAS_ALPHAFOLD_MODEL": "#6366F1",
    "HAS_UNIPROT_RECORD": "#0284C7",
    "HAS_PROTEIN_EMBEDDING": "#475569",
}


ACTION_REPORT_COPY: dict[str, dict[str, str]] = {
    "Show Similar Compounds": {
        "title": "Compound similarity neighborhood report",
        "purpose": "Summarizes structurally or feature-similar compounds connected through SIMILAR_TO relationships.",
        "interpretation": "Use this view to identify analogues, candidate neighbor compounds, and chemical neighborhoods that may support inference or link-prediction workflows.",
    },
    "Show Related BioAssays": {
        "title": "BioAssay evidence report",
        "purpose": "Summarizes PubChem BioAssay, measure group, endpoint, substance, source, organism, and reference evidence linked to the selected compounds.",
        "interpretation": "Use this report to verify assay provenance, endpoint coverage, measurement context, and supporting references before downstream modeling.",
    },
    "Co-Occurrence in Literature (Compound-Compound)": {
        "title": "Compound-compound shared evidence report",
        "purpose": "Summarizes compound pairs connected through shared references or shared assay evidence in the PRING graph.",
        "interpretation": "Use this view to detect compounds that appear in the same evidence context and may share experimental or literature support.",
    },
    "Evidence Co-Occurrence (Compound-Target)": {
        "title": "Compound-target evidence co-occurrence report",
        "purpose": "Summarizes evidence-backed compound-target paths and their optional assay, endpoint, reference, and gene context.",
        "interpretation": "Use this report to inspect whether selected compounds and CYP450 proteins are connected by direct interaction assertions or endpoint/measure-group evidence.",
    },
    "Compound-Target Interactions (PubChem)": {
        "title": "PubChem-derived compound-target interaction report",
        "purpose": "Summarizes PRING Interaction nodes asserting compound-protein activity, including assay, endpoint, organism, and reference support where available.",
        "interpretation": "Use this report as a compact evidence audit for CYP450 interaction records before graph querying, visualization, or model training.",
    },
    "Compound-Target Enrichment Context": {
        "title": "Compound-target enrichment context report",
        "purpose": "Summarizes compound and protein enrichment layers around selected compound-target interactions.",
        "interpretation": "Use this report to inspect contextual features such as molecular properties, structures, synonyms, UniProt, GO, Reactome, InterPro, PDB, AlphaFold, and embeddings.",
    },
}


NODE_TYPE_EXPLANATIONS: dict[str, str] = {
    "Compound": "PubChem-standardized chemical entity. In reports, this is the main chemical being explored or a neighbor compound.",
    "Substance": "Submitted PubChem substance record. It explains how an external submitted record was standardized to a Compound CID.",
    "Protein": "CYP450 or other protein target. In interaction views, this is the biological target asserted by the evidence.",
    "Gene": "Gene encoding or associated with a protein target. Use it mainly for biological naming and target context.",
    "Interaction": "Evidence-derived assertion connecting a compound to a protein target. This is usually the central evidence node for compound-target activity.",
    "BioAssay": "PubChem assay-level record. It gives the experimental source or protocol context behind measurements.",
    "MeasureGrp": "Assay measurement group. It groups endpoint observations under a specific experimental condition or readout grouping.",
    "Endpoint": "Measured activity/readout such as IC50, inhibition, activity class, or another assay endpoint. These are important for interpreting interaction evidence.",
    "Reference": "Publication or citation support. More references can strengthen provenance but do not automatically mean stronger biological effect.",
    "Source": "Data submitter or source organization. Useful for provenance and reproducibility checks.",
    "Organism": "Taxonomic context for assay or interaction evidence. For CYP450 interpretation, check that organism context matches the intended human target when relevant.",
    "GO": "Gene Ontology annotation providing functional context for the protein.",
    "Reactome": "Reactome pathway annotation giving pathway-level context.",
    "Pathway": "Pathway abstraction connected to Reactome or other pathway resources.",
    "InterPro": "Protein family/domain annotation. Useful for interpreting target biology and feature engineering.",
    "PDB": "Experimental protein structure context.",
    "AlphaFold": "Predicted protein structure context.",
    "UniProt": "UniProt protein record with curated target metadata.",
    "ProtEmbed": "Protein embedding feature node. Useful for downstream graph ML but not direct experimental evidence.",
    "Properties": "Computed or collected molecular property feature node.",
    "Structure": "Molecular representation such as SMILES/InChI. Useful for chemical feature extraction.",
    "Synonyms": "Alternative chemical names. Useful for search and identifier harmonization.",
    "MolGraph": "Chemical graph/fingerprint representation for modelling.",
    "Neighbors": "Similarity-neighborhood feature context.",
    "Node": "Other graph node type not explicitly mapped in the visualization palette.",
}

RELATIONSHIP_EXPLANATIONS: dict[str, str] = {
    "SIMILAR_TO": "Chemical-neighborhood relation. Treat it as structural or feature similarity, not direct biological activity evidence unless supported by assay/interactions.",
    "ASSERTS_CHEMICAL": "Links an Interaction assertion to the compound it concerns.",
    "ASSERTS_TARGET": "Links an Interaction assertion to the protein target it concerns.",
    "SUPPORTED_BY_ASSAY": "Connects an assertion to BioAssay support.",
    "SUPPORTED_BY_ENDPOINT": "Connects an assertion to endpoint/readout support.",
    "SUPPORTED_BY_REFERENCE": "Connects an assertion to literature or reference support.",
    "HAS_MEASURE_GROUP": "Assay-to-measurement-group structure. Use it to move from assay-level context to specific measured endpoints.",
    "HAS_ENDPOINT": "Connects a measure group to a measured endpoint/readout.",
    "ABOUT_SUBSTANCE": "Endpoint is about a submitted PubChem Substance record.",
    "STANDARDIZED_TO": "Maps a submitted Substance to a standardized PubChem Compound.",
    "TESTED_ON": "Indicates the target or biological entity tested in a measure group.",
    "HAS_SOURCE": "Links a record to its submitter/source.",
    "SUBMITTED_BY": "Links a Substance to the source that submitted it.",
    "DESCRIBED_BY": "Links assay/evidence to a descriptive reference.",
    "IN_ORGANISM": "Provides organism/taxonomic context.",
    "SCOPED_TO_ORGANISM": "Defines the organism context for an interaction assertion.",
    "ENCODED_BY": "Connects protein target to its gene.",
    "HAS_GO_ANNOTATION": "Adds GO functional context to a protein.",
    "MAPS_TO_REACTOME_PATHWAY": "Adds Reactome pathway context to a protein.",
    "PARTICIPATES_IN": "Connects protein/entity to a pathway process.",
    "ALIGNS_TO_PATHWAY": "Connects pathway resources that represent related pathway concepts.",
    "HAS_INTERPRO_DOMAIN": "Adds protein family/domain annotation.",
    "HAS_PDB_STRUCTURE": "Adds experimental structure context.",
    "HAS_ALPHAFOLD_MODEL": "Adds predicted structure context.",
    "HAS_UNIPROT_RECORD": "Connects a protein node to UniProt metadata.",
    "HAS_PROTEIN_EMBEDDING": "Connects a protein to embedding features for modelling.",
    "HAS_PROPERTIES": "Connects compound to molecular properties.",
    "HAS_STRUCTURE": "Connects compound to molecular structure representation.",
    "HAS_SYNONYMS": "Connects compound to alternative names.",
    "HAS_MOLECULAR_REPRESENTATION": "Connects compound to molecular representation features.",
    "HAS_NEIGHBOR_SET": "Connects compound to similarity-neighborhood features.",
}


def extract_graph_data(graph):
    """Extract nodes and relationships from a Neo4j Graph object."""
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[tuple[str, str, str, dict[str, Any]]] = []

    for node in graph.nodes:
        node_id = node.element_id
        props = dict(node)
        props["labels"] = list(node.labels)
        nodes[node_id] = props

    for rel in graph.relationships:
        edges.append((rel.start_node.element_id, rel.end_node.element_id, rel.type, dict(rel)))

    return nodes, edges


def unescape_quotes(value):
    """Replace escaped single and double quotes in string values."""
    if isinstance(value, str):
        return value.replace("\\'", "'").replace('\\"', '"')
    return value


def _first_non_empty(props: dict[str, Any], keys: list[str], default: str = "") -> str:
    """Return the first non-empty property value for display labels."""
    for key in keys:
        value = props.get(key)
        if value not in (None, "", [], {}):
            return str(value)
    return default


def _primary_label(props: dict[str, Any]) -> str:
    """Return the most informative node label for visualization."""
    labels = props.get("labels") or []
    for label in NODE_TYPE_COLORS:
        if label in labels:
            return label
    return labels[0] if labels else "Node"


NODE_LABEL_FIELDS: dict[str, list[str]] = {
    "Compound": ["preferred_name", "name", "CompoundName", "cid"],
    "Protein": ["cyp_symbol", "gene_symbol", "symbol", "name", "uniprot_id", "uniprot_accession", "accession", "protein_id", "id", "ProteinRefSeqAccession"],
    "Gene": ["gene_symbol", "cyp_symbol", "symbol", "name", "gene_id"],
    "BioAssay": ["assay_name", "AssayName", "name", "aid"],
    "MeasureGrp": ["measure_group_id", "measuregrp_id", "mgid", "id"],
    "Endpoint": ["endpoint_name", "endpoint_type", "activity_type", "endpoint_id", "id"],
    "Interaction": ["interaction_type", "activity_class", "interaction_id", "id"],
    "Reference": ["title", "pmid", "doi", "reference_id"],
    "Substance": ["sid", "substance_id", "name"],
    "Organism": ["scientific_name", "name", "taxid"],
    "Source": ["source_name", "name", "source_id"],
    "GO": ["go_label", "name", "go_id"],
    "Reactome": ["pathway_name", "name", "reactome_id"],
    "Pathway": ["pathway_name", "name", "pathway_id"],
    "InterPro": ["name", "interpro_id"],
    "PDB": ["pdb_id", "name"],
    "AlphaFold": ["alphafold_id", "model_id", "name"],
    "UniProt": ["uniprot_id", "accession", "name"],
    "ProtEmbed": ["embedding_id", "model", "name"],
    "Properties": ["molecular_weight", "xlogp", "tpsa", "id"],
    "Structure": ["canonical_smiles", "isomeric_smiles", "inchi_key", "id"],
    "Synonyms": ["synonym", "synonyms", "name", "id"],
    "MolGraph": ["molgraph_id", "fingerprint", "id"],
    "Neighbors": ["neighbor_set_id", "id"],
}


def _node_full_display_label(label: str, props: dict[str, Any]) -> str:
    """Build a full readable node label for reports and table exports."""
    return _first_non_empty(props, NODE_LABEL_FIELDS.get(label, ["name", "id"]), default=label)


def _node_display_label(label: str, props: dict[str, Any]) -> str:
    """Build short node text suitable for graph labels only."""
    value = _node_full_display_label(label, props)
    return value[:31] + "..." if len(value) > 34 else value


def _node_color(label: str) -> str:
    """Return a stable node color for the current PRING schema."""
    return NODE_TYPE_COLORS.get(label, NODE_TYPE_COLORS["Node"])


def _relationship_color(relationship_type: str) -> str:
    """Return a stable edge color when known, otherwise a neutral color."""
    return RELATIONSHIP_COLORS.get(relationship_type, "#64748B")


def _format_value(value: Any) -> str:
    """Format long/list values for compact HTML tooltips."""
    value = unescape_quotes(value)
    if isinstance(value, (list, tuple, set)):
        value = ", ".join(str(v) for v in list(value)[:10])
    elif isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False)[:500]
    else:
        value = str(value)
    if len(value) > 180:
        value = value[:177] + "..."
    return escape(value)



def _looks_like_url(value: Any) -> bool:
    """Return True when a property value looks like an external URL."""
    return isinstance(value, str) and bool(re.match(r"^https?://", value.strip(), flags=re.IGNORECASE))


def _clean_external_value(value: Any) -> str:
    """Convert scalar property values into normalized link text."""
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, (list, tuple, set)):
        for item in value:
            cleaned = _clean_external_value(item)
            if cleaned:
                return cleaned
        return ""
    text = str(unescape_quotes(value)).strip()
    return "" if text.lower() in {"none", "nan", "na", "n/a"} else text


def _first_external_prop(props: dict[str, Any], keys: list[str]) -> str:
    """Return first non-empty value from possible identifier properties."""
    for key in keys:
        value = _clean_external_value(props.get(key))
        if value:
            return value
    return ""


def _add_external_link(items: list[tuple[str, str]], seen: set[str], label: str, url: str) -> None:
    """Append an external link when valid and not already present."""
    if not _looks_like_url(url):
        return
    url = url.strip()
    if url in seen:
        return
    seen.add(url)
    items.append((label, url))


def _external_database_links(entity_type: str, props: dict[str, Any]) -> list[tuple[str, str]]:
    """Create useful external database links for tooltip entities.

    Links are derived from common PRING/PubChem/UniProt/GO/Reactome/PDB fields
    as well as any URL-like properties already stored in Neo4j.
    """
    items: list[tuple[str, str]] = []
    seen: set[str] = set()

    # Preserve direct URLs stored in the graph first.
    for key, value in props.items():
        if key == "labels" or value in (None, "", [], {}):
            continue
        if isinstance(value, str) and _looks_like_url(value):
            label = str(key).replace("_", " ").replace("url", "URL").title()
            _add_external_link(items, seen, label, value)
        elif isinstance(value, (list, tuple, set)):
            for idx, item in enumerate(value, start=1):
                if _looks_like_url(item):
                    label = f"{str(key).replace('_', ' ').title()} {idx}"
                    _add_external_link(items, seen, label, str(item))

    doi = _first_external_prop(props, ["doi", "DOI"])
    if doi:
        doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE).strip()
        _add_external_link(items, seen, "DOI", f"https://doi.org/{quote_plus(doi, safe='/._-;()')}")

    pmid = _first_external_prop(props, ["pmid", "PMID", "pubmed_id", "PubMedID"])
    if pmid:
        pmid = re.sub(r"[^0-9]", "", pmid)
        if pmid:
            _add_external_link(items, seen, "PubMed", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")

    cid = _first_external_prop(props, ["cid", "CID", "pubchem_cid", "PubChemCID"])
    if not cid:
        pubchem_uri = _first_external_prop(props, ["pubchem_uri", "compound_uri", "uri"])
        match = re.search(r"CID[:_/ ]?(\d+)", pubchem_uri, flags=re.IGNORECASE)
        if match:
            cid = match.group(1)
    if cid and str(cid).isdigit():
        _add_external_link(items, seen, "PubChem Compound", f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}")

    sid = _first_external_prop(props, ["sid", "SID", "substance_id", "PubChemSID"])
    if sid and str(sid).isdigit():
        _add_external_link(items, seen, "PubChem Substance", f"https://pubchem.ncbi.nlm.nih.gov/substance/{sid}")

    aid = _first_external_prop(props, ["aid", "AID", "bioassay_id", "assay_id", "PubChemAID"])
    if aid and str(aid).isdigit():
        _add_external_link(items, seen, "PubChem BioAssay", f"https://pubchem.ncbi.nlm.nih.gov/bioassay/{aid}")

    accession = _first_external_prop(
        props,
        ["uniprot_id", "uniprot_accession", "accession", "protein_id", "UniProt", "uniprot"],
    )
    if accession and re.match(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9](?:-[0-9]+)?$|^[OPQ][0-9][A-Z0-9]{3}[0-9](?:-[0-9]+)?$", accession):
        _add_external_link(items, seen, "UniProt", f"https://www.uniprot.org/uniprotkb/{accession}/entry")
        _add_external_link(items, seen, "AlphaFold", f"https://alphafold.ebi.ac.uk/entry/{accession.split('-')[0]}")

    alphafold_id = _first_external_prop(props, ["alphafold_id", "model_id", "model", "name"])
    match = re.match(r"^AF-([A-Z0-9]+)-F\d+", alphafold_id)
    if match:
        _add_external_link(items, seen, "AlphaFold", f"https://alphafold.ebi.ac.uk/entry/{match.group(1)}")

    refseq = _first_external_prop(props, ["refseq_accession", "ProteinRefSeqAccession", "refseq_id"])
    if refseq:
        _add_external_link(items, seen, "NCBI Protein", f"https://www.ncbi.nlm.nih.gov/protein/{quote_plus(refseq)}")

    symbol = _first_external_prop(props, ["cyp_symbol", "gene_symbol", "symbol", "hgnc_symbol"])
    if symbol:
        _add_external_link(items, seen, "NCBI Gene search", f"https://www.ncbi.nlm.nih.gov/gene/?term={quote_plus(symbol)}")

    gene_id = _first_external_prop(props, ["gene_id", "entrez_gene_id", "GeneID"])
    if gene_id and str(gene_id).isdigit():
        _add_external_link(items, seen, "NCBI Gene", f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}")

    go_id = _first_external_prop(props, ["go_id", "GO_ID", "id"])
    match = re.search(r"GO:\d+", go_id)
    if match:
        go = match.group(0)
        _add_external_link(items, seen, "Gene Ontology", f"https://amigo.geneontology.org/amigo/term/{go.replace(':', '%3A')}")

    reactome_id = _first_external_prop(props, ["reactome_id", "pathway_id", "stable_id", "id"])
    match = re.search(r"R-HSA-\d+", reactome_id)
    if match:
        _add_external_link(items, seen, "Reactome", f"https://reactome.org/content/detail/{match.group(0)}")

    interpro_id = _first_external_prop(props, ["interpro_id", "ipr_id", "id"])
    match = re.search(r"IPR\d+", interpro_id, flags=re.IGNORECASE)
    if match:
        _add_external_link(items, seen, "InterPro", f"https://www.ebi.ac.uk/interpro/entry/InterPro/{match.group(0).upper()}/")

    pdb_id = _first_external_prop(props, ["pdb_id", "PDB", "structure_id", "id"])
    if pdb_id and re.match(r"^[0-9][A-Za-z0-9]{3}$", pdb_id):
        _add_external_link(items, seen, "RCSB PDB", f"https://www.rcsb.org/structure/{pdb_id.upper()}")

    inchi_key = _first_external_prop(props, ["inchi_key", "InChIKey", "jchem_inchi_key", "indigo_inchi_key"])
    if inchi_key:
        _add_external_link(items, seen, "PubChem InChIKey search", f"https://pubchem.ncbi.nlm.nih.gov/#query={quote_plus(inchi_key)}")

    # Keep tooltip compact.
    return items[:8]

def _copy_value(value: Any) -> str:
    """Return the full untruncated value used by tooltip copy buttons."""
    value = unescape_quotes(value)
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def _format_tooltip_html(kind: str, title: str, entity_type: str, props: dict[str, Any]) -> str:
    """Create reusable HTML for hover and pinned tooltips with copyable values and external links."""
    rows = [("Type" if kind == "node" else "Relationship", entity_type)]
    clean_props = {k: v for k, v in props.items() if k not in {"labels"} and v not in (None, "", [], {})}
    for key, value in list(clean_props.items())[:28]:
        rows.append((str(key), value))

    hidden_count = max(0, len(clean_props) - 28)
    body_parts = []
    copy_lines = []
    for key, value in rows:
        display_value = _format_value(value)
        full_value = _copy_value(value)
        copy_lines.append(f"{key}: {full_value}")
        body_parts.append(
            "<div class='kg-tooltip-row'>"
            f"<span>{escape(str(key))}</span>"
            f"<strong>{display_value}</strong>"
            f"<button type='button' class='kg-copy-value' data-copy='{escape(full_value, quote=True)}' title='Copy value'>Copy</button>"
            "</div>"
        )

    link_items = _external_database_links(entity_type, props)
    links_html = ""
    if link_items:
        link_parts = []
        for link_label, url in link_items:
            link_parts.append(
                f"<a class='kg-tooltip-link' href='{escape(url, quote=True)}' target='_blank' rel='noopener noreferrer'>"
                f"{escape(link_label)}</a>"
            )
        links_html = (
            "<div class='kg-tooltip-links'>"
            "<div class='kg-tooltip-section-title'>External database links</div>"
            f"<div class='kg-tooltip-link-grid'>{''.join(link_parts)}</div>"
            "</div>"
        )

    hidden = f"<div class='kg-tooltip-more'>+ {hidden_count} more properties in the table view</div>" if hidden_count else ""
    copy_all = escape("\n".join(copy_lines), quote=True)
    return (
        "<div class='kg-tooltip-topbar'>"
        f"<div class='kg-tooltip-heading'>{escape(title)}</div>"
        f"<button type='button' class='kg-tooltip-copy-all' data-copy='{copy_all}' title='Copy all visible tooltip values'>Copy all</button>"
        "</div>"
        f"{links_html}<div class='kg-tooltip-body'>{''.join(body_parts)}</div>{hidden}"
    )


def _safe_filename(value: str, suffix: str) -> str:
    """Create a stable download filename from a label/action string."""
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_").lower() or "pring_export"
    return f"{base}{suffix}"


def _json_safe(value: Any) -> Any:
    """Convert arbitrary Neo4j/Python values into JSON-friendly values."""
    value = unescape_quotes(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


def _table_safe_value(value: Any) -> Any:
    """Convert nested values into readable table cells."""
    value = _json_safe(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def graph_to_dataframes(graph) -> OrderedDict[str, pd.DataFrame]:
    """Create one DataFrame per node label plus a relationship DataFrame."""
    nodes, edges = extract_graph_data(graph)
    tables: OrderedDict[str, pd.DataFrame] = OrderedDict()

    label_groups: dict[str, list[dict[str, Any]]] = {}
    for node_id, props in nodes.items():
        label = _primary_label(props)
        row = {
            "Node ID": node_id,
            "Node Label": label,
            "Display Name": _node_full_display_label(label, props),
        }
        for key, value in props.items():
            if key == "labels":
                row["All Labels"] = "; ".join(str(v) for v in value)
            else:
                row[key] = _table_safe_value(value)
        label_groups.setdefault(label, []).append(row)

    for label in sorted(label_groups):
        tables[f"{label} nodes"] = pd.DataFrame(label_groups[label])

    relationship_rows: list[dict[str, Any]] = []
    for start_node, end_node, rel_type, rel_props in edges:
        start_props = nodes.get(start_node, {})
        end_props = nodes.get(end_node, {})
        start_label = _primary_label(start_props) if start_props else "Unknown"
        end_label = _primary_label(end_props) if end_props else "Unknown"
        row = {
            "Start Node ID": start_node,
            "Start Label": start_label,
            "Start Name": _node_full_display_label(start_label, start_props) if start_props else start_node,
            "Relationship Type": rel_type,
            "End Node ID": end_node,
            "End Label": end_label,
            "End Name": _node_full_display_label(end_label, end_props) if end_props else end_node,
        }
        for key, value in (rel_props or {}).items():
            row[key] = _table_safe_value(value)
        relationship_rows.append(row)

    if relationship_rows:
        tables["Relationships"] = pd.DataFrame(relationship_rows)
    return tables


def _branded_data_readme_html(
    tables: OrderedDict[str, pd.DataFrame],
    package_title: str,
    csv_files: list[dict[str, Any]],
    extra_file_names: list[str],
    generated_at: str,
) -> str:
    """Create a branded standalone README for table ZIP exports."""
    rows = "".join(
        "<tr>"
        f"<td>{escape(item['file'])}</td>"
        f"<td>{escape(item['table'])}</td>"
        f"<td>{item['rows']}</td>"
        f"<td>{item['columns']}</td>"
        "</tr>"
        for item in csv_files
    )
    extras = "".join(f"<li>{escape(name)}</li>" for name in extra_file_names) or "<li>No additional files.</li>"
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{escape(package_title)}</title>
<style>
:root {{ --kg-red:#EF4444; --kg-charcoal:#505050; --kg-dark:#111827; --kg-bg:#F3F6FA; --kg-border:#E2E8F0; }}
body {{ margin:0; padding:28px; background:var(--kg-bg); color:var(--kg-dark); font-family:Arial, sans-serif; }}
.export-card {{ max-width:1100px; margin:0 auto; background:#fff; border:1px solid var(--kg-border); border-radius:20px; overflow:hidden; box-shadow:0 14px 42px rgba(17,24,39,.10); }}
.export-header {{ display:flex; gap:16px; align-items:center; padding:24px 28px; background:linear-gradient(135deg,#FFFFFF 0%,#F8FAFC 100%); border-top:5px solid var(--kg-red); border-bottom:1px solid var(--kg-border); }}
.export-brand-logo {{ width:74px; height:74px; object-fit:contain; border:1px solid var(--kg-border); border-radius:18px; background:#fff; padding:8px; }}
.export-brand-logo-fallback {{ display:grid; place-items:center; font-weight:900; color:var(--kg-red); }}
.kicker {{ margin:0 0 5px; color:var(--kg-red); font-size:12px; font-weight:900; letter-spacing:.14em; text-transform:uppercase; }}
h1 {{ margin:0; font-size:28px; color:#374151; }}
.subtitle {{ margin:6px 0 0; color:#64748B; }}
.doi {{ display:inline-flex; gap:8px; align-items:center; margin-top:10px; padding:7px 10px; border-radius:999px; border:1px solid var(--kg-border); background:#fff; color:#0F2F5F; font-weight:800; text-decoration:none; }}
.doi span {{ background:#505050; color:#fff; padding:4px 8px; border-radius:999px; font-size:12px; }}
section {{ padding:24px 28px; border-top:1px solid var(--kg-border); }}
h2 {{ margin:0 0 12px; font-size:19px; }}
h2::before {{ content:""; display:inline-block; width:8px; height:18px; margin-right:9px; border-radius:999px; background:var(--kg-red); vertical-align:-3px; }}
p, li {{ line-height:1.65; }}
.table-wrap {{ overflow-x:auto; border:1px solid var(--kg-border); border-radius:12px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; background:#fff; }}
th {{ background:#F3F4F6; text-align:left; color:#374151; }}
th, td {{ border-bottom:1px solid var(--kg-border); padding:10px 12px; vertical-align:top; overflow-wrap:anywhere; }}
.footer {{ color:#64748B; font-size:13px; }}
.license-notice {{ background:#F8FAFC; border-top:1px solid var(--kg-border); color:#475569; font-size:12.5px; line-height:1.6; }}
.license-notice strong {{ color:#111827; }}
.license-notice a {{ color:#EF4444; font-weight:800; text-decoration:none; }}
</style>
</head>
<body>
<div class="export-card">
  <div class="export-header">
    {_brand_logo_html()}
    <div>
      <p class="kicker">{escape(APP_BRAND_KICKER)}</p>
      <h1>{escape(APP_BRAND_NAME)}</h1>
      <p class="subtitle">{escape(package_title)}</p>
      <a class="doi" href="{APP_BRAND_DOI_URL}"><span>DOI</span>{APP_BRAND_DOI}</a>
    </div>
  </div>
  <section>
    <h2>What is inside this data package?</h2>
    <p>This ZIP was generated by <strong>{escape(APP_BRAND_NAME)}</strong>. It contains the full tabular data behind the selected graph view, with one CSV per node type and one relationship table when relationships are present.</p>
    <p class="footer">Generated at {escape(generated_at)}.</p>
  </section>
  <section>
    <h2>CSV tables</h2>
    <div class="table-wrap"><table><thead><tr><th>File</th><th>Table</th><th>Rows</th><th>Columns</th></tr></thead><tbody>{rows}</tbody></table></div>
  </section>
  <section>
    <h2>Additional files</h2>
    <ul>{extras}</ul>
  </section>
  <section class="footer">Use the CSV tables together with the HTML report and graph export for a reproducible evidence audit trail.</section>
  <section class="license-notice">
    <strong>License disclaimer:</strong>
    {escape(APP_LICENSE_DISCLAIMER)}
    <br><a href="{APP_LICENSE_URL}" target="_blank">{escape(APP_LICENSE_NAME)}</a>
  </section>
</div>
</body>
</html>"""


def _dataframes_zip_bytes(
    tables: OrderedDict[str, pd.DataFrame],
    extra_files: dict[str, bytes | str] | None = None,
    package_title: str = "CYP450-KG Explorer tabular data export",
) -> bytes:
    """Compress all tables as branded CSV package with manifest and README."""
    generated_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    buffer = BytesIO()
    csv_files: list[dict[str, Any]] = []
    extra_file_names = sorted((extra_files or {}).keys())

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in tables.items():
            filename = _safe_filename(name, ".csv")
            csv_files.append({"file": filename, "table": name, "rows": int(df.shape[0]), "columns": int(df.shape[1])})
            zf.writestr(filename, df.to_csv(index=False).encode("utf-8"))

        manifest = {
            "application": APP_BRAND_NAME,
            "brand": APP_BRAND_KICKER,
            "doi": APP_BRAND_DOI,
            "doi_url": APP_BRAND_DOI_URL,
            "license": "GPL-3.0",
            "license_name": APP_LICENSE_NAME,
            "license_url": APP_LICENSE_URL,
            "license_disclaimer": APP_LICENSE_DISCLAIMER,
            "package_title": package_title,
            "generated_at_utc": generated_at,
            "csv_files": csv_files,
            "extra_files": extra_file_names,
        }
        zf.writestr("00_README.html", _branded_data_readme_html(tables, package_title, csv_files, extra_file_names, generated_at).encode("utf-8"))
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
        zf.writestr(
            "LICENSE_NOTICE.md",
            (f"# License notice\n\n{APP_LICENSE_DISCLAIMER}\n\nFull license text: {APP_LICENSE_URL}\n").encode("utf-8"),
        )

        for filename, content in (extra_files or {}).items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            zf.writestr(filename, data)
    buffer.seek(0)
    return buffer.getvalue()


def _graph_export_payload(graph) -> dict[str, Any]:
    """Build a JSON export payload for the currently displayed graph."""
    nodes, edges = extract_graph_data(graph)
    node_payload = []
    for node_id, props in nodes.items():
        label = _primary_label(props)
        clean_props = {k: v for k, v in props.items() if k != "labels"}
        node_payload.append(
            {
                "id": node_id,
                "label": label,
                "labels": _json_safe(props.get("labels", [])),
                "display_name": _node_full_display_label(label, props),
                "color": _node_color(label),
                "properties": _json_safe(clean_props),
            }
        )

    edge_payload = []
    for start_node, end_node, rel_type, rel_props in edges:
        edge_payload.append(
            {
                "source": start_node,
                "target": end_node,
                "relationship_type": rel_type,
                "color": _relationship_color(rel_type),
                "properties": _json_safe(rel_props or {}),
            }
        )

    return {
        "application": {
            "name": APP_BRAND_NAME,
            "brand": APP_BRAND_KICKER,
            "doi": APP_BRAND_DOI,
            "doi_url": APP_BRAND_DOI_URL,
            "tagline": APP_BRAND_TAGLINE,
        },
        "exported_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "node_count": len(node_payload),
        "edge_count": len(edge_payload),
        "nodes": node_payload,
        "edges": edge_payload,
    }


def display_graph(graph, action: str = "pring_graph") -> None:
    """Display an interactive PyVis graph and provide graph JSON/HTML downloads."""
    if not graph:
        st.warning("No graph data to display.")
        return

    nodes, edges = extract_graph_data(graph)
    if not nodes and not edges:
        st.info("The query returned no graph paths for the current selection.")
        return

    # Use in-line resources so the exported graph and the Streamlit iframe are self-contained.
    # Do not add custom overlay zoom/pan controls here: vis-network's native interaction layer
    # handles dragging, mouse-wheel zoom, node dragging, selection, and the navigation buttons.
    net = Network(
        height="760px",
        width="100%",
        bgcolor="#F8FAFC",
        font_color="#111827",
        directed=True,
        cdn_resources="in_line",
    )

    used_labels: dict[str, str] = {}
    for node_id, props in nodes.items():
        label = _primary_label(props)
        node_color = _node_color(label)
        used_labels[label] = node_color
        node_label = _node_display_label(label, props)
        tooltip_html = _format_tooltip_html("node", node_label, label, props)

        net.add_node(
            node_id,
            label=node_label,
            color={
                "background": node_color,
                "border": "#0F172A",
                "highlight": {"background": node_color, "border": "#111827"},
                "hover": {"background": node_color, "border": "#111827"},
            },
            shape="dot",
            size=20 if label in {"Compound", "Protein", "Interaction"} else 14,
            borderWidth=1.5,
            font={"size": 15, "face": "Arial", "color": "#111827", "strokeWidth": 3, "strokeColor": "#FFFFFF"},
            tooltipHtml=tooltip_html,
            tooltipTitle=node_label,
            schemaLabel=label,
            mass=2 if label in {"Compound", "Protein", "Interaction"} else 1,
        )

    for edge_index, (start_node, end_node, relationship_type, rel_props) in enumerate(edges):
        title = relationship_type.replace("_", " ").title()
        start_props = nodes.get(start_node, {})
        end_props = nodes.get(end_node, {})
        edge_display_props = {
            "Source": _node_full_display_label(_primary_label(start_props), start_props) if start_props else start_node,
            "Target": _node_full_display_label(_primary_label(end_props), end_props) if end_props else end_node,
        }
        edge_display_props.update(rel_props or {})
        tooltip_html = _format_tooltip_html("edge", title, relationship_type, edge_display_props)
        edge_color = _relationship_color(relationship_type)
        net.add_edge(
            start_node,
            end_node,
            id=f"edge_{edge_index}",
            label=relationship_type,
            arrows="to",
            color={"color": edge_color, "highlight": "#0F172A", "hover": "#0F172A"},
            font={"size": 10, "align": "middle", "color": "#334155", "strokeWidth": 3, "strokeColor": "#FFFFFF"},
            smooth={"enabled": True, "type": "dynamic"},
            tooltipHtml=tooltip_html,
            tooltipTitle=title,
            relationshipType=relationship_type,
            selectionWidth=3,
            hoverWidth=2,
        )

    # The options below intentionally mirror the interaction style of the reference implementation:
    # native drag, mouse wheel zoom, navigation buttons, hover, and selection remain owned by vis-network.
    net.set_options(
        json.dumps(
            {
                "interaction": {
                    "hover": True,
                    "tooltipDelay": 0,
                    "navigationButtons": True,
                    "keyboard": {"enabled": True, "bindToWindow": False},
                    "dragNodes": True,
                    "dragView": True,
                    "zoomView": True,
                    "zoomSpeed": 1,
                    "multiselect": False,
                    "selectable": True,
                    "hoverConnectedEdges": True,
                    "selectConnectedEdges": False,
                },
                "physics": {
                    "enabled": True,
                    "solver": "forceAtlas2Based",
                    "forceAtlas2Based": {
                        "gravitationalConstant": -55,
                        "centralGravity": 0.015,
                        "springLength": 140,
                        "springConstant": 0.08,
                        "avoidOverlap": 0.25,
                    },
                    "stabilization": {"enabled": True, "iterations": 120, "fit": True},
                },
                "edges": {
                    "smooth": {"enabled": True, "type": "dynamic"},
                    "selectionWidth": 3,
                    "hoverWidth": 2,
                },
                "nodes": {
                    "font": {"size": 15, "face": "Arial", "multi": "html"},
                },
                "layout": {"improvedLayout": False},
            }
        )
    )

    legend_items = "".join(
        f"<li><span style='background:{color}'></span>{escape(label)}</li>" for label, color in sorted(used_labels.items())
    )
    legend_html = f"""
    <details class="kg-legend">
      <summary><span>Node legend</span><em>{len(used_labels)}</em></summary>
      <ul>{legend_items}</ul>
    </details>
    """

    node_color_map_json = json.dumps({label: _node_color(label) for label in used_labels}, ensure_ascii=False)

    style = """
    <style>
    html, body { margin: 0; background: #F8FAFC; overflow: hidden; width: 100%; height: 100%; }
    body { position: relative; }
    #mynetwork {
        border: 1px solid #E2E8F0 !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08) !important;
        cursor: grab;
        pointer-events: auto !important;
        user-select: none !important;
        touch-action: auto !important;
    }
    #mynetwork:active { cursor: grabbing; }
    #mynetwork canvas { pointer-events: auto !important; }

    .kg-legend {
        position: absolute;
        left: 14px;
        top: 14px;
        width: fit-content;
        max-width: min(300px, calc(100% - 28px));
        max-height: none;
        overflow: visible;
        background: rgba(255,255,255,0.95);
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
        padding: 0;
        z-index: 40;
        backdrop-filter: blur(8px);
        font-family: Arial, sans-serif;
        pointer-events: auto;
    }
    .kg-legend summary {
        list-style: none;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        min-width: 158px;
        padding: 9px 11px;
        color: #0F172A;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: .02em;
        text-transform: uppercase;
        user-select: none;
    }
    .kg-legend summary::-webkit-details-marker { display: none; }
    .kg-legend summary:after { content: "▾"; color: #64748B; font-size: 11px; }
    .kg-legend[open] summary:after { content: "▴"; }
    .kg-legend summary em {
        margin-left: auto;
        font-style: normal;
        border-radius: 999px;
        padding: 2px 6px;
        background: #F1F5F9;
        color: #475569;
        font-size: 10px;
        font-weight: 900;
    }
    .kg-legend ul {
        list-style: none;
        padding: 0 11px 11px;
        margin: 0;
        display: grid;
        grid-template-columns: repeat(2, minmax(96px, 1fr));
        gap: 5px 9px;
        overflow: visible;
    }
    .kg-legend li { display: flex; align-items: center; gap: 6px; min-width: 0; font-size: 11.5px; color: #334155; white-space: nowrap; overflow: visible; }
    .kg-legend li span { flex: 0 0 9px; width: 9px; height: 9px; border-radius: 999px; display: inline-block; border: 1px solid rgba(15,23,42,.22); }


    .kg-tooltip-links {
        padding: 10px 12px;
        border-bottom: 1px solid #E2E8F0;
        background: #F8FAFC;
    }
    .kg-tooltip-section-title {
        margin-bottom: 6px;
        color: #475569;
        font-size: 11px;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: .06em;
    }
    .kg-tooltip-link-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .kg-tooltip-link {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        border-radius: 999px;
        padding: 5px 8px;
        background: #FFFFFF;
        border: 1px solid #CBD5E1;
        color: #334155 !important;
        text-decoration: none !important;
        font-size: 11px;
        font-weight: 800;
    }
    .kg-tooltip-link::after { content: "↗"; color: #EF4444; font-weight: 900; }
    .kg-tooltip-link:hover {
        border-color: #EF4444;
        color: #111827 !important;
        box-shadow: 0 4px 12px rgba(239, 68, 68, .12);
    }

    #kg-tooltip {
        display: none;
        position: fixed;
        min-width: 260px;
        max-width: 420px;
        max-height: 76vh;
        overflow: auto;
        background: rgba(255, 255, 255, 0.98);
        border: 1px solid #CBD5E1;
        border-radius: 14px;
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.20);
        padding: 14px;
        z-index: 9999;
        font-family: Arial, sans-serif;
        color: #0F172A;
        backdrop-filter: blur(10px);
        white-space: normal;
        overflow-wrap: anywhere;
        pointer-events: auto;
    }
    #kg-tooltip.kg-pinned { border-color: #EF4444; box-shadow: 0 20px 60px rgba(239, 68, 68, 0.20); }
    #kg-tooltip, #kg-tooltip * { user-select: text !important; }
    .kg-tooltip-topbar { display: flex; gap: 10px; align-items: flex-start; justify-content: space-between; margin: 0 28px 10px 0; padding-bottom: 8px; border-bottom: 1px solid #E2E8F0; }
    .kg-tooltip-heading { font-size: 15px; line-height: 1.35; font-weight: 800; color: #0F172A; margin: 0; }
    .kg-tooltip-row { display: grid; grid-template-columns: minmax(92px, 34%) minmax(0, 1fr) auto; gap: 10px; align-items: start; padding: 6px 0; border-bottom: 1px solid #F1F5F9; font-size: 12.5px; }
    .kg-tooltip-row span { color: #64748B; font-weight: 700; overflow-wrap: anywhere; }
    .kg-tooltip-row strong { color: #111827; font-weight: 500; overflow-wrap: anywhere; white-space: pre-wrap; }
    .kg-tooltip-more { margin-top: 8px; color: #64748B; font-size: 12px; font-style: italic; }
    .kg-tooltip-close { display: none; position: absolute; right: 10px; top: 8px; width: 24px; height: 24px; border: 0; border-radius: 999px; background: #E2E8F0; color: #0F172A; cursor: pointer; font-weight: 800; line-height: 24px; text-align: center; }
    #kg-tooltip.kg-pinned .kg-tooltip-close { display: block; }
    .kg-tooltip-hint { margin-top: 10px; font-size: 11.5px; color: #64748B; }
    .kg-copy-value, .kg-tooltip-copy-all { user-select: none !important; border: 1px solid #CBD5E1; background: #F8FAFC; color: #475569; border-radius: 999px; padding: 3px 7px; font-size: 10.5px; font-weight: 800; cursor: pointer; white-space: nowrap; }
    .kg-copy-value:hover, .kg-tooltip-copy-all:hover { border-color: #EF4444; color: #991B1B; background: #FEF2F2; }

    div.vis-tooltip, .vis-tooltip {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    .vis-navigation,
    .vis-network .vis-navigation {
        z-index: 9999 !important;
        pointer-events: auto !important;
    }
    .vis-navigation *,
    .vis-network .vis-navigation * {
        pointer-events: auto !important;
    }
    .vis-navigation .vis-button,
    .vis-network .vis-navigation .vis-button,
    .vis-network .vis-manipulation .vis-button {
        position: absolute !important;
        z-index: 10000 !important;
        pointer-events: auto !important;
        background-color: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        background-image: none !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.12), 0 8px 18px rgba(15, 23, 42, 0.06) !important;
        width: 30px !important;
        height: 30px !important;
        opacity: 1 !important;
        cursor: pointer !important;
    }
    .vis-network .vis-navigation .vis-button::before,
    .vis-network .vis-navigation .vis-button::after,
    .vis-network .vis-manipulation .vis-button::before,
    .vis-network .vis-manipulation .vis-button::after {
        content: "";
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        background-image: none !important;
    }
    .vis-network .vis-navigation .vis-button.vis-up::after { border-style: solid; border-width: 0 6px 8px 6px; border-color: transparent transparent #475569 transparent; background: transparent; }
    .vis-network .vis-navigation .vis-button.vis-down::after { border-style: solid; border-width: 8px 6px 0 6px; border-color: #475569 transparent transparent transparent; background: transparent; }
    .vis-network .vis-navigation .vis-button.vis-left::after { border-style: solid; border-width: 6px 8px 6px 0; border-color: transparent #475569 transparent transparent; background: transparent; }
    .vis-network .vis-navigation .vis-button.vis-right::after { border-style: solid; border-width: 6px 0 6px 8px; border-color: transparent transparent transparent #475569; background: transparent; }
    .vis-network .vis-navigation .vis-button.vis-zoomIn::before { width: 10px; height: 2px; background-color: #475569; }
    .vis-network .vis-navigation .vis-button.vis-zoomIn::after { width: 2px; height: 10px; background-color: #475569; }
    .vis-network .vis-navigation .vis-button.vis-zoomOut::after { width: 10px; height: 2px; background-color: #475569; }
    .vis-network .vis-navigation .vis-button.vis-zoomExtends::after { width: 12px; height: 12px; border: 2px solid #475569; border-radius: 3px; background: transparent; }
    .vis-network .vis-navigation .vis-button:hover,
    .vis-network .vis-manipulation .vis-button:hover { background-color: #FFFFFF !important; border-color: #EF4444 !important; box-shadow: 0 6px 18px rgba(239, 68, 68, 0.16) !important; }
    </style>
    """

    custom_js = f"""
    <script type="text/javascript">
    (function () {{
        const NODE_COLORS = {node_color_map_json};
        const tooltip = document.createElement("div");
        tooltip.id = "kg-tooltip";
        document.body.appendChild(tooltip);
        const networkContainer = document.getElementById("mynetwork");
        let pinned = false;
        let fittedAfterStabilization = false;
        let userInteractedWithView = false;

        function enforceNodeColors() {{
            try {{
                const updates = [];
                nodes.forEach(function (node) {{
                    const schemaLabel = node.schemaLabel || node.group || "Node";
                    const color = NODE_COLORS[schemaLabel];
                    if (color) {{
                        updates.push({{
                            id: node.id,
                            group: undefined,
                            color: {{
                                background: color,
                                border: "#0F172A",
                                highlight: {{background: color, border: "#111827"}},
                                hover: {{background: color, border: "#111827"}}
                            }}
                        }});
                    }}
                }});
                if (updates.length) nodes.update(updates);
            }} catch (e) {{ console.warn("Could not enforce node colors", e); }}
        }}

        function sourceEvent(params) {{
            if (!params || !params.event) return null;
            return params.event.srcEvent || params.event;
        }}

        function graphLocalPosition(params, itemKind) {{
            try {{
                const rect = networkContainer ? networkContainer.getBoundingClientRect() : {{left: 0, top: 0}};
                if (itemKind === "node" && params && params.nodes && params.nodes.length > 0) {{
                    const pos = network.getPositions([params.nodes[0]])[params.nodes[0]];
                    const dom = network.canvasToDOM(pos);
                    return {{x: rect.left + dom.x + 18, y: rect.top + dom.y + 18}};
                }}
                if (params && params.pointer && params.pointer.DOM) {{
                    return {{x: rect.left + params.pointer.DOM.x + 18, y: rect.top + params.pointer.DOM.y + 18}};
                }}
            }} catch (e) {{}}
            return null;
        }}

        function eventPosition(evt) {{
            if (evt && typeof evt.clientX === "number" && typeof evt.clientY === "number") return {{x: evt.clientX + 16, y: evt.clientY + 16}};
            return {{x: Math.max(24, window.innerWidth * 0.45), y: 72}};
        }}

        function positionTooltip(position) {{
            const point = position || {{x: Math.max(24, window.innerWidth * 0.45), y: 72}};
            const maxLeft = Math.max(18, window.innerWidth - tooltip.offsetWidth - 18);
            const maxTop = Math.max(18, window.innerHeight - tooltip.offsetHeight - 18);
            tooltip.style.left = Math.max(18, Math.min(point.x, maxLeft)) + "px";
            tooltip.style.top = Math.max(18, Math.min(point.y, maxTop)) + "px";
        }}

        function copyTextToClipboard(text, button) {{
            const resetText = button ? button.textContent : "";
            function markCopied() {{
                if (!button) return;
                button.textContent = "Copied";
                window.setTimeout(function () {{ button.textContent = resetText || "Copy"; }}, 1200);
            }}
            try {{
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard.writeText(text).then(markCopied).catch(function () {{ fallbackCopy(text); markCopied(); }});
                }} else {{
                    fallbackCopy(text);
                    markCopied();
                }}
            }} catch (e) {{
                fallbackCopy(text);
                markCopied();
            }}
        }}

        function fallbackCopy(text) {{
            const textarea = document.createElement("textarea");
            textarea.value = text || "";
            textarea.setAttribute("readonly", "readonly");
            textarea.style.position = "fixed";
            textarea.style.left = "-9999px";
            document.body.appendChild(textarea);
            textarea.select();
            try {{ document.execCommand("copy"); }} catch (e) {{ console.warn("Copy failed", e); }}
            document.body.removeChild(textarea);
        }}

        function attachTooltipCopyHandlers() {{
            tooltip.querySelectorAll(".kg-copy-value, .kg-tooltip-copy-all").forEach(function (button) {{
                button.onclick = function (event) {{
                    event.preventDefault();
                    event.stopPropagation();
                    copyTextToClipboard(button.getAttribute("data-copy") || "", button);
                }};
            }});
        }}

        function renderTooltip(item, position, shouldPin) {{
            if (!item || !item.tooltipHtml) return;
            pinned = !!shouldPin;
            tooltip.classList.toggle("kg-pinned", pinned);
            tooltip.innerHTML = "<button class='kg-tooltip-close' title='Close'>×</button>" + item.tooltipHtml +
                "<div class='kg-tooltip-hint'>" + (pinned ? "Pinned beside the clicked graph item. Select text or use Copy buttons. Click × or empty graph space to close." : "Click to pin. Pinned tooltips allow selecting/copying values.") + "</div>";
            tooltip.style.display = "block";
            positionTooltip(position);
            attachTooltipCopyHandlers();
            const closeButton = tooltip.querySelector(".kg-tooltip-close");
            if (closeButton) {{
                closeButton.onclick = function (event) {{
                    event.stopPropagation();
                    hideTooltip(true);
                }};
            }}
        }}

        function hideTooltip(force) {{
            if (pinned && !force) return;
            pinned = false;
            tooltip.classList.remove("kg-pinned");
            tooltip.style.display = "none";
        }}

        function fitGraph(animated) {{
            try {{
                network.fit({{animation: animated ? {{duration: 250, easingFunction: "easeInOutQuad"}} : false}});
            }} catch (e) {{}}
        }}

        function clampScale(value) {{
            return Math.max(0.08, Math.min(5.0, value));
        }}

        function panBy(dx, dy, animated) {{
            try {{
                const scale = network.getScale() || 1;
                const view = network.getViewPosition();
                network.moveTo({{
                    position: {{x: view.x + (dx / scale), y: view.y + (dy / scale)}},
                    scale: scale,
                    animation: animated ? {{duration: 120, easingFunction: "easeInOutQuad"}} : false
                }});
                userInteractedWithView = true;
            }} catch (e) {{ console.warn("Pan failed", e); }}
        }}

        function zoomBy(factor, animated) {{
            try {{
                const view = network.getViewPosition();
                const nextScale = clampScale((network.getScale() || 1) * factor);
                network.moveTo({{
                    position: view,
                    scale: nextScale,
                    animation: animated ? {{duration: 120, easingFunction: "easeInOutQuad"}} : false
                }});
                userInteractedWithView = true;
            }} catch (e) {{ console.warn("Zoom failed", e); }}
        }}

        function attachReliableNavigationButtons() {{
            try {{
                const root = networkContainer || document;
                const buttons = root.querySelectorAll(".vis-navigation .vis-button, .vis-button");
                buttons.forEach(function (button) {{
                    if (button.dataset.kgBound === "1") return;
                    button.dataset.kgBound = "1";
                    button.addEventListener("mousedown", function (event) {{
                        event.preventDefault();
                        event.stopPropagation();
                    }}, true);
                    button.addEventListener("click", function (event) {{
                        event.preventDefault();
                        event.stopPropagation();
                        hideTooltip(false);
                        const cls = button.className || "";
                        // Match the visible arrow direction: up moves the view up, down moves it down,
                        // left moves it left, and right moves it right.
                        if (cls.indexOf("vis-up") !== -1) panBy(0, -120, true);
                        else if (cls.indexOf("vis-down") !== -1) panBy(0, 120, true);
                        else if (cls.indexOf("vis-left") !== -1) panBy(-120, 0, true);
                        else if (cls.indexOf("vis-right") !== -1) panBy(120, 0, true);
                        else if (cls.indexOf("vis-zoomIn") !== -1) zoomBy(1.22, true);
                        else if (cls.indexOf("vis-zoomOut") !== -1) zoomBy(1 / 1.22, true);
                        else if (cls.indexOf("vis-zoomExtends") !== -1) fitGraph(true);
                    }}, true);
                }});
            }} catch (e) {{ console.warn("Navigation binding failed", e); }}
        }}

        function attachReliableWheelZoom() {{
            if (!networkContainer || networkContainer.dataset.kgWheelBound === "1") return;
            networkContainer.dataset.kgWheelBound = "1";
            networkContainer.addEventListener("wheel", function (event) {{
                event.preventDefault();
                event.stopPropagation();
                hideTooltip(false);
                const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
                zoomBy(factor, false);
            }}, {{passive: false, capture: true}});
        }}

        function attachReliableBackgroundDrag() {{
            if (!networkContainer || networkContainer.dataset.kgDragBound === "1") return;
            networkContainer.dataset.kgDragBound = "1";
            let dragging = false;
            let last = null;
            networkContainer.addEventListener("mousedown", function (event) {{
                if (event.button !== 0) return;
                if (event.target && event.target.closest && event.target.closest(".vis-button, .kg-legend, #kg-tooltip, a, button, input, select, textarea")) return;
                try {{
                    const rect = networkContainer.getBoundingClientRect();
                    const domPoint = {{x: event.clientX - rect.left, y: event.clientY - rect.top}};
                    if (network.getNodeAt(domPoint)) return;  // keep native node dragging
                }} catch (e) {{}}
                dragging = true;
                last = {{x: event.clientX, y: event.clientY}};
                userInteractedWithView = true;
                hideTooltip(false);
                event.preventDefault();
                event.stopPropagation();
            }}, true);
            window.addEventListener("mousemove", function (event) {{
                if (!dragging || !last) return;
                const dx = last.x - event.clientX;
                const dy = last.y - event.clientY;
                last = {{x: event.clientX, y: event.clientY}};
                panBy(dx, dy, false);
                event.preventDefault();
            }}, true);
            window.addEventListener("mouseup", function () {{
                dragging = false;
                last = null;
            }}, true);
        }}

        enforceNodeColors();
        try {{
            network.setOptions({{
                interaction: {{
                    hover: true,
                    tooltipDelay: 0,
                    navigationButtons: true,
                    keyboard: {{enabled: true, bindToWindow: false}},
                    dragNodes: true,
                    dragView: true,
                    zoomView: true,
                    selectable: true,
                    multiselect: false,
                    hoverConnectedEdges: true,
                    selectConnectedEdges: false
                }}
            }});
        }} catch (e) {{}}

        attachReliableNavigationButtons();
        attachReliableWheelZoom();
        attachReliableBackgroundDrag();
        try {{
            if (networkContainer) {{
                networkContainer.style.pointerEvents = "auto";
                const canvas = networkContainer.querySelector("canvas");
                if (canvas) canvas.style.pointerEvents = "auto";
            }}
            network.redraw();
        }} catch (e) {{}}
        window.setTimeout(function () {{
            attachReliableNavigationButtons();
            attachReliableWheelZoom();
            attachReliableBackgroundDrag();
            try {{ network.redraw(); }} catch (e) {{}}
        }}, 250);

        network.on("hoverNode", function (params) {{
            if (!pinned) renderTooltip(nodes.get(params.node), graphLocalPosition(params, "node") || eventPosition(sourceEvent(params)), false);
        }});
        network.on("blurNode", function () {{ hideTooltip(false); }});
        network.on("hoverEdge", function (params) {{
            if (!pinned) renderTooltip(edges.get(params.edge), graphLocalPosition(params, "edge") || eventPosition(sourceEvent(params)), false);
        }});
        network.on("blurEdge", function () {{ hideTooltip(false); }});
        network.on("click", function (params) {{
            if (params.nodes && params.nodes.length > 0) {{
                renderTooltip(nodes.get(params.nodes[0]), graphLocalPosition(params, "node"), true);
            }} else if (params.edges && params.edges.length > 0) {{
                renderTooltip(edges.get(params.edges[0]), graphLocalPosition(params, "edge"), true);
            }} else {{
                hideTooltip(true);
            }}
        }});
        network.on("dragStart", function () {{ userInteractedWithView = true; hideTooltip(false); }});
        network.on("zoom", function () {{ userInteractedWithView = true; hideTooltip(false); }});
        network.once("stabilizationIterationsDone", function () {{
            if (!fittedAfterStabilization) {{
                fittedAfterStabilization = true;
                enforceNodeColors();
                if (!userInteractedWithView) fitGraph(false);
                network.stopSimulation();
            }}
        }});
        window.addEventListener("resize", function () {{
            if (!userInteractedWithView) window.setTimeout(function () {{ fitGraph(true); }}, 180);
        }});
        window.setTimeout(function () {{ if (!userInteractedWithView) fitGraph(false); }}, 220);
    }})();
    </script>
    """

    with tempfile.NamedTemporaryFile(
        prefix="pring-pyvis-",
        suffix=".html",
        delete=False,
    ) as temp_file:
        output_path = Path(temp_file.name)
    try:
        net.save_graph(str(output_path))
        graph_html = output_path.read_text(encoding="utf-8")
    finally:
        output_path.unlink(missing_ok=True)

    additions = style + legend_html + custom_js
    graph_html = graph_html.replace("</body>", additions + "</body>") if "</body>" in graph_html else graph_html + additions
    export_payload = _graph_export_payload(graph)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download graph JSON",
            data=json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=_safe_filename(action, "_graph.json"),
            mime="application/json",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download graph HTML",
            data=graph_html.encode("utf-8"),
            file_name=_safe_filename(action, "_graph.html"),
            mime="text/html",
            use_container_width=True,
        )
    components.html(graph_html, height=780, scrolling=False)

def display_table(graph) -> None:
    """Display collapsible node/relationship tables plus a branded ZIP export."""
    if not graph:
        st.warning("No data to display.")
        return

    tables = graph_to_dataframes(graph)
    if not tables:
        st.info("The query returned no tabular data for the current selection.")
        return

    zip_data = _dataframes_zip_bytes(tables, package_title="CYP450-KG Explorer tabular data export")
    st.download_button(
        label="Download tables ZIP",
        data=zip_data,
        file_name="pring_tabular_data.zip",
        mime="application/zip",
        use_container_width=True,
    )

    total_rows = sum(len(df) for df in tables.values())
    st.markdown(
        f"""
        <div class="kg-table-intro">
            <strong>Tabular data preview</strong>
            <span>{len(tables)} table sections · {total_rows:,} total rows. Expand a node or relationship type to inspect exact properties. For interpretation, use the Summary Report tab.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for table_name, df in tables.items():
        row_count, col_count = df.shape
        section_label = f"{table_name} · {row_count:,} rows · {col_count:,} columns"
        with st.expander(section_label, expanded=False):
            if df.empty:
                st.info("No rows.")
            else:
                height = min(520, max(180, 34 * (len(df) + 1)))
                st.dataframe(df, use_container_width=True, height=height)


def _counter_table(counter: Counter, headers: tuple[str, str]) -> str:
    """Render a small HTML count table."""
    if not counter:
        return "<p class='muted'>No records.</p>"
    rows = "".join(f"<tr><td>{escape(str(key))}</td><td>{value}</td></tr>" for key, value in counter.most_common())
    return f"<table><thead><tr><th>{escape(headers[0])}</th><th>{escape(headers[1])}</th></tr></thead><tbody>{rows}</tbody></table>"


def _selected_html(title: str, values: list[str] | None) -> str:
    """Render selected compounds/targets into HTML."""
    if not values:
        return f"<div class='selected-card'><strong>{escape(title)}</strong><span>Not specified</span></div>"
    items = "".join(f"<li>{escape(str(v))}</li>" for v in values)
    return f"<div class='selected-card'><strong>{escape(title)}</strong><ul>{items}</ul></div>"


def _relationship_preview_html(nodes: dict[str, dict[str, Any]], edges: list[tuple[str, str, str, dict[str, Any]]], limit: int | None = None) -> str:
    """Render relationship rows for the report without truncating text values."""
    if not edges:
        return "<p class='muted'>No relationships were returned.</p>"
    visible_edges = edges if limit is None else edges[:limit]
    rows = []
    for start, end, rel_type, props in visible_edges:
        sp = nodes.get(start, {})
        ep = nodes.get(end, {})
        sl = _primary_label(sp) if sp else "Unknown"
        el = _primary_label(ep) if ep else "Unknown"
        score = _first_non_empty(props or {}, ["score", "edge_weight", "tanimoto", "similarity", "activity_value", "standard_value"], default="")
        rows.append(
            "<tr>"
            f"<td>{escape(_node_full_display_label(sl, sp) if sp else start)}</td>"
            f"<td>{escape(sl)}</td>"
            f"<td><code>{escape(rel_type)}</code></td>"
            f"<td>{escape(_node_full_display_label(el, ep) if ep else end)}</td>"
            f"<td>{escape(el)}</td>"
            f"<td>{escape(score)}</td>"
            "</tr>"
        )
    note = ""
    if limit is not None and len(edges) > limit:
        note = f"<p class='muted'>Showing first {min(limit, len(edges))} of {len(edges)} relationships.</p>"
    return (
        "<div class='table-scroll'><table><thead><tr><th>Source</th><th>Source type</th><th>Relationship</th><th>Target</th><th>Target type</th><th>Key value</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>{note}"
    )


def _dataframe_to_report_table(df: pd.DataFrame) -> str:
    """Render a complete DataFrame as a report-safe HTML table."""
    if df.empty:
        return "<p class='muted'>No rows.</p>"
    header = "".join(f"<th>{escape(str(col))}</th>" for col in df.columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for value in row.tolist():
            if pd.isna(value):
                value = ""
            cells.append(f"<td>{escape(str(value))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table-scroll'><table class='full-data-table'><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"


def _full_report_tables_html(graph) -> str:
    """Render every node/relationship table inside the report, with all rows and columns."""
    tables = graph_to_dataframes(graph)
    if not tables:
        return "<p class='muted'>No tabular data was returned.</p>"
    sections = []
    for table_name, df in tables.items():
        rows, cols = df.shape
        sections.append(
            "<div class='full-table-block'>"
            f"<h3>{escape(table_name)}</h3>"
            f"<p class='muted'>{rows} row(s), {cols} column(s). This table is shown in full; use horizontal scrolling if needed.</p>"
            f"{_dataframe_to_report_table(df)}"
            "</div>"
        )
    return "".join(sections)


def _action_specific_findings(action: str, label_counts: Counter, rel_counts: Counter) -> str:
    """Generate action-aware findings from graph counts using interpretive wording."""
    if action == "Show Similar Compounds":
        return (
            f"<li>The result is a <strong>chemical-neighborhood view</strong>: {label_counts.get('Compound', 0)} compound nodes are connected by "
            f"{rel_counts.get('SIMILAR_TO', 0)} similarity relationships.</li>"
            "<li>Similarity edges should be interpreted as <strong>candidate analogues or nearest neighbors</strong>, not as direct CYP450 activity evidence unless the same compounds also have assay or interaction support.</li>"
        )
    if action == "Show Related BioAssays":
        return (
            f"<li>The graph traces assay evidence from standardized compounds through substances, endpoints, measure groups, and BioAssays. It includes "
            f"<strong>{label_counts.get('BioAssay', 0)}</strong> assays, <strong>{label_counts.get('MeasureGrp', 0)}</strong> measure groups, and "
            f"<strong>{label_counts.get('Endpoint', 0)}</strong> endpoints.</li>"
            "<li>For interpretation, start at the Compound node, follow the standardized Substance and Endpoint nodes, then use MeasureGrp and BioAssay nodes to understand the experimental context.</li>"
        )
    if action == "Co-Occurrence in Literature (Compound-Compound)":
        return (
            "<li>This view connects compounds through shared evidence contexts such as the same reference or assay. It is useful for discovering compounds that are experimentally or bibliographically related.</li>"
            "<li>A shared context is <strong>supporting context</strong>, not necessarily proof that two compounds have the same activity profile.</li>"
        )
    if action in {"Evidence Co-Occurrence (Compound-Target)", "Compound-Target Interactions (PubChem)"}:
        return (
            f"<li>The result centers on <strong>{label_counts.get('Interaction', 0)}</strong> Interaction assertion nodes connecting "
            f"<strong>{label_counts.get('Compound', 0)}</strong> compounds to <strong>{label_counts.get('Protein', 0)}</strong> protein targets.</li>"
            f"<li>Evidence support in this subgraph includes <strong>{label_counts.get('BioAssay', 0)}</strong> BioAssays, "
            f"<strong>{label_counts.get('Endpoint', 0)}</strong> endpoints, and <strong>{label_counts.get('Reference', 0)}</strong> references. Higher support coverage usually makes the assertion easier to audit.</li>"
        )
    if action == "Compound-Target Enrichment Context":
        return (
            "<li>This is a <strong>context expansion</strong> around compound-target interactions. It combines interaction evidence with compound features and protein annotations.</li>"
            f"<li>The result includes {label_counts.get('Properties', 0)} property nodes, {label_counts.get('Structure', 0)} structure nodes, "
            f"{label_counts.get('GO', 0)} GO annotations, {label_counts.get('Reactome', 0)} Reactome nodes, and {label_counts.get('InterPro', 0)} InterPro domains.</li>"
            "<li>Use these enrichment layers as biological/chemical context or modelling features, not as standalone proof of activity.</li>"
        )
    return (
        f"<li>The returned graph contains <strong>{sum(label_counts.values())}</strong> nodes and "
        f"<strong>{sum(rel_counts.values())}</strong> relationships.</li>"
        "<li>Inspect the role explanations below to understand what each node and relationship type contributes to the evidence story.</li>"
    )


def _graph_reading_guide_html(action: str) -> str:
    """Return a plain-language reading guide tailored to the selected action."""
    if action == "Show Similar Compounds":
        steps = [
            "Start with the selected Compound node.",
            "Follow SIMILAR_TO edges to identify chemical neighbors or analogues.",
            "Check similarity-related edge properties such as score, tanimoto, similarity, or edge_weight when available.",
            "Use the neighbors as candidates for further CYP450 evidence lookup rather than treating them as confirmed interactions.",
        ]
    elif action == "Show Related BioAssays":
        steps = [
            "Start from the Compound and Substance nodes to confirm the PubChem standardization path.",
            "Move to Endpoint nodes to understand what was measured, such as IC50, inhibition, activity, or another readout.",
            "Use MeasureGrp nodes to inspect the measurement grouping or condition context.",
            "Use BioAssay, Source, Organism, and Reference nodes to check provenance, assay context, and biological relevance.",
        ]
    elif action == "Co-Occurrence in Literature (Compound-Compound)":
        steps = [
            "Start with the selected compound and identify the second compound reached through shared evidence.",
            "Look at the intermediate BioAssay or Reference node to understand why the two compounds are connected.",
            "Treat the relation as shared context unless direct endpoint or interaction evidence is also present.",
        ]
    elif action in {"Evidence Co-Occurrence (Compound-Target)", "Compound-Target Interactions (PubChem)"}:
        steps = [
            "Start at the Interaction node because it represents the compound-target assertion.",
            "Follow ASSERTS_CHEMICAL to identify the compound and ASSERTS_TARGET to identify the CYP450 protein.",
            "Inspect SUPPORTED_BY_ENDPOINT, SUPPORTED_BY_ASSAY, and SUPPORTED_BY_REFERENCE to understand the evidence behind the assertion.",
            "Check endpoint values/classes and organism context before interpreting the interaction as biologically meaningful for the intended use case.",
        ]
    elif action == "Compound-Target Enrichment Context":
        steps = [
            "First confirm the base compound-target Interaction path.",
            "Use compound feature nodes to interpret chemical structure, properties, synonyms, and molecular representation.",
            "Use protein annotation nodes to interpret biological function, pathway involvement, domains, and structure context.",
            "Separate evidence nodes from feature/enrichment nodes when preparing ML datasets or thesis figures.",
        ]
    else:
        steps = [
            "Identify the central nodes selected by the query.",
            "Follow relationship labels to understand whether each edge is evidence, provenance, similarity, or enrichment context.",
            "Use the exported tables to inspect complete properties that may be hidden in the graph labels.",
        ]
    return "<ol>" + "".join(f"<li>{escape(step)}</li>" for step in steps) + "</ol>"


def _evidence_quality_html(action: str, label_counts: Counter, rel_counts: Counter) -> str:
    """Provide interpretive evidence-quality notes based on returned node/edge composition."""
    notes: list[str] = []
    if rel_counts.get("SUPPORTED_BY_ENDPOINT", 0):
        notes.append("Endpoint support is present, so inspect endpoint properties to understand the measured activity/readout and any reported value, unit, or activity class.")
    if rel_counts.get("SUPPORTED_BY_ASSAY", 0) or label_counts.get("BioAssay", 0):
        notes.append("Assay support is present, which helps connect the graph result to an experimental PubChem BioAssay context.")
    if rel_counts.get("SUPPORTED_BY_REFERENCE", 0) or label_counts.get("Reference", 0):
        notes.append("Reference support is present, giving provenance that can be cited or manually reviewed.")
    if label_counts.get("Organism", 0):
        notes.append("Organism context is available. Check it carefully when focusing on human CYP450 interpretation.")
    if action == "Show Similar Compounds" and not rel_counts.get("ASSERTS_TARGET", 0):
        notes.append("This result is similarity-only unless additional interaction/evidence edges are also present; use it for hypothesis generation.")
    if action == "Compound-Target Enrichment Context":
        notes.append("Enrichment nodes provide features and biological context; they should be separated from direct evidence when explaining model inputs or validating interactions.")
    if not notes:
        notes.append("The returned subgraph is sparse. Use the tables and graph JSON export to inspect whether relevant evidence properties were returned but not visible in node labels.")
    return "<ul>" + "".join(f"<li>{escape(note)}</li>" for note in notes) + "</ul>"


def _node_role_summary_html(label_counts: Counter) -> str:
    """Render node counts with plain-language explanations."""
    if not label_counts:
        return "<p class='muted'>No node roles were returned.</p>"
    rows = []
    for label, count in label_counts.most_common():
        rows.append(
            "<tr>"
            f"<td><span class='dot' style='background:{_node_color(label)}'></span><strong>{escape(label)}</strong></td>"
            f"<td>{count}</td>"
            f"<td>{escape(NODE_TYPE_EXPLANATIONS.get(label, NODE_TYPE_EXPLANATIONS['Node']))}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Node type</th><th>Count</th><th>How to interpret it</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _relationship_meaning_html(rel_counts: Counter) -> str:
    """Render relationship counts with interpretation."""
    if not rel_counts:
        return "<p class='muted'>No relationship types were returned.</p>"
    rows = []
    for rel_type, count in rel_counts.most_common():
        rows.append(
            "<tr>"
            f"<td><code>{escape(rel_type)}</code></td>"
            f"<td>{count}</td>"
            f"<td>{escape(RELATIONSHIP_EXPLANATIONS.get(rel_type, 'Graph relationship returned by the query. Inspect its source/target nodes and properties for interpretation.'))}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Relationship type</th><th>Count</th><th>What it means</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _evidence_path_status(rel_counts: Counter, required_rels: list[str]) -> tuple[str, str]:
    """Return a compact observed/missing status for an expected evidence path."""
    if not required_rels:
        return "Context path", "This path is conceptual for the selected action; inspect the relationship table for exact edges."
    present = [rel for rel in required_rels if rel_counts.get(rel, 0)]
    missing = [rel for rel in required_rels if not rel_counts.get(rel, 0)]
    if not missing:
        return "Visible in returned graph", "All key relationship types for this path were returned in the current subgraph."
    if present:
        return "Partially visible", "Visible: " + ", ".join(present) + ". Not returned here: " + ", ".join(missing) + "."
    return "Not visible in this result", "The key relationship types for this path were not returned by the current query selection."


def _evidence_path_map_html(action: str, label_counts: Counter, rel_counts: Counter) -> str:
    """Explain the expected evidence paths for the selected action in plain language."""
    path_specs: list[dict[str, Any]]
    if action == "Show Similar Compounds":
        path_specs = [
            {
                "name": "Chemical similarity path",
                "path": "Compound ─[SIMILAR_TO]→ Compound",
                "required": ["SIMILAR_TO"],
                "meaning": "This path means two compounds are close according to a similarity or neighbourhood calculation. It is useful for analogue discovery but is not direct assay evidence.",
                "read": "Start with the selected compound and follow SIMILAR_TO edges to candidate neighbours. Then run an assay or interaction action for the neighbours before making CYP450 conclusions.",
            }
        ]
    elif action == "Show Related BioAssays":
        path_specs = [
            {
                "name": "Assay-to-compound provenance path",
                "path": "BioAssay ─[HAS_MEASURE_GROUP]→ MeasureGrp ─[HAS_ENDPOINT]→ Endpoint ─[ABOUT_SUBSTANCE]→ Substance ─[STANDARDIZED_TO]→ Compound",
                "required": ["HAS_MEASURE_GROUP", "HAS_ENDPOINT", "ABOUT_SUBSTANCE", "STANDARDIZED_TO"],
                "meaning": "This is the core PubChem evidence trail. It shows that a measured endpoint belongs to an assay/measure group and refers to a submitted substance that maps to the selected standardized compound.",
                "read": "Read from BioAssay to Compound when auditing provenance, or from Compound back to Endpoint/BioAssay when asking what was measured for a selected compound.",
            },
            {
                "name": "Explicit compound-target assertion path when available",
                "path": "Compound ←[ASSERTS_CHEMICAL]─ Interaction ─[ASSERTS_TARGET]→ Protein, with Interaction ─[SUPPORTED_BY_ENDPOINT / SUPPORTED_BY_ASSAY]→ evidence",
                "required": ["ASSERTS_CHEMICAL", "ASSERTS_TARGET"],
                "meaning": "This path is stronger than assay context alone because it explicitly connects the selected compound to a CYP450 protein through an Interaction node. Endpoint or assay support explains why the assertion exists.",
                "read": "Use this path to separate direct compound-CYP450 assertions from broader assay records that mention the compound but may not assert the selected target.",
            },
        ]
    elif action == "Co-Occurrence in Literature (Compound-Compound)":
        path_specs = [
            {
                "name": "Shared evidence-context path",
                "path": "Compound A → evidence context ← Compound B, commonly through shared Interaction, BioAssay, Endpoint, or Reference neighbourhoods",
                "required": [],
                "meaning": "Co-occurrence means two compounds are connected through the same returned evidence neighbourhood. It suggests shared context, not necessarily shared activity or potency.",
                "read": "Look for the intermediate node type. A shared Reference means bibliographic context; a shared BioAssay/Endpoint means experimental context; a shared Protein/Interaction means target-neighbourhood context.",
            }
        ]
    elif action in {"Evidence Co-Occurrence (Compound-Target)", "Compound-Target Interactions (PubChem)"}:
        path_specs = [
            {
                "name": "Direct compound-target assertion path",
                "path": "Compound ←[ASSERTS_CHEMICAL]─ Interaction ─[ASSERTS_TARGET]→ Protein",
                "required": ["ASSERTS_CHEMICAL", "ASSERTS_TARGET"],
                "meaning": "This is the main path to look for. The Interaction node is the explicit assertion that a compound and CYP450 protein are connected in the PRING graph.",
                "read": "Do not interpret the Compound and Protein as directly connected unless this Interaction bridge is present. Use the Interaction properties and support edges for evidence strength.",
            },
            {
                "name": "Endpoint support path",
                "path": "Interaction ─[SUPPORTED_BY_ENDPOINT]→ Endpoint ←[HAS_ENDPOINT]─ MeasureGrp ←[HAS_MEASURE_GROUP]─ BioAssay",
                "required": ["SUPPORTED_BY_ENDPOINT"],
                "meaning": "Endpoint support connects the assertion to a measured readout such as activity, inhibition, IC50, or another assay value. This is the most important place to inspect values, units, and activity class.",
                "read": "Use this path to decide whether the interaction can become a modelling label after thresholding/curation.",
            },
            {
                "name": "Assay/reference support path",
                "path": "Interaction ─[SUPPORTED_BY_ASSAY]→ BioAssay and/or Interaction ─[SUPPORTED_BY_REFERENCE]→ Reference",
                "required": ["SUPPORTED_BY_ASSAY", "SUPPORTED_BY_REFERENCE"],
                "meaning": "Assay and reference support improve auditability by explaining where the assertion came from. They do not replace endpoint-level interpretation.",
                "read": "Use assay/reference support to trace provenance and cite evidence, then validate endpoint details before biological conclusions.",
            },
        ]
    elif action == "Compound-Target Enrichment Context":
        path_specs = [
            {
                "name": "Base evidence path to confirm first",
                "path": "Compound ←[ASSERTS_CHEMICAL]─ Interaction ─[ASSERTS_TARGET]→ Protein",
                "required": ["ASSERTS_CHEMICAL", "ASSERTS_TARGET"],
                "meaning": "This confirms that enrichment nodes are being interpreted around a real selected compound-target assertion rather than as isolated features.",
                "read": "Start here before using feature nodes for explanation or ML inputs.",
            },
            {
                "name": "Compound feature branches",
                "path": "Compound → Properties / Structure / Synonyms / MolGraph / Neighbors",
                "required": [],
                "meaning": "These branches explain chemical identity and modelling features. They support feature engineering but do not prove CYP450 activity.",
                "read": "Use them as model features or report context after direct evidence has been checked.",
            },
            {
                "name": "Protein annotation branches",
                "path": "Protein → UniProt / GO / Reactome / InterPro / PDB / AlphaFold / ProtEmbed",
                "required": [],
                "meaning": "These branches explain target biology, pathways, domains, structures, and embeddings. They provide biological context around the CYP450 target.",
                "read": "Use them to interpret target function and feature availability; do not treat them as compound-specific activity evidence.",
            },
        ]
    else:
        path_specs = [
            {
                "name": "Returned graph path",
                "path": "Use the relationship labels in the graph and tables to follow source → relationship → target.",
                "required": [],
                "meaning": "The selected action returned a graph path that should be interpreted by relationship type and node role.",
                "read": "Start from selected Compound/Protein nodes, then follow each relationship label and inspect complete properties in the tables.",
            }
        ]

    cards = []
    for spec in path_specs:
        status, status_text = _evidence_path_status(rel_counts, spec.get("required", []))
        count_items = []
        for rel in spec.get("required", []):
            count_items.append(f"<li><code>{escape(rel)}</code>: {rel_counts.get(rel, 0)}</li>")
        counts_html = "<ul class='path-counts'>" + "".join(count_items) + "</ul>" if count_items else "<p class='muted'>This path is interpreted from the returned node/relationship context.</p>"
        cards.append(
            "<div class='evidence-path-card'>"
            f"<div class='path-card-head'><h3>{escape(spec['name'])}</h3><span>{escape(status)}</span></div>"
            f"<div class='path-expression'>{escape(spec['path'])}</div>"
            f"<p><strong>Meaning:</strong> {escape(spec['meaning'])}</p>"
            f"<p><strong>How to read it:</strong> {escape(spec['read'])}</p>"
            f"<p class='muted'>{escape(status_text)}</p>"
            f"{counts_html}"
            "</div>"
        )
    node_summary = (
        f"Returned node context: {label_counts.get('Compound', 0)} Compound, "
        f"{label_counts.get('Interaction', 0)} Interaction, {label_counts.get('Protein', 0)} Protein, "
        f"{label_counts.get('Endpoint', 0)} Endpoint, {label_counts.get('BioAssay', 0)} BioAssay, "
        f"and {label_counts.get('Reference', 0)} Reference nodes."
    )
    return (
        "<div class='evidence-path-intro'>"
        "<p>The cards below translate the Neo4j paths into plain language. They explain which path is direct evidence, which path is provenance, and which path is only contextual enrichment.</p>"
        f"<p class='muted'>{escape(node_summary)}</p>"
        "</div>"
        "<div class='evidence-path-grid'>" + "".join(cards) + "</div>"
    )


def _entity_examples_html(nodes: dict[str, dict[str, Any]], limit_per_type: int = 4) -> str:
    """Show representative entities from each node type for easier report reading."""
    if not nodes:
        return "<p class='muted'>No representative entities were returned.</p>"
    grouped: dict[str, list[str]] = {}
    for props in nodes.values():
        label = _primary_label(props)
        grouped.setdefault(label, [])
        if len(grouped[label]) < limit_per_type:
            grouped[label].append(_node_full_display_label(label, props))
    cards = []
    for label in sorted(grouped):
        examples = "".join(f"<li>{escape(v)}</li>" for v in grouped[label])
        cards.append(
            f"<div class='example-card'><h3><span class='dot' style='background:{_node_color(label)}'></span>{escape(label)}</h3><ul>{examples}</ul></div>"
        )
    return "<div class='examples-grid'>" + "".join(cards) + "</div>"


def _ranked_relationships_html(nodes: dict[str, dict[str, Any]], edges: list[tuple[str, str, str, dict[str, Any]]], limit: int = 10) -> str:
    """Show high-signal relationships using score/value-like properties when available."""
    score_keys = ["score", "edge_weight", "tanimoto", "similarity", "activity_value", "standard_value", "value", "pchembl_value"]
    ranked = []
    for start, end, rel_type, props in edges:
        props = props or {}
        key = next((k for k in score_keys if props.get(k) not in (None, "", [], {})), None)
        if not key:
            continue
        try:
            sort_value = float(props[key])
        except Exception:
            sort_value = 0.0
        ranked.append((sort_value, key, start, end, rel_type, props))
    if not ranked:
        return "<p class='muted'>No score/value-like relationship properties were detected in this returned subgraph. Use the relationship table export for full property inspection.</p>"
    ranked.sort(key=lambda x: x[0], reverse=True)
    rows = []
    for _, key, start, end, rel_type, props in ranked[:limit]:
        sp = nodes.get(start, {})
        ep = nodes.get(end, {})
        sl = _primary_label(sp) if sp else "Unknown"
        el = _primary_label(ep) if ep else "Unknown"
        rows.append(
            "<tr>"
            f"<td>{escape(_node_full_display_label(sl, sp) if sp else start)}</td>"
            f"<td><code>{escape(rel_type)}</code></td>"
            f"<td>{escape(_node_full_display_label(el, ep) if ep else end)}</td>"
            f"<td>{escape(key)}</td>"
            f"<td>{escape(_format_value(props.get(key)))}</td>"
            "</tr>"
        )
    return "<table><thead><tr><th>Source</th><th>Relationship</th><th>Target</th><th>Value field</th><th>Value</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _recommended_next_steps_html(action: str, label_counts: Counter, rel_counts: Counter) -> str:
    """Return practical next checks for a user interpreting this graph result."""
    if action == "Show Similar Compounds":
        steps = [
            "Download the relationship table and sort by similarity/score fields when available.",
            "Run the interaction or BioAssay actions for the top similar compounds to test whether similarity is supported by CYP450 evidence.",
            "Avoid using similarity alone as a positive interaction label for model training.",
        ]
    elif action == "Show Related BioAssays":
        steps = [
            "Inspect endpoint values, units, activity classes, and assay descriptions before deciding whether the evidence is active, inactive, or ambiguous.",
            "Check organism and target context to confirm relevance for the intended CYP450 enzyme case study.",
            "Use the report package ZIP as an evidence audit trail for thesis figures or manual curation.",
        ]
    elif action in {"Evidence Co-Occurrence (Compound-Target)", "Compound-Target Interactions (PubChem)"}:
        steps = [
            "Use the Interaction node as the central evidence assertion and verify its supporting assay/endpoint/reference links.",
            "Compare endpoint properties across interactions before converting them into modelling labels.",
            "Flag interactions without endpoint or assay support as lower-auditability records unless other evidence is available.",
        ]
    elif action == "Compound-Target Enrichment Context":
        steps = [
            "Separate direct evidence tables from enrichment/feature tables when preparing modelling-ready datasets.",
            "Use protein annotation and chemical feature nodes to explain why a prediction model may find a compound-target pair plausible.",
            "Check whether enrichment layers are consistently available across all selected targets to avoid feature missingness bias.",
        ]
    else:
        steps = [
            "Use the graph to understand topology and the tables to inspect complete properties.",
            "Export JSON/HTML for reproducible sharing and CSV ZIP for downstream analysis.",
            "Follow evidence/provenance paths before making biological conclusions.",
        ]
    return "<ol>" + "".join(f"<li>{escape(step)}</li>" for step in steps) + "</ol>"




def _html_list(items: list[str], class_name: str | None = None) -> str:
    """Render a safe HTML unordered list."""
    if not items:
        return "<p class='muted'>No specific items were detected in this result.</p>"
    class_attr = f" class='{escape(class_name)}'" if class_name else ""
    return f"<ul{class_attr}>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"


def _unique_node_names(nodes: dict[str, dict[str, Any]], label: str, limit: int = 6) -> list[str]:
    """Return representative display names for nodes with the requested primary label."""
    seen: set[str] = set()
    names: list[str] = []
    for props in nodes.values():
        current_label = _primary_label(props)
        if current_label != label:
            continue
        name = _node_full_display_label(current_label, props)
        if name and name not in seen:
            seen.add(name)
            names.append(name)
        if len(names) >= limit:
            break
    return names


def _relationship_value_stats(edges: list[tuple[str, str, str, dict[str, Any]]]) -> dict[str, tuple[int, float, float, float]]:
    """Collect count, min, max, and mean for common numeric relationship fields."""
    score_keys = [
        "score",
        "edge_weight",
        "tanimoto",
        "similarity",
        "activity_value",
        "standard_value",
        "value",
        "pchembl_value",
    ]
    values: dict[str, list[float]] = {key: [] for key in score_keys}
    for _, _, _, props in edges:
        for key in score_keys:
            raw = (props or {}).get(key)
            if raw in (None, "", [], {}):
                continue
            try:
                values[key].append(float(raw))
            except (TypeError, ValueError):
                continue
    return {
        key: (len(vals), min(vals), max(vals), sum(vals) / len(vals))
        for key, vals in values.items()
        if vals
    }


def _interaction_pair_count(nodes: dict[str, dict[str, Any]], edges: list[tuple[str, str, str, dict[str, Any]]]) -> int:
    """Estimate unique compound-target assertions represented by Interaction nodes."""
    interaction_to_compounds: dict[str, set[str]] = {}
    interaction_to_targets: dict[str, set[str]] = {}
    for start, end, rel_type, _ in edges:
        start_label = _primary_label(nodes.get(start, {})) if start in nodes else ""
        end_label = _primary_label(nodes.get(end, {})) if end in nodes else ""
        if rel_type == "ASSERTS_CHEMICAL" and start_label == "Interaction" and end_label == "Compound":
            interaction_to_compounds.setdefault(start, set()).add(end)
        elif rel_type == "ASSERTS_TARGET" and start_label == "Interaction" and end_label == "Protein":
            interaction_to_targets.setdefault(start, set()).add(end)
    pairs: set[tuple[str, str]] = set()
    for interaction_id, compounds in interaction_to_compounds.items():
        for compound_id in compounds:
            for target_id in interaction_to_targets.get(interaction_id, set()):
                pairs.add((compound_id, target_id))
    return len(pairs)


def _main_endpoint_phrase(nodes: dict[str, dict[str, Any]]) -> str:
    """Summarize endpoint names present in the returned graph."""
    endpoints = _unique_node_names(nodes, "Endpoint", limit=5)
    if not endpoints:
        return "No explicit Endpoint nodes were returned."
    if len(endpoints) == 1:
        return f"The main measured endpoint visible in the graph is {endpoints[0]}."
    return "Visible endpoint/readout labels include " + ", ".join(endpoints[:4]) + "."


def _evidence_level(action: str, label_counts: Counter, rel_counts: Counter) -> tuple[str, str, str]:
    """Return a simple evidence-level label, CSS class, and explanation."""
    has_interaction = label_counts.get("Interaction", 0) > 0
    has_endpoint = rel_counts.get("SUPPORTED_BY_ENDPOINT", 0) > 0 or label_counts.get("Endpoint", 0) > 0
    has_assay = rel_counts.get("SUPPORTED_BY_ASSAY", 0) > 0 or label_counts.get("BioAssay", 0) > 0
    has_reference = rel_counts.get("SUPPORTED_BY_REFERENCE", 0) > 0 or label_counts.get("Reference", 0) > 0

    if action == "Show Similar Compounds":
        return (
            "Hypothesis-generating",
            "badge-hypothesis",
            "This result points to chemically related compounds. It is useful for prioritization, but it is not direct activity evidence.",
        )
    if action == "Compound-Target Enrichment Context":
        return (
            "Context / feature expansion",
            "badge-context",
            "This result explains chemical and biological context around the selected entities. Treat it as feature/context information unless direct evidence edges are also present.",
        )
    if has_interaction and has_endpoint and (has_assay or has_reference):
        return (
            "Evidence-backed interaction view",
            "badge-strong",
            "The graph includes direct interaction assertions and supporting endpoint, assay, or reference provenance, so it is suitable for manual evidence review.",
        )
    if has_interaction and has_endpoint:
        return (
            "Partial interaction evidence",
            "badge-medium",
            "The graph includes interaction and endpoint support, but assay/reference provenance is limited in the returned subgraph.",
        )
    if has_assay or has_endpoint:
        return (
            "Assay evidence view",
            "badge-medium",
            "The graph contains measurement or assay context. Inspect endpoint values and units before drawing activity conclusions.",
        )
    return (
        "Exploratory graph view",
        "badge-context",
        "The graph mainly shows context or topology. Use the exported tables to inspect complete properties and provenance.",
    )


def _simple_result_summary(
    action: str,
    nodes: dict[str, dict[str, Any]],
    edges: list[tuple[str, str, str, dict[str, Any]]],
    label_counts: Counter,
    rel_counts: Counter,
) -> dict[str, Any]:
    """Create action-aware plain-language findings for non-technical interpretation."""
    node_total = sum(label_counts.values())
    edge_total = sum(rel_counts.values())
    compounds = label_counts.get("Compound", 0)
    proteins = label_counts.get("Protein", 0)
    interactions = label_counts.get("Interaction", 0)
    endpoints = label_counts.get("Endpoint", 0)
    assays = label_counts.get("BioAssay", 0)
    references = label_counts.get("Reference", 0)
    measure_groups = label_counts.get("MeasureGrp", 0)
    substances = label_counts.get("Substance", 0)
    organisms = label_counts.get("Organism", 0)
    pair_count = _interaction_pair_count(nodes, edges)
    endpoint_phrase = _main_endpoint_phrase(nodes)
    protein_names = _unique_node_names(nodes, "Protein", limit=5)
    compound_examples = _unique_node_names(nodes, "Compound", limit=3)
    score_stats = _relationship_value_stats(edges)

    summary: dict[str, Any] = {
        "headline": f"This query returned {node_total} nodes and {edge_total} relationships from the PRING Neo4j graph.",
        "supports": [],
        "does_not_prove": [],
        "cards": [],
    }

    if action == "Show Similar Compounds":
        sim_edges = rel_counts.get("SIMILAR_TO", 0)
        summary["headline"] = (
            f"The selected compounds have a chemical-neighborhood result: {compounds} compound nodes are linked by {sim_edges} SIMILAR_TO relationships."
        )
        summary["supports"] = [
            "Finding compounds that are structurally or feature-wise close to the selected compounds.",
            "Prioritizing analogues for follow-up evidence lookup, especially when direct CYP450 data are sparse.",
            "Building candidate negative/unknown pairs or neighborhood features for graph-based modelling, after careful curation.",
        ]
        summary["does_not_prove"] = [
            "A SIMILAR_TO edge does not mean that the neighbour compound has the same CYP450 activity.",
            "Similarity alone should not be used as a positive interaction label.",
            "Potency, inhibition strength, or selectivity still need endpoint or assay evidence.",
        ]
        value_text = "No numeric similarity score was detected in the returned relationship properties."
        for key in ["score", "tanimoto", "similarity", "edge_weight"]:
            if key in score_stats:
                count, min_v, max_v, mean_v = score_stats[key]
                value_text = f"Detected {count} {key} values; range {min_v:.3g} to {max_v:.3g}, average {mean_v:.3g}."
                break
        summary["cards"] = [
            ("Returned neighbourhood", f"{compounds} compounds and {sim_edges} similarity links."),
            ("How to use it", "Treat neighbours as candidates for follow-up interaction or assay searches."),
            ("Similarity scores", value_text),
        ]
    elif action == "Show Related BioAssays":
        summary["headline"] = (
            f"The graph traces assay evidence for the selected compounds through {substances} PubChem substances, {endpoints} endpoints, {measure_groups} measure groups, and {assays} BioAssays."
        )
        summary["supports"] = [
            "Checking whether there is assay-level evidence behind the selected compounds.",
            "Following the provenance chain from submitted Substance records to standardized Compound records.",
            "Understanding what was measured before using the evidence for CYP450 interpretation or modelling.",
        ]
        summary["does_not_prove"] = [
            "Assay presence alone does not say whether the compound is active or inactive.",
            "Endpoint labels such as IC50 still need value, unit, threshold, and assay-condition inspection.",
            "Different assays may not be directly comparable without normalization and curation.",
        ]
        summary["cards"] = [
            ("Evidence path", "BioAssay → MeasureGrp → Endpoint → Substance → Compound."),
            ("Endpoint coverage", endpoint_phrase),
            ("Auditability", f"{assays} BioAssays and {measure_groups} measurement groups were returned."),
        ]
    elif action in {"Compound-Target Interactions (PubChem)", "Evidence Co-Occurrence (Compound-Target)"}:
        assay_links = rel_counts.get("SUPPORTED_BY_ASSAY", 0)
        endpoint_links = rel_counts.get("SUPPORTED_BY_ENDPOINT", 0)
        reference_links = rel_counts.get("SUPPORTED_BY_REFERENCE", 0)
        target_text = ", ".join(protein_names) if protein_names else f"{proteins} protein target(s)"
        pair_text = pair_count if pair_count else interactions
        summary["headline"] = (
            f"The result contains {interactions} interaction assertion nodes, representing about {pair_text} compound-target assertion(s) involving {target_text}."
        )
        summary["supports"] = [
            "Reviewing direct compound-target assertions stored in the graph.",
            "Checking whether an assertion has endpoint, assay, reference, and organism context.",
            "Preparing auditable candidate labels for CYP450 interaction prediction, after endpoint-level curation.",
        ]
        summary["does_not_prove"] = [
            "An Interaction node should not be treated as a final active/inactive label until endpoint value, unit, direction, and threshold are checked.",
            "A sparse result with no assay or reference links is less auditable than one with full provenance.",
            "Counts show returned graph coverage, not necessarily all evidence available in PubChem or the full database.",
        ]
        if action == "Evidence Co-Occurrence (Compound-Target)" and not (assays or references or assay_links or reference_links):
            summary["does_not_prove"].append(
                "This returned co-occurrence view has no explicit BioAssay or Reference nodes, so interpret it as a partial evidence path rather than a complete provenance record."
            )
        summary["cards"] = [
            ("Interaction assertions", f"{interactions} Interaction nodes; estimated {pair_text} compound-target pair(s)."),
            ("Evidence support", f"{endpoint_links} endpoint links, {assay_links} assay links, {reference_links} reference links."),
            ("Targets visible", target_text),
            ("Endpoint coverage", endpoint_phrase),
        ]
    elif action == "Co-Occurrence in Literature (Compound-Compound)":
        summary["headline"] = (
            f"The result groups {compounds} compounds through shared graph context. It returned {interactions} Interaction nodes, {proteins} Protein nodes, {assays} BioAssays, and {references} References."
        )
        summary["supports"] = [
            "Finding compounds that appear in the same target, assay, reference, or interaction neighbourhood.",
            "Identifying candidate compound groups for manual comparison.",
            "Exploring whether selected compounds share evidence context before deeper endpoint review.",
        ]
        summary["does_not_prove"] = [
            "Co-occurrence does not prove that two compounds have the same potency, mechanism, or CYP450 profile.",
            "A shared target or shared evidence context is weaker than a direct matched endpoint comparison.",
            "If no Reference or BioAssay nodes are returned, the result should not be described as literature co-citation evidence.",
        ]
        if references == 0 and assays == 0:
            summary["cards"] = [
                ("Important caution", "No explicit Reference or BioAssay nodes were returned, so this specific result is not a literature-provenance view."),
                ("Returned context", f"{compounds} compounds, {interactions} interaction nodes, and {proteins} protein node(s)."),
                ("Best use", "Use this as a candidate grouping view, then open interaction and BioAssay reports for confirmation."),
            ]
        else:
            summary["cards"] = [
                ("Shared context", f"{assays} BioAssays and {references} References are visible in this result."),
                ("Compound coverage", f"{compounds} compounds were returned in the shared-context graph."),
                ("Best use", "Compare endpoints and provenance before claiming similar activity."),
            ]
    elif action == "Compound-Target Enrichment Context":
        feature_counts = {
            "molecular property nodes": label_counts.get("Properties", 0),
            "structure nodes": label_counts.get("Structure", 0),
            "synonym nodes": label_counts.get("Synonyms", 0),
            "GO annotations": label_counts.get("GO", 0),
            "Reactome/pathway nodes": label_counts.get("Reactome", 0) + label_counts.get("Pathway", 0),
            "InterPro domains": label_counts.get("InterPro", 0),
            "PDB structures": label_counts.get("PDB", 0),
            "AlphaFold models": label_counts.get("AlphaFold", 0),
            "protein embeddings": label_counts.get("ProtEmbed", 0),
        }
        nonzero_features = [f"{count} {name}" for name, count in feature_counts.items() if count]
        summary["headline"] = (
            f"This is a context-expansion result: it adds chemical features and protein annotations around the selected compound-target space, with {len(label_counts)} node types returned."
        )
        summary["supports"] = [
            "Explaining the chemical and biological context around selected compounds and CYP450 targets.",
            "Finding feature layers that may be useful for graph ML, such as chemical properties, pathways, domains, structures, and embeddings.",
            "Preparing thesis figures that show why the knowledge graph is more than a simple compound-target edge table.",
        ]
        summary["does_not_prove"] = [
            "Enrichment nodes are contextual features; they are not direct evidence that a compound inhibits or activates a target.",
            "Feature availability can be uneven across targets and compounds, so check missingness before modelling.",
            "Pathway/domain/structure annotations explain target biology, not compound-specific activity by themselves.",
        ]
        summary["cards"] = [
            ("Feature layers returned", "; ".join(nonzero_features[:6]) if nonzero_features else "No named enrichment feature layers were detected."),
            ("Direct evidence visible", f"{interactions} Interaction nodes, {endpoints} Endpoint nodes, {assays} BioAssay nodes."),
            ("Best use", "Use as model features and explanatory context after confirming direct evidence paths."),
        ]
    else:
        summary["supports"] = [
            "Understanding which entities and relationships were returned by the selected query.",
            "Inspecting topology before opening the full table exports.",
            "Creating a reproducible audit trail through the HTML and CSV ZIP downloads.",
        ]
        summary["does_not_prove"] = [
            "Graph presence alone does not prove biological causality or activity strength.",
            "Use complete properties and provenance before making scientific conclusions.",
        ]
        summary["cards"] = [
            ("Graph size", f"{node_total} nodes and {edge_total} relationships."),
            ("Node diversity", f"{len(label_counts)} node types."),
            ("Relationship diversity", f"{len(rel_counts)} relationship types."),
        ]

    if compound_examples:
        summary["cards"].append(("Example compounds", ", ".join(compound_examples)))
    if organisms:
        summary["cards"].append(("Organism context", f"{organisms} organism node(s) were returned; check human relevance when interpreting CYP450 results."))
    return summary


def _plain_language_summary_html(
    action: str,
    nodes: dict[str, dict[str, Any]],
    edges: list[tuple[str, str, str, dict[str, Any]]],
    label_counts: Counter,
    rel_counts: Counter,
    report_interpretation: str,
) -> str:
    """Render a richer plain-language report interpretation section."""
    level, level_class, level_text = _evidence_level(action, label_counts, rel_counts)
    summary = _simple_result_summary(action, nodes, edges, label_counts, rel_counts)
    card_html = "".join(
        "<div class='finding-card'>"
        f"<span>{escape(title)}</span>"
        f"<strong>{escape(text)}</strong>"
        "</div>"
        for title, text in summary["cards"][:6]
    )
    return f"""
        <div class="interpretation-panel">
            <div class="takeaway-strip">
                <div>
                    <span class="evidence-badge {escape(level_class)}">{escape(level)}</span>
                    <h3>In simple terms</h3>
                    <p>{escape(summary['headline'])}</p>
                </div>
                <p class="level-note">{escape(level_text)}</p>
            </div>
            <p class="report-purpose-note">{escape(report_interpretation)}</p>
            <div class="finding-grid">{card_html}</div>
            <div class="interpretation-columns">
                <div class="callout callout-support"><h3>What this result supports</h3>{_html_list(summary['supports'])}</div>
                <div class="callout callout-caution"><h3>What it does not prove yet</h3>{_html_list(summary['does_not_prove'])}</div>
            </div>
        </div>
    """

def generate_report_html(
    action: str,
    graph,
    selected_compounds: list[str] | None = None,
    selected_targets: list[str] | None = None,
) -> str:
    """Generate an action-aware HTML report for the current graph result."""
    nodes, edges = extract_graph_data(graph)
    label_counts = Counter(_primary_label(props) for props in nodes.values())
    rel_counts = Counter(edge[2] for edge in edges)
    reading_guide_html = _graph_reading_guide_html(action)
    evidence_quality_html = _evidence_quality_html(action, label_counts, rel_counts)
    evidence_path_html = _evidence_path_map_html(action, label_counts, rel_counts)
    node_role_html = _node_role_summary_html(label_counts)
    relationship_meaning_html = _relationship_meaning_html(rel_counts)
    entity_examples_html = _entity_examples_html(nodes)
    ranked_relationships_html = _ranked_relationships_html(nodes, edges)
    next_steps_html = _recommended_next_steps_html(action, label_counts, rel_counts)
    full_tables_html = _full_report_tables_html(graph)
    report_copy = ACTION_REPORT_COPY.get(
        action,
        {
            "title": f"{action} report",
            "purpose": "Summarizes the current PRING knowledge graph query result.",
            "interpretation": "Use this report together with the graph and exported tables to inspect evidence and provenance.",
        },
    )
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plain_summary_html = _plain_language_summary_html(
        action=action,
        nodes=nodes,
        edges=edges,
        label_counts=label_counts,
        rel_counts=rel_counts,
        report_interpretation=report_copy["interpretation"],
    )

    metrics_html = f"""
    <div class="metrics">
        <div><strong>{len(nodes)}</strong><span>Nodes</span></div>
        <div><strong>{len(edges)}</strong><span>Relationships</span></div>
        <div><strong>{len(label_counts)}</strong><span>Node types</span></div>
        <div><strong>{len(rel_counts)}</strong><span>Relationship types</span></div>
    </div>
    """

    node_color_legend = "".join(
        f"<li><span style='background:{_node_color(label)}'></span>{escape(label)} <em>{count}</em></li>"
        for label, count in sorted(label_counts.items())
    )
    report_brand_logo = _brand_logo_html("report-brand-logo")

    report_html = f"""
    <!doctype html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{escape(report_copy['title'])}</title>
        <style>
            :root {{ --app-charcoal: #505050; --app-mid-gray: #808080; --app-red: #EF4444; --app-bg: #F8FAFC; --app-text: #111827; --app-muted: #64748B; --app-border: #E2E8F0; }}
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 28px; color: var(--app-text); background: linear-gradient(180deg, #F8FAFC 0%, #EEF2F7 100%); overflow-x: auto; }}
            .report {{ max-width: 1280px; margin: 0 auto; background: #FFFFFF; border: 1px solid var(--app-border); border-radius: 18px; box-shadow: 0 14px 40px rgba(17,24,39,.10); overflow: visible; }}
            header {{ padding: 30px 34px; background: linear-gradient(135deg, #505050 0%, #2F343B 58%, #111827 100%); color: #FFFFFF; border-bottom: 5px solid var(--app-red); }}
            header h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: .2px; }}
            header p {{ margin: 0; color: #F3F4F6; line-height: 1.55; }}
            .report-brand-row {{ display: flex; align-items: center; gap: 16px; margin-bottom: 20px; padding-bottom: 18px; border-bottom: 1px solid rgba(255,255,255,.16); }}
            .report-brand-logo {{ width: 72px; height: 72px; object-fit: contain; border-radius: 18px; background: #FFFFFF; border: 1px solid rgba(255,255,255,.35); padding: 8px; box-shadow: 0 10px 26px rgba(0,0,0,.18); }}
            .report-brand-logo-fallback {{ display: grid; place-items: center; font-weight: 900; color: var(--app-red); }}
            .report-brand-kicker {{ margin: 0 0 5px; color: #FCA5A5; font-size: 12px; font-weight: 900; letter-spacing: .14em; text-transform: uppercase; }}
            .report-brand-name {{ margin: 0; color: #FFFFFF; font-size: 25px; font-weight: 900; }}
            .report-brand-tagline {{ margin: 4px 0 0; color: #E5E7EB; font-size: 13.5px; }}
            .report-doi-pill {{ display: inline-flex; align-items: center; gap: 8px; margin-top: 9px; padding: 7px 10px; border-radius: 999px; background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.24); color: #FFFFFF; font-size: 13px; font-weight: 800; text-decoration: none; }}
            .report-doi-pill span {{ background: rgba(255,255,255,.20); color: #FFFFFF; padding: 3px 8px; border-radius: 999px; font-size: 11px; letter-spacing: .04em; }}
            .report-title-block {{ margin-top: 4px; }}
            section {{ padding: 24px 34px; border-top: 1px solid var(--app-border); }}
            h2 {{ margin: 0 0 14px; font-size: 19px; color: var(--app-text); }}
            h2::before {{ content: ""; display: inline-block; width: 8px; height: 18px; margin-right: 9px; border-radius: 999px; background: var(--app-red); vertical-align: -3px; }}
            p, li {{ line-height: 1.65; }}
            .muted {{ color: var(--app-muted); font-size: 13px; }}
            .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 18px; }}
            .metrics div {{ background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.20); border-radius: 14px; padding: 16px; backdrop-filter: blur(4px); }}
            .metrics strong {{ display: block; font-size: 28px; color: #FFFFFF; }}
            .metrics span {{ color: #E5E7EB; font-size: 13px; }}
            .interpretation-panel {{ background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%); border: 1px solid var(--app-border); border-radius: 18px; padding: 18px; box-shadow: 0 8px 24px rgba(17,24,39,.06); }}
            .takeaway-strip {{ display: grid; grid-template-columns: minmax(0, 1.45fr) minmax(240px, .75fr); gap: 16px; align-items: stretch; }}
            .takeaway-strip h3 {{ margin: 10px 0 6px; font-size: 24px; color: #111827; }}
            .takeaway-strip p {{ margin: 0; font-size: 15.5px; color: #334155; }}
            .level-note {{ background: #FFF7ED; border: 1px solid #FED7AA; border-left: 5px solid var(--app-red); border-radius: 14px; padding: 14px; }}
            .evidence-badge {{ display: inline-flex; width: fit-content; align-items: center; border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 700; letter-spacing: .2px; text-transform: uppercase; }}
            .badge-strong {{ background: #DCFCE7; color: #166534; border: 1px solid #86EFAC; }}
            .badge-medium {{ background: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }}
            .badge-hypothesis {{ background: #FFE4E6; color: #9F1239; border: 1px solid #FDA4AF; }}
            .badge-context {{ background: #E0F2FE; color: #075985; border: 1px solid #7DD3FC; }}
            .report-purpose-note {{ margin: 14px 0 0; color: #475569; }}
            .finding-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: 16px; }}
            .finding-card {{ border: 1px solid var(--app-border); border-radius: 14px; padding: 14px; background: #FFFFFF; box-shadow: 0 3px 12px rgba(17,24,39,.04); }}
            .finding-card span {{ display: block; color: #6B7280; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .3px; margin-bottom: 7px; }}
            .finding-card strong {{ display: block; color: #111827; font-size: 14px; line-height: 1.5; }}
            .interpretation-columns, .insight-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }}
            .reading-grid {{ margin-top: 18px; }}
            .evidence-path-intro {{ background:#F8FAFC; border:1px solid var(--app-border); border-radius:14px; padding:14px 16px; margin-bottom:14px; }}
            .evidence-path-intro p {{ margin:0 0 7px; }}
            .evidence-path-intro p:last-child {{ margin-bottom:0; }}
            .evidence-path-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:14px; }}
            .evidence-path-card {{ background:#FFFFFF; border:1px solid var(--app-border); border-left:5px solid var(--app-red); border-radius:16px; padding:16px; box-shadow:0 6px 18px rgba(17,24,39,.05); }}
            .path-card-head {{ display:flex; align-items:flex-start; justify-content:space-between; gap:10px; margin-bottom:10px; }}
            .path-card-head h3 {{ margin:0; font-size:16px; color:#111827; }}
            .path-card-head span {{ flex:0 0 auto; border-radius:999px; padding:5px 8px; background:#F3F4F6; color:#374151; font-size:11px; font-weight:900; text-transform:uppercase; letter-spacing:.03em; }}
            .path-expression {{ font-family:Consolas, 'Courier New', monospace; background:#111827; color:#F9FAFB; border-radius:12px; padding:10px 12px; margin:10px 0 12px; font-size:12.5px; line-height:1.55; overflow-wrap:anywhere; white-space:normal; }}
            .path-counts {{ margin:8px 0 0; padding-left:18px; }}
            .path-counts li {{ margin:4px 0; }}
            .callout {{ background: #FFFFFF; border: 1px solid var(--app-border); border-left: 4px solid var(--app-mid-gray); border-radius: 14px; padding: 16px; }}
            .callout h3 {{ margin: 0 0 8px; font-size: 15px; color: #111827; }}
            .callout ul, .callout ol {{ margin: 0; padding-left: 20px; }}
            .callout-support {{ border-left-color: #22C55E; background: #F0FDF4; }}
            .callout-caution {{ border-left-color: var(--app-red); background: #FFF7ED; }}
            .dot {{ width: 12px; height: 12px; border-radius: 999px; border: 1px solid rgba(17,24,39,.22); display: inline-block; margin-right: 8px; vertical-align: -1px; }}
            .examples-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
            .example-card {{ background: #F8FAFC; border: 1px solid var(--app-border); border-radius: 14px; padding: 12px; }}
            .example-card h3 {{ margin: 0 0 8px; font-size: 15px; color: #111827; }}
            .example-card ul {{ margin: 0; padding-left: 18px; }}
            .selected {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
            .selected-card {{ background: #F8FAFC; border: 1px solid var(--app-border); border-radius: 14px; padding: 14px; }}
            .selected-card strong {{ display: block; margin-bottom: 8px; color: #334155; }}
            .selected-card ul {{ margin: 0; padding-left: 18px; }}
            .selected-card, .selected-card li, .finding-card strong {{ overflow-wrap: anywhere; word-break: break-word; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: #FFFFFF; }}
            th {{ background: #F3F4F6; color: #374151; text-align: left; border-bottom: 2px solid var(--app-mid-gray); }}
            th, td {{ border: 1px solid var(--app-border); padding: 9px 10px; vertical-align: top; overflow-wrap: anywhere; word-break: break-word; white-space: normal; max-width: none; min-width: 120px; }}
            .table-scroll {{ width: 100%; overflow-x: auto; overflow-y: visible; border: 1px solid var(--app-border); border-radius: 12px; background: #FFFFFF; }}
            .table-scroll table {{ min-width: 100%; margin: 0; table-layout: auto; }}
            .full-data-table {{ table-layout: auto; }}
            .full-table-block {{ margin-top: 18px; }}
            .full-table-block h3 {{ margin: 0 0 6px; font-size: 16px; color: #111827; }}
            code {{ background: #F3F4F6; color: #374151; border: 1px solid #E5E7EB; border-radius: 6px; padding: 2px 5px; }}
            .legend {{ list-style: none; padding: 0; margin: 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; }}
            .legend li {{ display: flex; align-items: center; gap: 8px; background: #F8FAFC; border: 1px solid var(--app-border); border-radius: 10px; padding: 7px 9px; }}
            .legend span {{ width: 12px; height: 12px; border-radius: 999px; border: 1px solid rgba(17,24,39,.22); display: inline-block; }}
            .legend em {{ margin-left: auto; color: var(--app-muted); font-style: normal; font-weight: 700; }}
            .license-notice {{ background:#F8FAFC; color:#475569; font-size:12.5px; line-height:1.65; }}
            .license-notice strong {{ color:#111827; }}
            .license-notice a {{ color:#EF4444; font-weight:800; text-decoration:none; }}
            @media (max-width: 860px) {{ .metrics, .selected, .takeaway-strip, .interpretation-columns, .insight-grid {{ grid-template-columns: 1fr; }} body {{ padding: 12px; }} section, header {{ padding: 20px; }} }}
            @media print {{ body {{ background: #FFFFFF; padding: 0; }} .report {{ box-shadow: none; border: 0; }} }}
        </style>
    </head>
    <body>
        <div class="report">
            <header>
                <div class="report-brand-row">
                    {report_brand_logo}
                    <div>
                        <p class="report-brand-kicker">{escape(APP_BRAND_KICKER)}</p>
                        <div class="report-brand-name">{escape(APP_BRAND_NAME)}</div>
                        <p class="report-brand-tagline">{escape(APP_BRAND_TAGLINE)}</p>
                        <a class="report-doi-pill" href="{APP_BRAND_DOI_URL}" target="_blank"><span>DOI</span>{APP_BRAND_DOI}</a>
                    </div>
                </div>
                <div class="report-title-block">
                    <h1>{escape(report_copy['title'])}</h1>
                    <p>{escape(report_copy['purpose'])}</p>
                    <p class="muted" style="color:#F3F4F6; margin-top:8px;">Generated at {escape(generated_at)}</p>
                </div>
                {metrics_html}
            </header>
            <section>
                <h2>Query selection</h2>
                <div class="selected">
                    {_selected_html('Selected compounds', selected_compounds)}
                    {_selected_html('Selected CYP450 targets / proteins', selected_targets)}
                </div>
            </section>
            <section>
                <h2>Plain-language interpretation</h2>
                {plain_summary_html}
                <div class="insight-grid reading-grid">
                    <div class="callout"><h3>How to read this graph</h3>{reading_guide_html}</div>
                    <div class="callout"><h3>Evidence and caution notes</h3>{evidence_quality_html}</div>
                </div>
            </section>
            <section>
                <h2>Evidence path map</h2>
                {evidence_path_html}
            </section>
            <section>
                <h2>Node roles and color legend</h2>
                <p class="muted">The colors below use the same fixed palette as the graph visualization.</p>
                <ul class="legend">{node_color_legend}</ul>
                <div style="margin-top:14px;">{node_role_html}</div>
            </section>
            <section>
                <h2>Relationship meanings</h2>
                {relationship_meaning_html}
            </section>
            <section>
                <h2>Representative entities</h2>
                <p class="muted">Examples below help non-technical users understand what entities are present before opening the full tables.</p>
                {entity_examples_html}
            </section>
            <section>
                <h2>High-signal relationship values</h2>
                {ranked_relationships_html}
            </section>
            <section>
                <h2>Complete relationship details</h2>
                <p class="muted">All returned relationships are shown below with full source and target labels. Long values wrap instead of being cropped.</p>
                {_relationship_preview_html(nodes, edges)}
            </section>
            <section>
                <h2>Complete node and relationship tables</h2>
                <p class="muted">These are the same full tables available from the tabular export. They are included here so the HTML report can be interpreted without opening separate CSV files.</p>
                {full_tables_html}
            </section>
            <section>
                <h2>Recommended next checks</h2>
                {next_steps_html}
            </section>
            <section class="license-notice">
                <h2>License disclaimer</h2>
                <p><strong>{escape(APP_LICENSE_NAME)}</strong></p>
                <p>{escape(APP_LICENSE_DISCLAIMER)}</p>
                <p><a href="{APP_LICENSE_URL}" target="_blank">Read the full GPL-3.0 license text</a></p>
            </section>
        </div>
    </body>
    </html>
    """
    return report_html


def display_report(
    action: str,
    graph,
    selected_compounds: list[str] | None = None,
    selected_targets: list[str] | None = None,
) -> None:
    """Display an action-aware report and provide report/table ZIP downloads."""
    if not graph:
        st.warning("No report can be generated because the query returned no graph data.")
        return

    report_html = generate_report_html(action, graph, selected_compounds, selected_targets)
    tables = graph_to_dataframes(graph)
    report_name = _safe_filename(action, "_report.html")
    package_data = _dataframes_zip_bytes(
        tables,
        extra_files={report_name: report_html},
        package_title=f"{action} report and tabular data package",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download HTML report",
            data=report_html.encode("utf-8"),
            file_name=report_name,
            mime="text/html",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download report + tables ZIP",
            data=package_data,
            file_name=_safe_filename(action, "_report_package.zip"),
            mime="application/zip",
            use_container_width=True,
        )

    components.html(report_html, height=800, scrolling=True)


def display_neo4j_statistics(neo4j_conn) -> None:
    """Display compact Neo4j statistics without fixed-position page overlays."""
    statistics = get_neo4j_statistics(neo4j_conn.driver)
    statistics_html = f"""
    <div style="
        margin: 18px 0 8px;
        background-color: white;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 12px 16px;
        box-shadow: 0 6px 16px rgba(15,23,42,0.06);
    ">
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; font-family: Arial, sans-serif; text-align:center;">
            <div><h5 style="margin:0;color:#334155;">Data statistics</h5><p style="margin:4px 0 0;color:#64748B;">Current Neo4j graph</p></div>
            <div><h5 style="margin:0;color:#334155;">Compounds</h5><p style="margin:4px 0 0;font-weight:700;">{statistics['compounds_count']}</p></div>
            <div><h5 style="margin:0;color:#334155;">Proteins</h5><p style="margin:4px 0 0;font-weight:700;">{statistics['proteins_count']}</p></div>
            <div><h5 style="margin:0;color:#334155;">BioAssays</h5><p style="margin:4px 0 0;font-weight:700;">{statistics['bioassays_count']}</p></div>
            <div><h5 style="margin:0;color:#334155;">Relationships</h5><p style="margin:4px 0 0;font-weight:700;">{statistics['relationships_count']}</p></div>
        </div>
    </div>
    """
    st.markdown(statistics_html, unsafe_allow_html=True)
