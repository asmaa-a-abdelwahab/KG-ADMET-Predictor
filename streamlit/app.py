from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
from utils.config import (
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    PAGE_ICON,
    PAGE_TITLE,
    PRING_REPO_URL,
    logger,
)
from utils.neo4j_utils import (
    Neo4jBase,
    get_compound_names,
    get_gene_symbols,
    get_neo4j_statistics,
    get_similar_compounds,
    show_bioassays,
    show_cooccurrence_cpd_cpd,
    show_cooccurrence_cpd_gene,
    show_pubchem_interactions,
    show_external_interactions,
)
from utils.visualization_utils import (
    APP_LICENSE_DISCLAIMER,
    APP_LICENSE_NAME,
    APP_LICENSE_URL,
    display_graph,
    display_table,
    display_neo4j_statistics,
    display_report,
)
from utils.ui_utils import display_sidebar, apply_custom_styles


VIEW_TABS = ["Knowledge Graph", "Tabular Data", "Summary Report"]


def _all_statistics_zero(statistics: dict[str, Any]) -> bool:
    """Return True when the connected Neo4j database appears empty."""
    keys = ["compounds_count", "proteins_count", "bioassays_count", "relationships_count"]
    return all(int(statistics.get(key, 0) or 0) == 0 for key in keys)


def _format_count(value: Any) -> str:
    """Format a graph count for the landing page."""
    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


def _show_welcome(statistics: dict[str, Any]) -> None:
    """Render the landing page shown before the user submits an analysis."""
    empty_graph = _all_statistics_zero(statistics)
    setup_callout = ""
    if empty_graph:
        setup_callout = """
        <div class="kg-setup-callout">
            <strong>No graph data was detected yet.</strong>
            Load a PRING run into Neo4j first, then return to this app to explore compounds,
            CYP450 targets, assay evidence, and interaction context.
        </div>
        """

    st.markdown(
        f"""
        <section class="kg-landing-hero kg-landing-hero-compact">
            <div class="kg-hero-eyebrow">PRING knowledge graph</div>
            <h1>CYP450-KG Explorer</h1>
            <p>
                A guided evidence browser for compound similarity, PubChem BioAssay provenance,
                CYP450 interaction assertions, and enrichment context from PRING-generated Neo4j graphs.
            </p>
            {setup_callout}
        </section>

        <section class="kg-analysis-overview">
            <div class="kg-overview-copy">
                <div class="kg-section-kicker">What can you ask?</div>
                <h2>Choose the analysis that matches your scientific question</h2>
                <p>
                    The app separates hypothesis-generating chemical neighbourhoods from evidence-backed
                    compound-target assertions. Use the graph to inspect topology, tables to audit raw
                    properties, and the report to explain what the returned evidence path means.
                </p>
            </div>
            <div class="kg-analysis-cards">
                <div class="kg-analysis-card kg-analysis-card-muted">
                    <strong>Compound-only analyses</strong>
                    <span>Similarity, BioAssays, and shared evidence between compounds.</span>
                </div>
                <div class="kg-analysis-card kg-analysis-card-red">
                    <strong>Target-aware analyses</strong>
                    <span>Compound-CYP450 co-occurrence, direct interactions, and enrichment context.</span>
                </div>
            </div>
        </section>

        <section class="kg-howto-grid kg-howto-grid-compact">
            <div class="kg-howto-card">
                <div class="kg-howto-title">
                    <span class="kg-step-badge">1</span>
                    <h3>Choose an analysis</h3>
                </div>
                <p>Start from the scientific question, then use the sidebar guide to select the correct query type.</p>
            </div>
            <div class="kg-howto-card">
                <div class="kg-howto-title">
                    <span class="kg-step-badge">2</span>
                    <h3>Select entities</h3>
                </div>
                <p>Select up to five compounds. CYP450 targets appear only for analyses that require them.</p>
            </div>
            <div class="kg-howto-card">
                <div class="kg-howto-title">
                    <span class="kg-step-badge">3</span>
                    <h3>Inspect evidence</h3>
                </div>
                <p>Click nodes and edges for copyable tooltips, external database links, and evidence details.</p>
            </div>
            <div class="kg-howto-card">
                <div class="kg-howto-title">
                    <span class="kg-step-badge">4</span>
                    <h3>Export results</h3>
                </div>
                <p>Download graph JSON/HTML, tables, and the HTML report for documentation or thesis use.</p>
            </div>
        </section>

        <section class="kg-functionality-panel">
            <div>
                <h3>Result views</h3>
                <ul>
                    <li><strong>Knowledge Graph:</strong> interactive topology with draggable nodes, zoom, pan, and pinned tooltips.</li>
                    <li><strong>Tabular Data:</strong> collapsible node and relationship tables for exact property inspection.</li>
                    <li><strong>Summary Report:</strong> the main interpretable output explaining evidence paths in plain language.</li>
                </ul>
            </div>
            <div>
                <h3>Evidence interpretation</h3>
                <ul>
                    <li><strong>Similarity</strong> supports analogue discovery, not direct activity labels.</li>
                    <li><strong>BioAssay/Endpoint paths</strong> support provenance and measurement auditability.</li>
                    <li><strong>Interaction paths</strong> connect selected compounds to explicit CYP450 protein assertions.</li>
                </ul>
            </div>
        </section>

        <section class="kg-stat-panel">
            <div>
                <h3>Current Neo4j graph</h3>
                <p>Live database statistics from the connected PRING knowledge graph.</p>
            </div>
            <div class="kg-stat-grid">
                <div><strong>{_format_count(statistics.get('compounds_count'))}</strong><span>Compounds</span></div>
                <div><strong>{_format_count(statistics.get('proteins_count'))}</strong><span>Proteins</span></div>
                <div><strong>{_format_count(statistics.get('bioassays_count'))}</strong><span>BioAssays</span></div>
                <div><strong>{_format_count(statistics.get('relationships_count'))}</strong><span>Relationships</span></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_result_tabs(
    result: Any,
    result_action: str,
    result_compounds: list[str],
    result_genes: list[str],
    neo4j_conn: Neo4jBase,
) -> None:
    """Render all result views after a submitted query.

    Native Streamlit tabs are used instead of radio-button tabs so changing
    between result views does not submit another query. The graph tab is first,
    therefore it is selected by default immediately after Submit is clicked.
    """
    graph_tab, table_tab, report_tab = st.tabs(VIEW_TABS)

    with graph_tab:
        display_graph(result, result_action)

    with table_tab:
        display_table(result)

    with report_tab:
        display_report(result_action, result, result_compounds, result_genes)


def _run_action(action: str, neo4j_conn: Neo4jBase, selected_compounds: list[str], selected_genes: list[str]):
    """Dispatch the selected sidebar action to the schema-aware query function."""
    if action == "Show Similar Compounds":
        return get_similar_compounds(neo4j_conn.driver, selected_compounds)
    if action == "Show Related BioAssays":
        return show_bioassays(neo4j_conn.driver, selected_compounds)
    if action == "Co-Occurrence in Literature (Compound-Compound)":
        return show_cooccurrence_cpd_cpd(neo4j_conn.driver, selected_compounds)
    if action == "Evidence Co-Occurrence (Compound-Target)":
        return show_cooccurrence_cpd_gene(neo4j_conn.driver, selected_compounds, selected_genes)
    if action == "Compound-Target Interactions (PubChem)":
        return show_pubchem_interactions(neo4j_conn.driver, selected_compounds, selected_genes)
    if action == "Compound-Target Enrichment Context":
        return show_external_interactions(neo4j_conn.driver, selected_compounds, selected_genes)
    return None


def main() -> None:
    """Main entry point for the CYP450-KG Streamlit app."""
    page_icon = PAGE_ICON if Path(PAGE_ICON).exists() else "🧬"
    st.set_page_config(page_title=PAGE_TITLE, page_icon=page_icon, layout="wide")
    st.markdown(
        """
        <style>
        :root {
            --app-top-offset: 0.85rem;
        }

        .main .block-container {
            padding-top: var(--app-top-offset) !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stSidebarContent"] {
            padding-top: var(--app-top-offset) !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
            gap: 0.85rem !important;
        }

        section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div:first-child {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        .sidebar-card:first-child,
        .sidebar-logo-card,
        .brand-card,
        .kg-brand-card {
            margin-top: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    apply_custom_styles()

    neo4j_conn = Neo4jBase(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE,
    )

    try:
        neo4j_conn.connect_to_neo4j()
    except Exception as exc:
        st.error("Could not connect to Neo4j. Make sure the Neo4j container is healthy and the PRING run has been loaded.")
        st.exception(exc)
        return

    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_action", None)
    st.session_state.setdefault("last_selected_compounds", [])
    st.session_state.setdefault("last_selected_genes", [])

    try:
        statistics = get_neo4j_statistics(neo4j_conn.driver)
        compound_list = get_compound_names(neo4j_conn.driver)
        gene_list = get_gene_symbols(neo4j_conn.driver)
    except Exception as exc:
        st.error("Could not read compounds/targets from the PRING Neo4j graph.")
        st.exception(exc)
        neo4j_conn.close()
        return

    submitted, selected_compounds, selected_genes, action = display_sidebar(compound_list, gene_list)

    if submitted:
        try:
            result = _run_action(action, neo4j_conn, selected_compounds, selected_genes)
            st.session_state["last_result"] = result
            st.session_state["last_action"] = action
            st.session_state["last_selected_compounds"] = selected_compounds
            st.session_state["last_selected_genes"] = selected_genes
        except ValueError as exc:
            st.warning(str(exc))
            st.session_state["last_result"] = None
            st.session_state["last_action"] = None
            st.session_state["last_selected_compounds"] = []
            st.session_state["last_selected_genes"] = []
        except Exception as exc:
            st.error("The query failed. Check whether the selected data exists in the PRING Neo4j graph.")
            st.exception(exc)
            logger.exception("Streamlit action failed")
            st.session_state["last_result"] = None
            st.session_state["last_action"] = None
            st.session_state["last_selected_compounds"] = []
            st.session_state["last_selected_genes"] = []

    result = st.session_state.get("last_result")
    result_action = st.session_state.get("last_action")
    result_compounds = st.session_state.get("last_selected_compounds", [])
    result_genes = st.session_state.get("last_selected_genes", [])

    if result is None:
        _show_welcome(statistics)
    else:
        _render_result_tabs(result, result_action, result_compounds, result_genes, neo4j_conn)

    st.markdown(
        f"""
        <div class="kg-license-footer">
            <strong>License disclaimer:</strong>
            {APP_LICENSE_DISCLAIMER}
            <a href="{APP_LICENSE_URL}" target="_blank">{APP_LICENSE_NAME}</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    github_html = f"""
    <div class="kg-sidebar-footer-compact">
        <a class="kg-github-row" href="https://github.com/asmaa-a-abdelwahab" target="_blank">
            <img
                src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
                alt="GitHub Logo"
            >
            <span>@asmaa-a-abdelwahab</span>
        </a>
        <div class="kg-footer-note">
            <span>
                CYP450-KG Explorer queries a Neo4j knowledge graph built from PRING run data.
                Build or update the graph using
                <a href="{PRING_REPO_URL}" target="_blank">PRING</a>.
            </span>
        </div>
    </div>
    """

    st.sidebar.markdown(github_html, unsafe_allow_html=True)

    neo4j_conn.close()


if __name__ == "__main__":
    main()
