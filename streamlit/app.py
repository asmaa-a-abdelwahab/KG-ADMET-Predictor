# main.py

"""
Streamlit app for the CYP450-KG project.

This app serves as a frontend for the CYP450-KG knowledge graph and provides
various visualizations and search functionalities.

"""

import streamlit as st
from utils.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    logger,
)
from utils.neo4j_utils import (
    Neo4jBase,
    get_compound_names,
    get_gene_symbols,
    get_similar_compounds,
    show_bioassays,
    show_cooccurrence,
    show_pubchem_interactions,
    show_external_interactions,
)
from utils.visualization_utils import display_graph, display_table
from utils.ui_utils import display_sidebar, apply_custom_styles


def main() -> None:
    """
    Main entry point of the Streamlit app.

    This function sets up the Streamlit page config, initializes the Neo4j
    connection, fetches compound names and gene symbols, renders the sidebar and
    gets user selections, executes the selected action, and closes the Neo4j
    connection at the end.
    """

    # Set up Streamlit page config
    st.set_page_config(layout="wide")

    # Apply custom styles
    apply_custom_styles()

    # Initialize Neo4j connection
    neo4j_conn = Neo4jBase(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    neo4j_conn.connect_to_neo4j()

    # Fetch compound names and gene symbols
    compound_list = get_compound_names(neo4j_conn.driver)
    gene_list = get_gene_symbols(neo4j_conn.driver)

    # Render the sidebar and get user selections
    selected_compounds, selected_genes = display_sidebar(compound_list, gene_list)

    # Dropdown for user action selection
    action = st.sidebar.selectbox(
        "Select Action",
        [
            "Show Similar Compounds",
            "Show Related BioAssays",
            "Co-Occurrence in Literature",
            "Compound-Gene Interactions (PubChem)",
            "Compound-Gene Interactions (External Sources)",
        ],
    )

    tab1, tab2, tab3 = st.tabs(["Knowledge Graph", "Tabular Data", "Prediction Report"])

    # Add a single submit button
    if st.sidebar.button("Submit", key="submit"):
        # Execute based on the selected action
        if action == "Show Similar Compounds":
            result = get_similar_compounds(neo4j_conn.driver, selected_compounds)
            with tab1:
                display_graph(result)
            with tab2:
                display_table(result)

        elif action == "Show Related BioAssays":
            result = show_bioassays(neo4j_conn.driver, selected_compounds)
            with tab1:
                display_graph(result)
            with tab2:
                display_table(result)

        elif action == "Co-Occurrence in Literature":
            result = show_cooccurrence(neo4j_conn.driver, selected_compounds)
            with tab1:
                display_graph(result)
            with tab2:
                display_table(result)

        elif action == "Compound-Gene Interactions (PubChem)":
            result = show_pubchem_interactions(
                neo4j_conn.driver, selected_compounds, selected_genes
            )
            with tab1:
                display_graph(result)
            with tab2:
                display_table(result)

        elif action == "Compound-Gene Interactions (External Sources)":
            result = show_external_interactions(
                neo4j_conn.driver, selected_compounds, selected_genes
            )
            with tab1:
                display_graph(result)
            with tab2:
                display_table(result)

    # Close Neo4j connection at the end
    neo4j_conn.close()


if __name__ == "__main__":
    main()
