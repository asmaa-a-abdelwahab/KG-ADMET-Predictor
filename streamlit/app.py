# main.py

import streamlit as st
from utils.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from utils.neo4j_utils import (
    Neo4jBase,
    get_compound_names,
    get_gene_symbols,
    get_similar_compounds,
    show_bioassays,
)
from utils.visualization_utils import display_graph, display_table
from utils.ui_utils import display_sidebar, apply_custom_styles


def main():
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

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Knowledge Graph", "Tabular Data", "Prediction Report"])

    if st.sidebar.button("Show Similar Compounds"):
        result = get_similar_compounds(neo4j_conn.driver, selected_compounds)
        with tab1:
            # st.write(result)
            display_graph(result)
        with tab2:
            display_table(result)

    if st.sidebar.button("Show BioAssays"):
        result = show_bioassays(neo4j_conn.driver, selected_compounds)
        with tab1:
            # st.write(result)
            display_graph(result)
        with tab2:
            display_table(result)

    # Add content for other tabs as necessary

    # Close Neo4j connection at the end
    neo4j_conn.close()


if __name__ == "__main__":
    main()
