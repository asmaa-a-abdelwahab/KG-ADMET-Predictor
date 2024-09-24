import streamlit as st
import pandas as pd
import sys
import os

# Add the project root (directory containing main.py) to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from utils.neo4j_utils import (
    Neo4jBase,
    get_compound_names,
    get_gene_symbols,
    get_similar_compounds,
)
from utils.ui_utils import render_sidebar, setup_custom_css
from utils.visualization_utils import visualize_graph, display_table


def main() -> None:
    """
    Main entry point of the application.
    """
    st.set_page_config(layout="wide")
    setup_custom_css()

    # Initialize Neo4j connection
    neo4j_conn = Neo4jBase(uri="bolt://neo4j:7687", user="neo4j", password="cyp450kg")
    neo4j_conn.connect_to_neo4j()  # Connect to Neo4j

    # Streamlit app layout
    tab1, tab2, tab3 = st.tabs(["Knowledge Graph", "Tabular Data", "Prediction Report"])

    with st.sidebar:
        render_sidebar(neo4j_conn)

    with tab1:
        st.write("Knowledge Graph Visualization")

    with tab2:
        st.header("Tabular Data")

    with tab3:
        st.header("Prediction Report")


if __name__ == "__main__":
    main()
