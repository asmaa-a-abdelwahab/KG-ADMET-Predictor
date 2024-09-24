import streamlit as st
from neo4j_utils import get_compound_names, get_gene_symbols


def setup_custom_css():
    st.markdown(
        """
        <style>
        div[role="tablist"] > button { width: 33.33% !important; font-size: 20px !important; }
        div[role="tablist"] > button:hover { background-color: #696969 !important; }
        div[role="tablist"] > button[aria-selected="true"] { background-color: #505050 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(neo4j_conn):
    st.markdown('<p class="sidebar-title">CYP450-KG</p>', unsafe_allow_html=True)
    st.image("images/kg_icon.webp", width=120)

    compound_list = get_compound_names(neo4j_conn.driver)
    gene_list = get_gene_symbols(neo4j_conn.driver)

    selected_compounds = st.multiselect("Select Compound/s", compound_list)
    selected_genes = st.multiselect("Select Gene/s", gene_list)

    if st.button("Show Similar Compounds"):
        return selected_compounds
    return None
