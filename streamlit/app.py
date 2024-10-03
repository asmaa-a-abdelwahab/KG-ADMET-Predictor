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
    show_cooccurrence_cpd_cpd,
    show_cooccurrence_cpd_gene,
    show_pubchem_interactions,
    show_external_interactions,
)
from utils.visualization_utils import (
    display_graph,
    display_table,
    display_neo4j_statistics,
    display_report,
)
from utils.ui_utils import display_sidebar, apply_custom_styles


def main() -> None:
    """
    Main entry point of the Streamlit app.
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
            "Co-Occurrence in Literature (Compound-Compound)",
            "Co-Occurrence in Literature (Compound-Gene)",
            "Compound-Gene Interactions (PubChem)",
            "Compound-Gene Interactions (External Sources)",
        ],
    )

    # Tabs for visualizing data
    tab1, tab2, tab3 = st.tabs(["Knowledge Graph", "Tabular Data", "Summary Report"])

    # Submit button to trigger the action
    if st.sidebar.button("Submit", key="submit"):
        # Execute based on the selected action
        if action == "Show Similar Compounds":
            result = get_similar_compounds(neo4j_conn.driver, selected_compounds)
        elif action == "Show Related BioAssays":
            result = show_bioassays(neo4j_conn.driver, selected_compounds)
        elif action == "Co-Occurrence in Literature (Compound-Compound)":
            result = show_cooccurrence_cpd_cpd(neo4j_conn.driver, selected_compounds)
        elif action == "Co-Occurrence in Literature (Compound-Gene)":
            result = show_cooccurrence_cpd_gene(
                neo4j_conn.driver, selected_compounds, selected_genes
            )
        elif action == "Compound-Gene Interactions (PubChem)":
            result = show_pubchem_interactions(
                neo4j_conn.driver, selected_compounds, selected_genes
            )
        elif action == "Compound-Gene Interactions (External Sources)":
            result = show_external_interactions(
                neo4j_conn.driver, selected_compounds, selected_genes
            )

        # Display the results in the appropriate tabs
        with tab1:

            @st.fragment
            def fragment_function():
                # Display the download button
                display_graph(result)

            fragment_function()
        with tab2:
            display_table(result)
        with tab3:
            display_report(action, result)
    else:
        with tab1:
            st.markdown(
                """
                <div style="text-align: center; margin-top: 10px;">
                    <h2>Welcome to the CYP450 Knowledge Graph Application!</h2>
                </div>
            """,
                unsafe_allow_html=True,
            )

            st.video("images/SampleVideo_1280x720_1mb.mp4", format="video/mp4")

    with tab1:
        display_neo4j_statistics(neo4j_conn)

    # GitHub Section with Logo and Hyperlink
    github_html = """
    <div style="display: flex; align-items: center; justify-content: center; margin-top: 10px;">
        <a href="https://github.com/asmaa-a-abdelwahab" target="_blank" style="text-decoration: none;">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub Logo" style="width:40px; height:40px; margin-right: 10px;">
        </a>
        <a href="https://github.com/asmaa-a-abdelwahab" target="_blank" style="text-decoration: none;">
            <p style="font-size: 16px; font-weight: bold; color: black; margin: 0;">@asmaa-a-abdelwahab</p>
        </a>
    </div>
    """
    # <div style="text-align: center; margin-top: 20px;">
    #     <p style="font-size: 14px; color: gray;">Check out my GitHub for more projects!</p>
    # </div>
    st.sidebar.markdown(github_html, unsafe_allow_html=True)

    # Sidebar content
    # st.sidebar.markdown("### Interested in Building the Neo4j Database?")

    # Attractive note with a link to your package
    st.sidebar.markdown(
        """
        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; text-align: center; margin-top: 30px;">
            <p style="font-size: 14px; color: #333;">
                Learn how to build a comprehensive Neo4j database from scratch 
                using automated interaction data retrieval from PubChem with 
                my Python package!
            </p>
            <p style="font-size: 14px; color: #333;">
                Check out my package 
                <a href="https://github.com/asmaa-a-abdelwahab/ChemGraphBuilder" 
                style="color: #ff4b4b; text-decoration: none; font-weight: bold;">
                ChemGraphBuilder
                </a> to get started!
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Close Neo4j connection at the end
    neo4j_conn.close()


if __name__ == "__main__":
    main()
