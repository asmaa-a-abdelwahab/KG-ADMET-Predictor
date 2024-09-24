"""
Main application for the CYP450-KG Streamlit app.

This application visualizes the knowledge graph, provides a tabular data view,
and predicts compound-gene interactions using a deep learning model.

"""

import logging
from typing import List, Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import ServiceUnavailable


class Neo4jConnectionError(Exception):
    """
    Custom exception for Neo4j connection errors.
    """

    pass


class Neo4jBase:
    """
    Base class to manage connections with the Neo4j database.
    """

    def __init__(
        self,
        logger: logging.Logger = None,
        uri: str = "bolt://0.0.0.0:7687",
        user: str = "neo4j",
        password: str = "cyp450kg",
    ) -> None:
        """
        Initialize the Neo4jBase class.

        :param logger: Logging object to use for logging.
        :param uri: URI of the Neo4j database.
        :param user: Username to use for connecting to the Neo4j database.
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

        # Set up logging configuration
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logger or logging.getLogger(__name__)

    def connect_to_neo4j(self) -> None:
        """
        Establish a connection to the Neo4j database using provided URI and username.
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            self.logger.info("Successfully connected to the Neo4j database.")
        except Exception as e:
            self.logger.error("Failed to connect to the Neo4j database: %s", e)
            raise Neo4jConnectionError(
                "Failed to connect to the Neo4j database."
            ) from e

    def close(self) -> None:
        """
        Close the connection to the Neo4j database.
        """
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed successfully.")


def execute_query(driver: Driver, query: str) -> List[List[Any]]:
    """
    Execute a query and fetch results.

    :param driver: Neo4j driver object.
    :param query: Cypher query to execute.
    :return: List of records returned by the query.
    """
    try:
        with driver.session() as session:
            result = session.run(query)
            records = [record for record in result]
            if records:
                return [record.values() for record in records]
            else:
                return []
    except ServiceUnavailable as e:
        st.error(f"Error executing query: {e}")
        return []


def get_compound_names(driver: Driver) -> List[str]:
    """
    Fetch compound names from Neo4j.

    :param driver: Neo4j driver object.
    :return: List of compound names.
    """
    query = "MATCH (c:Compound) RETURN c.CompoundName AS name"
    compounds = execute_query(driver, query)
    return [record[0] for record in compounds if record]  # Safeguard for empty records


def get_gene_symbols(driver: Driver) -> List[str]:
    """
    Fetch gene symbols from Neo4j.

    :param driver: Neo4j driver object.
    :return: List of gene symbols.
    """
    query = "MATCH (g:Gene) RETURN g.GeneSymbol AS symbol"
    genes = execute_query(driver, query)
    return [record[0] for record in genes if record]  # Safeguard for empty records


def main() -> None:
    """
    Main entry point of the application.
    """
    # Set Streamlit to use the wide layout
    st.set_page_config(layout="wide")

    # Custom CSS for controlling each tab's width and text style
    st.markdown(
        """
        <style>
        /* Styling all tabs to be 33.33% wide */
        div[role="tablist"] > button {
            width: 33.33% !important;  /* Set the width of each tab to 33.33% */
            font-size: 20px !important;  /* Change the font size */
            font-family: 'Arial', sans-serif !important;  /* Change the font family */
            color: #ffffff !important;  /* Change the text color */
            background-color: #808080 !important;  /* Light gray background color */
            padding: 10px !important;  /* Add padding for more space */
            border-radius: 5px !important;  /* Rounded corners for the tabs */
            border: none !important;  /* Remove border */
            text-align: center !important;  /* Center-align text */
        }

        /* Change the hover effect */
        div[role="tablist"] > button:hover {
            background-color: #696969 !important;  /* Darker gray on hover */
            color: #ffffff !important;  /* Text color on hover */
        }

        /* Change the appearance of the active tab */
        div[role="tablist"] > button[aria-selected="true"] {
            background-color: #505050 !important;  /* Even darker gray for active tab */
            font-weight: bold !important;  /* Bold font for active tab */
            color: #ffffff !important;  /* Text color for active tab */
        }

        /* Styling for the sidebar buttons */
        .sidebar-buttons button {
            margin-bottom: 10px !important;
            width: 100% !important;
            background-color: #ffffff !important;
            border: 2px solid #808080 !important;
            border-radius: 5px !important;
            color: #000000 !important;
        }

        .sidebar-buttons button:hover {
            background-color: #d3d3d3 !important;  /* Change the hover effect */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize Neo4j connection
    neo4j_conn = Neo4jBase(uri="bolt://neo4j:7687", user="neo4j", password="cyp450kg")

    neo4j_conn.connect_to_neo4j()  # Connect to Neo4j

    # Streamlit app layout
    tab1, tab2, tab3 = st.tabs(["Knowledge Graph", "Tabular Data", "Prediction Report"])

    # Tab 1: Knowledge Graph Visualization and Interactions
    with tab1:
        with st.sidebar:
            st.sidebar.markdown(
                """
                <style>
                .sidebar-title {
                    font-size: 40px;  /* Increase font size */
                    font-weight: bold;  /* Optional: Make the font bold */
                    color: black;  /* Optional: Change the color */
                    text-align: center; /* Center the title */
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<p class="sidebar-title">CYP450-KG</p>', unsafe_allow_html=True
            )

            # Displaying the logo image
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                st.image("images/kg_icon.webp", width=120, use_column_width=False)

            # Fetch compound names from Neo4j
            compound_list = get_compound_names(neo4j_conn.driver)

            # Fetch gene symbols from Neo4j
            gene_list = get_gene_symbols(neo4j_conn.driver)

            # Multi-select dropdown for compounds
            if compound_list:
                selected_compounds = st.multiselect("Select Compound/s", compound_list)
            else:
                st.write("No compounds available")

            # Multi-select dropdown for genes
            if gene_list:
                selected_genes = st.multiselect("Select Gene/s", gene_list)
            else:
                st.write("No genes available")

            # Group buttons inside a div with custom class
            st.markdown('<div class="sidebar-buttons">', unsafe_allow_html=True)
            st.button("Show Similar Compounds", key="similar_compounds")
            st.button("Show BioAssays", key="show_bioassays")
            st.button(
                "Compound-Gene Interactions (Other Sources)",
                key="other_sources_interactions",
            )
            st.markdown("</div>", unsafe_allow_html=True)

            # Placeholder for showing interactions
            if st.sidebar.button(
                "Compound-Gene Interactions (PubChem)",
                key="main_pubchem_interactions",
            ):
                st.write(
                    f"Showing interactions for {selected_compounds} and {selected_genes}"
                )
                graph_data = pd.DataFrame(
                    {
                        "compound": [", ".join(selected_compounds)],
                        "gene": [", ".join(selected_genes)],
                        "interaction_type": ["Sample Interaction"],
                    }
                )
                st.write(graph_data)

                # Placeholder graph
                fig, ax = plt.subplots()
                ax.plot([1, 2, 3], [1, 2, 3])  # Example graph
                st.pyplot(fig)

            # Button for predicting compound-gene interactions using the deep learning model
            if st.sidebar.button(
                "Predict Compound-Gene Interaction", key="main_predict_interaction"
            ):
                st.write(
                    f"Predicted interaction for {selected_compounds} and {selected_genes}"
                )
                st.write("Confidence Score: 0.85")
                prediction_report = {
                    "compound": selected_compounds,
                    "gene": selected_genes,
                    "prediction": "Sample Prediction",
                    "confidence_score": 0.85,
                }

    # Tab 2: Tabular Data View
    with tab2:
        st.header("Tabular Data")
        if "graph_data" in locals():
            st.dataframe(graph_data)
        else:
            st.write("No data to display yet.")

    # Tab 3: Report
    with tab3:
        st.header("Prediction Report")
        if "prediction_report" in locals():
            st.write(f"Compound: {', '.join(prediction_report['compound'])}")
            st.write(f"Gene: {', '.join(prediction_report['gene'])}")
            st.write(f"Prediction: {prediction_report['prediction']}")
            st.write(f"Confidence Score: {prediction_report['confidence_score']}")
            st.write("Download the report: [Download PDF](#)")
        else:
            st.write("No report available yet.")


if __name__ == "__main__":
    main()
