"""
Functions for visualizing Neo4j query results in Streamlit.

Functions:
    - extract_graph_data: Extracts nodes and relationships from a Neo4j Graph object.
    - display_graph: Displays an interactive PyVis graph based on the Neo4j query results.
    - display_table: Displays the Neo4j Graph result as a table, including nodes and relationships.
"""

import pandas as pd
from pyvis.network import Network

import streamlit as st
import streamlit.components.v1 as components


def extract_graph_data(graph):
    """
    Extracts nodes and relationships from the Neo4j Graph object.

    Parameters:
        graph (Graph): Neo4j Graph object containing nodes and relationships.

    Returns:
        A tuple (nodes, edges) representing the graph.
            - nodes (dict): A dictionary of node IDs to dictionaries of node properties.
            - edges (list): A list of tuples containing the start node ID, end node ID, and relationship type.
    """
    nodes = {}
    edges = []

    # Extract nodes
    for node in graph.nodes:
        node_id = node.element_id
        props = dict(node)  # Get node properties as a dictionary
        nodes[node_id] = props

    # Extract relationships
    for rel in graph.relationships:
        start_node = rel.start_node.element_id
        end_node = rel.end_node.element_id
        relationship_type = rel.type
        edges.append((start_node, end_node, relationship_type))

    return nodes, edges


def display_graph(graph):
    """
    Displays an interactive PyVis graph based on the Neo4j query results.

    Parameters:
        graph (Graph): Neo4j Graph object containing nodes and relationships.
    """
    # Extract nodes and edges from the graph
    nodes, edges = extract_graph_data(graph)

    # Create a PyVis network with a white background and black font color
    net = Network(
        height="100vh",
        width="100vw",
        bgcolor="#f9f9f9",
        font_color="black",
        #     filter_menu=True,
        #     select_menu=True,
    )

    net.show_buttons(filter_=["nodes", "edges", "physics"])

    # Add nodes to the PyVis network
    for node_id, props in nodes.items():
        label = props.get("CompoundName", "Unknown")
        net.add_node(
            node_id.split(":")[-1], title=label
        )  # I can set node color here: color="#FF0000"

    # Add edges to the PyVis network with relationship types as edge titles
    for edge in edges:
        start_node = edge[0].split(":")[-1]
        end_node = edge[1].split(":")[-1]
        relationship_type = edge[2]
        net.add_edge(
            start_node, end_node, title=relationship_type
        )  # I can set edge color here: color="#FF0000"

    # Customize the network layout
    net.repulsion(
        node_distance=420,
        central_gravity=0.33,
        spring_length=110,
        spring_strength=0.10,
        damping=0.95,
    )

    net.set_edge_smooth("dynamic")

    # Save the graph as an HTML file
    path = "/tmp"  # or use a relative path for local development
    net.save_graph(f"{path}/pyvis_graph.html")

    # Load and display the HTML file in Streamlit
    HtmlFile = open(f"{path}/pyvis_graph.html", "r", encoding="utf-8")
    components.html(HtmlFile.read(), width=1172, height=850)


def display_table(graph):
    """
    Displays the Neo4j Graph result as a table, including nodes and relationships.
    This function dynamically adapts to the structure of both the nodes' and relationships' properties.

    Parameters:
        graph (Graph): Neo4j Graph object containing nodes and relationships.
    """
    if not graph:
        st.warning("No data to display.")
        return

    # Extract nodes and relationships data
    nodes, edges = extract_graph_data(graph)

    # Prepare node data for the table
    node_rows = []
    for node_id, props in nodes.items():
        row = {"Node ID": node_id.split(":")[-1]}
        row.update(props)  # Add all node properties (e.g., CompoundName, etc.)
        node_rows.append(row)

    # Prepare relationship data for the table
    relationship_rows = []
    for start_node, end_node, rel_type in edges:
        rel_row = {
            "Start Node": start_node.split(":")[-1],
            "End Node": end_node.split(":")[-1],
            "Relationship Type": rel_type,
        }
        relationship_rows.append(rel_row)

    # Convert node and relationship rows into DataFrames
    node_df = pd.DataFrame(node_rows)
    relationship_df = pd.DataFrame(relationship_rows)

    # Display the node and relationship DataFrames as tables in Streamlit
    st.subheader("Node Information")
    st.dataframe(node_df)

    st.subheader("Relationship Information")
    st.dataframe(relationship_df)
