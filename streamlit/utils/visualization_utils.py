import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components
from py2neo import Graph
import pandas as pd


def extract_graph_data(graph):
    """
    Extracts nodes and relationships from the Neo4j Graph object.
    :param graph: Neo4j Graph object containing nodes and relationships.
    :return: A tuple (nodes, edges) representing the graph.
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
    :param graph: Neo4j Graph object.
    """
    # Extract nodes and edges from the graph
    nodes, edges = extract_graph_data(graph)

    # Create a PyVis network
    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")

    # Add nodes and edges to the PyVis network
    for node_id, props in nodes.items():
        label = props.get("CompoundName", "Unknown")
        net.add_node(node_id, label=label, title=label)

    for edge in edges:
        net.add_edge(edge[0], edge[1], title=edge[2])

    # Customize the network layout
    net.repulsion(
        node_distance=420,
        central_gravity=0.33,
        spring_length=110,
        spring_strength=0.10,
        damping=0.95,
    )

    # Save the graph as an HTML file
    path = "/tmp"  # or use a relative path for local development
    net.save_graph(f"{path}/pyvis_graph.html")

    # Load and display the HTML file in Streamlit
    HtmlFile = open(f"{path}/pyvis_graph.html", "r", encoding="utf-8")
    components.html(HtmlFile.read(), height=500)


def display_table(graph):
    """
    Display the Neo4j Graph result as a table.
    This function dynamically adapts to the structure of the nodes' properties.

    :param graph: Neo4j Graph object containing nodes and relationships.
    """
    if not graph:
        st.warning("No data to display.")
        return

    # Extract nodes and relationships data
    nodes, _ = extract_graph_data(graph)

    # Prepare data for the table by extracting node properties
    rows = []
    for node_id, props in nodes.items():
        row = {"Node ID": node_id}  # Include Node ID in the table
        row.update(props)  # Include all node properties (e.g., CompoundName, etc.)
        rows.append(row)

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(rows)

    # Display the DataFrame as a table using Streamlit
    st.dataframe(df)
