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
import random


def extract_graph_data(graph):
    """
    Extracts nodes and relationships from the Neo4j Graph object.

    Parameters:
        graph (Graph): Neo4j Graph object containing nodes and relationships.

    Returns:
        A tuple (nodes, edges) representing the graph.
            - nodes (dict): A dictionary of node IDs to dictionaries of node properties, including labels.
            - edges (list): A list of tuples containing the start node ID, end node ID, and relationship type.
    """
    nodes = {}
    edges = []

    # Extract nodes
    for node in graph.nodes:
        node_id = node.element_id
        props = dict(node)  # Get node properties as a dictionary

        # Extract node labels (assuming nodes can have multiple labels)
        labels = list(node.labels)  # Get labels as a list
        props["labels"] = labels  # Add labels to the properties dictionary

        nodes[node_id] = props

    # Extract relationships
    for rel in graph.relationships:
        start_node = rel.start_node.element_id
        end_node = rel.end_node.element_id
        relationship_type = rel.type
        edges.append((start_node, end_node, relationship_type))

    return nodes, edges


# Define a color map for node labels
def get_label_color(label, label_colors):
    if label not in label_colors:
        # Generate a random color for a new label
        label_colors[label] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
    return label_colors[label]


def display_graph(graph):
    """
    Displays an interactive PyVis graph based on the Neo4j query results, with nodes
    colored based on their labels.

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
    )

    net.show_buttons(filter_=["nodes", "edges", "physics"])

    # Define a dictionary to store colors for each label
    label_colors = {}

    # Add nodes to the PyVis network
    for node_id, props in nodes.items():
        label = props.get("labels", ["Unknown"])[0]  # Get the first label
        color = get_label_color(
            label, label_colors
        )  # Get or assign a color for this label

        if label == "Compound":
            node_label = props.get("CompoundName")  # Set the display name
        elif label == "Gene":
            node_label = props.get("GeneSymbol")
        elif label == "BioAssay":
            node_label = props.get("AssayName")
        elif label == "Protein":
            node_label = props.get("ProteinRefSeqAccession")
        else:
            node_label = "Unknown"

        net.add_node(
            node_id.split(":")[-1],
            title=node_label,
            # label=node_label,
            color=color,  # Use the color based on the label
        )

    # Add edges to the PyVis network with relationship types as edge titles
    for edge in edges:
        start_node = edge[0].split(":")[-1]
        end_node = edge[1].split(":")[-1]
        relationship_type = edge[2]
        net.add_edge(start_node, end_node, title=relationship_type)

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
    Displays the Neo4j Graph result as separate tables for each node label
    and relationships. This function dynamically adapts to the structure
    of both the nodes' and relationships' properties.

    Parameters:
        graph (Graph): Neo4j Graph object containing nodes and relationships.
    """
    if not graph:
        st.warning("No data to display.")
        return

    # Extract nodes and relationships data
    nodes, edges = extract_graph_data(graph)

    # Group nodes by label
    label_groups = {}
    for node_id, props in nodes.items():
        label = props.get("labels", "Unknown")[
            0
        ]  # Assumes 'label' key holds the node label
        if label not in label_groups:
            label_groups[label] = []
        row = {
            "Node ID": node_id.split(":")[-1]
        }  # Get just the ID part after the colon
        row.update(props)  # Add all node properties (e.g., CompoundName, etc.)
        label_groups[label].append(row)

    # Prepare relationship data for the table
    relationship_rows = []
    for start_node, end_node, rel_type in edges:
        rel_row = {
            "Start Node": start_node.split(":")[-1],
            "End Node": end_node.split(":")[-1],
            "Relationship Type": rel_type,
        }
        relationship_rows.append(rel_row)

    # Display nodes in separate tables for each label
    for label, nodes in label_groups.items():
        st.subheader(f"{label} Nodes")
        node_df = pd.DataFrame(nodes)
        st.dataframe(node_df)

    # Convert relationship rows into DataFrame and display it
    if relationship_rows:
        relationship_df = pd.DataFrame(relationship_rows)
        st.subheader("Relationships Information")
        st.dataframe(relationship_df)
    else:
        st.info("No relationships to display.")
