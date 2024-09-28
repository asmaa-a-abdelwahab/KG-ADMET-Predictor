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
            - edges (list): A list of tuples containing the start node ID, end node ID, relationship type, and relationship properties.
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

        # Extract relationship properties as a dictionary
        rel_props = dict(rel)  # Get relationship properties

        # Append the relationship details and properties
        edges.append((start_node, end_node, relationship_type, rel_props))

    return nodes, edges


def display_graph(graph):
    """
    Displays an interactive PyVis graph based on the Neo4j query results, with nodes
    colored based on their labels, and ensures the graph is always centered.

    Parameters:
        graph (Graph): Neo4j Graph object containing nodes and relationships.
    """
    # Extract nodes and edges from the graph
    nodes, edges = extract_graph_data(graph)

    # Create a PyVis network with a white background and black font color
    net = Network(
        height="850px",
        width="100%",
        bgcolor="#f9f9f9",
        font_color="black",
    )

    # Enable the 'nodes', 'edges', and 'physics' options in the toolbar
    # net.show_buttons(filter_=["nodes", "edges", "physics"])

    # Add nodes to the PyVis network
    for node_id, props in nodes.items():
        label = props.get("labels", ["Unknown"])[
            0
        ]  # Get the first labelGet or assign a color for this label

        # Set node label based on its type
        if label == "Compound":
            node_label = props.get("CompoundName")
            color = "#FFA500"
        elif label == "Gene":
            node_label = props.get("GeneSymbol")
            color = "#0099FF"
        elif label == "BioAssay":
            node_label = props.get("AssayName")
            color = "#FF3333"
        elif label == "Protein":
            node_label = props.get("ProteinRefSeqAccession")
            color = "#CCCCCC"
        else:
            node_label = "Unknown"

        net.add_node(
            node_id.split(":")[-1],
            title=node_label,
            color=color,  # Use the color based on the label
        )

    # Add edges to the PyVis network with relationship types as edge titles
    for edge in edges:
        start_node = edge[0].split(":")[-1]
        end_node = edge[1].split(":")[-1]
        relationship_type = edge[2]
        net.add_edge(start_node, end_node, title=relationship_type)

    # Customize the network layout
    # net.repulsion()
    net.repulsion(
        node_distance=220,  # Reduce the node distance to better fit the screen
        central_gravity=0.2,  # Increase gravity to make nodes cluster more in the center
        spring_length=100,
        spring_strength=0.05,
        damping=0.8,
    )

    # Save the graph as an HTML file
    path = "/tmp"  # or use a relative path for local development
    net.save_graph(f"{path}/pyvis_graph.html")

    # Load and display the HTML file in Streamlit
    HtmlFile = open(f"{path}/pyvis_graph.html", "r", encoding="utf-8")

    # Use width '100%' to take the full width of the container
    components.html(HtmlFile.read(), height=850)


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
            "Node ID": node_id.split(":")[-1]  # Get just the ID part after the colon
        }
        row.update(props)  # Add all node properties (e.g., CompoundName, etc.)
        label_groups[label].append(row)

    # Prepare relationship data for the table, including properties
    relationship_rows = []
    for start_node, end_node, rel_type, rel_props in edges:
        rel_row = {
            "Start Node": start_node.split(":")[-1],
            "End Node": end_node.split(":")[-1],
            "Relationship Type": rel_type,
        }
        # Include all relationship properties in the table
        rel_row.update(rel_props)  # Add relationship properties dynamically
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
