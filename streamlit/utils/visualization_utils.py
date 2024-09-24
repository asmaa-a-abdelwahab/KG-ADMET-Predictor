# visualization.py

import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Any
import pandas as pd
import streamlit as st


def display_graph(results: List[dict]) -> None:
    """
    Displays a NetworkX graph based on the Neo4j query results.
    :param results: A list of dictionaries representing nodes and relationships.
    """
    # Initialize a NetworkX graph
    G = nx.DiGraph()
    node_data = {}

    # Process the results
    for record in results:
        # If it's a node (has 'labels' and 'properties')
        if "labels" in record and "properties" in record:
            node_id = record["identity"]
            compound_name = record["properties"].get("CompoundName", "Unknown")
            G.add_node(node_id, label=compound_name)
            node_data[node_id] = compound_name

        # If it's a relationship (has 'type' and 'start', 'end')
        elif "type" in record and "start" in record and "end" in record:
            start_node = record["start"]
            end_node = record["end"]
            G.add_edge(start_node, end_node, label=record["type"])

    # Draw the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    labels = {node: data for node, data in node_data.items()}

    # Draw nodes with labels
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=labels,
        node_color="skyblue",
        node_size=2000,
        font_size=12,
        font_weight="bold",
        arrows=True,
    )

    plt.title("Compound Similarity Graph")
    st.pyplot(plt)


def display_table(result: List[dict]) -> None:
    """
    Display the query result as a table.
    This function dynamically determines the structure of the result and adapts the table accordingly.

    :param result: List of dictionaries where each dictionary represents a row from the query result.
    """
    if not result:
        st.warning("No data to display.")
        return

    # Process the result to convert it to a flat structure if needed
    rows = []
    for record in result:
        # Flatten the nested structure
        row = {}
        for key, value in record.items():
            if isinstance(value, dict):  # If the value is a node or relationship
                for sub_key, sub_value in value.items():
                    row[f"{key}_{sub_key}"] = sub_value
            else:
                row[key] = value
        rows.append(row)

    # Create a DataFrame from the processed rows
    df = pd.DataFrame(rows)

    # Display the DataFrame in Streamlit
    st.dataframe(df)
