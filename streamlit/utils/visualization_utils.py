"""
Functions for visualizing Neo4j query results in Streamlit.

Functions:
    - extract_graph_data: Extracts nodes and relationships from a Neo4j Graph object.
    - display_graph: Displays an interactive PyVis graph based on the Neo4j query results.
    - display_table: Displays the Neo4j Graph result as a table, including nodes and relationships.
"""

import os
from io import BytesIO

import pandas as pd
import pdfkit
from pyvis.network import Network

import streamlit as st
import streamlit.components.v1 as components
from utils.neo4j_utils import get_neo4j_statistics


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


def unescape_quotes(value):
    """
    Replace escaped single and double quotes in a string value.

    :param value: String that might contain escaped quotes
    :return: String with non-escaped quotes, or the original value if not a string.
    """
    if isinstance(value, str):
        return value.replace("\\'", "'").replace('\\"', '"')
    return value  # Return the value as is if it's not a string


def display_graph(graph):
    """
    Displays an interactive PyVis graph based on the Neo4j query results, with nodes
    colored based on their labels, and ensures the graph is always centered.
    Also adds a fixed info panel to show node/edge properties when selected.
    """

    # Extract nodes and edges from the graph
    nodes, edges = extract_graph_data(graph)

    # Create a PyVis network with a white background and black font color
    net = Network(
        height="740px",
        width="100%",
        bgcolor="#f9f9f9",
        font_color="black",
    )

    # Add nodes to the PyVis network with properties displayed on click/hover
    for node_id, props in nodes.items():
        label = props.get("labels")[0]  # Get the first label

        # Set node label and color based on its type
        if label == "Compound":
            node_label = props.get("CompoundName")
            color = "#FFA500"  # Orange
        elif label == "Gene":
            node_label = props.get("GeneSymbol")
            color = "#0099FF"  # Blue
        elif label == "BioAssay":
            node_label = props.get("AssayName")
            color = "#FF3333"  # Red
        elif label == "Protein":
            node_label = props.get("ProteinRefSeqAccession")
            color = "#CCCCCC"  # Gray
        else:
            node_label = "Unknown"
            color = "#CCCCCC"

        # Get the last key-value pair
        last_key, last_value = list(props.items())[-1]

        # Remove the last item
        props.pop(last_key)

        # Create a new dictionary with the renamed last item at the start
        new_dict = {"Label": last_value[0]}

        # Add the rest of the original dictionary items
        new_dict.update(props)
        formatted_properties = "<br>".join(
            [
                f"<b>{key}</b>: {unescape_quotes(value)}"
                for key, value in new_dict.items()
            ]
        )

        # Add node with title showing all properties in a formatted way
        net.add_node(
            node_id.split(":")[-1],
            title=formatted_properties,
            label=node_label,
            color=color,
        )

    # Add edges to the PyVis network with relationship types and properties as edge titles
    for edge in edges:
        start_node = edge[0].split(":")[-1]
        end_node = edge[1].split(":")[-1]
        relationship_type = edge[2]
        edge_properties = edge[3] if len(edge) > 3 else {}

        relationship_info = f"<b>Type:</b> {relationship_type}<br>"
        relationship_info += "<br>".join(
            [
                f"<b>{key}</b>: {unescape_quotes(value)}"
                for key, value in edge_properties.items()
            ]
        )

        # Add edge with title showing relationship properties
        net.add_edge(start_node, end_node, title=relationship_info)

    # Customize the network layout
    net.repulsion(
        node_distance=220,
        central_gravity=0.2,
        spring_length=100,
        spring_strength=0.05,
        damping=0.8,
    )

    # Save the graph as an HTML file
    path = "/tmp"
    net.save_graph(f"{path}/pyvis_graph.html")

    # Load the graph and add a fixed info panel for selected node/edge details
    HtmlFile = open(f"{path}/pyvis_graph.html", "r", encoding="utf-8")
    graph_html = HtmlFile.read()
    HtmlFile.close()

    # JavaScript to capture node or edge click and update info panel, also reset when clicking outside
    custom_js = """
    <script type="text/javascript">
        function updateInfoPanel(content) {
            document.getElementById('info-panel').innerHTML = content;
        }

        // Function to reset the info panel
        function resetInfoPanel() {
            document.getElementById('info-panel').innerHTML = "<h4>Properties</h4><p>Select a node or edge to see details here.</p>";
        }
        
        function removeKeys(obj, keysToRemove) {
            keysToRemove.forEach(key => {
                delete obj[key];
            });
            return obj;
        }

        // Capture node click event
        network.on("selectNode", function(params) {
            let nodeId = params.nodes[0];
            let node = nodes.get(nodeId);  // Get the node properties
            let keysToRemove = ["color", "font", "label", "shape", "labels"];
            let updatedNode = removeKeys(node, keysToRemove);
            let content = "<h4>Node Properties:</h4><br>";
            for (let key in updatedNode) {
                if (key !== "title") {
                    content += "<b>" + key + "</b>: " + updatedNode[key] + "<br>";
                }
                else {
                    content += updatedNode["title"] + "<br>";
                }
            }
            updateInfoPanel(content);  // Update info panel with node properties

        });

        // Capture edge click event, only if no node is selected
        network.on("selectEdge", function(params) {
            if (params.nodes.length === 0) {  // Only handle edge if no node is selected
                let edgeId = params.edges[0];
                let edge = edges.get(edgeId);  // Get the edge properties
                let keysToRemove = ["from", "to", "id"];
                let updatedEdge = removeKeys(edge, keysToRemove);
                let content = "<h4>Edge Properties:</h4><br>";
                for (let key in updatedEdge) {
                    if (key !== "title") {
                        content += "<b>" + key + "</b>: " + updatedEdge[key] + "<br>";
                    }
                    else {
                        content += updatedEdge["title"] + "<br>";
                    }
                }
                updateInfoPanel(content);  // Update info panel with edge properties
            }
        });

        // Add a click event to reset the info panel when clicking outside nodes or edges
        network.on("deselectNode", function() {
            resetInfoPanel();
        });

        network.on("deselectEdge", function() {
            resetInfoPanel();
        });
    </script>
    """

    # Fixed info panel HTML with scrollable content
    info_panel_html = """
    <div id="info-panel" style="position: fixed; top: 20px; right: 10px; width: 300px; height: 720px; padding: 10px; background-color: #f9f9f9; border-radius: 5px; box-shadow: 0px 4px 8px rgba(0,0,0,0.1); z-index: 999; overflow-x: auto; overflow-y: auto;">
        <h4>Properties</h4>
        <p>Select a node or edge to see details here.</p>
    </div>
    """

    # Legend HTML
    legend_html = """
    <div style="position: fixed; left: 10px; top: 20px; background-color: #f9f9f9; padding: 10px; border-radius: 5px; box-shadow: 0px 4px 8px rgba(0,0,0,0.1); z-index: 999;">
        <h4 style="text-align: center; font-family: Arial, sans-serif;">Node Legend</h4>
        <ul style="list-style-type: none; padding: 0; font-family: Arial, sans-serif;">
            <li><span style="background-color: #FFA500; border-radius: 50%; display: inline-block; width: 12px; height: 12px; margin-right: 10px;"></span> Compound</li>
            <li><span style="background-color: #0099FF; border-radius: 50%; display: inline-block; width: 12px; height: 12px; margin-right: 10px;"></span> Gene</li>
            <li><span style="background-color: #FF3333; border-radius: 50%; display: inline-block; width: 12px; height: 12px; margin-right: 10px;"></span> BioAssay</li>
        </ul>
    </div>
    """

    # Combine everything into the final HTML output
    full_html = f"""
    <div style="position: fixed; width: 100%; height: 770px;">
        {legend_html}
        {info_panel_html}
        {graph_html}
        {custom_js}
    </div>
    """

    # Use width '100%' to take the full width of the container
    components.html(full_html, height=760, scrolling=True)


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
        label = props.get("labels")[0]  # Assumes 'label' key holds the node label
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
        node_df.replace(r"\\'", "'", regex=True, inplace=True)
        node_df.replace(r'\\"', '"', regex=True, inplace=True)
        st.dataframe(node_df)

    # Convert relationship rows into DataFrame and display it
    if relationship_rows:
        relationship_df = pd.DataFrame(relationship_rows)
        relationship_df.replace(r"\\'", "'", regex=True, inplace=True)
        relationship_df.replace(r'\\"', '"', regex=True, inplace=True)
        st.subheader("Relationships Information")
        st.dataframe(relationship_df)
    else:
        st.info("No relationships to display.")


def group_by_start_node(list_of_relationships):
    """
    Divides the list of relationships into multiple lists where each sublist
    contains relationships with the same start_node_info (first list element).

    Parameters:
        list_of_relationships: List of lists in the form [start_node_info, end_node_info, relationship_type, rel_props]

    Returns:
        A list of lists, where each inner list contains sublists that share the same start_node_info.
    """
    grouped_relationships = []
    processed_start_nodes = []

    # Iterate through the list of relationships
    for relationship in list_of_relationships:
        start_node_info = relationship[0]

        # Check if this start_node_info has already been processed
        if start_node_info not in processed_start_nodes:
            # Group all relationships with the same start_node_info
            group = [rel for rel in list_of_relationships if rel[0] == start_node_info]
            grouped_relationships.append(group)
            processed_start_nodes.append(start_node_info)

    return grouped_relationships


def download_html_as_pdf(html_content: str, output_file: str):
    """
    Converts HTML content to a PDF using pdfkit.

    :param html_content: HTML content as a string.
    :param output_file: The name of the output PDF file.
    """
    pdfkit.from_string(html_content, output_file)


def display_report(action, graph):
    """
    Generates and displays a summary HTML report from the extracted graph data,
    focusing on relationships and retrieving important node information.

    Parameters:
        graph: The Neo4j graph object with nodes and relationships.
    """
    # Extract nodes and edges from the graph using the provided function
    nodes, edges = extract_graph_data(graph)

    # Helper function to retrieve important node information
    def get_node_info(node_id):
        """
        Retrieves important information about a node based on its ID.

        Parameters:
            node_id (str): The ID of the node to retrieve information for.

        Returns:
            A list containing the node ID, label, name, and properties.
                - node_id (str): The ID of the node.
                - label (str): The label of the node (e.g. Compound, Gene, etc.).
                - name (str): The name of the node (e.g. CompoundName, GeneSymbol, etc.).
                - props (dict): The properties of the node.
        """
        props = nodes.get(node_id, {})
        label = props.get("labels", ["Unknown"])[0]

        # Retrieve the name of the node based on its label
        if label == "Compound":
            name = props.get("CompoundName", "N/A")
        elif label == "Gene":
            name = props.get("GeneSymbol", "N/A")
        elif label == "BioAssay":
            name = props.get("AssayName", "N/A")
        else:
            name = "N/A"  # Default case if label doesn't match

        return [node_id, label, name, props]

    # Prepare content for relationships (edges) and associated node information
    relationships_info = []
    compound_info = {}  # To store compound details for the table, without repetition
    all_props = set()  # To gather all unique property names

    for edge in edges:
        start_node = edge[0]
        end_node = edge[1]
        relationship_type = edge[2]
        rel_props = edge[3]

        # Retrieve important information for the nodes involved
        start_node_info = get_node_info(start_node)
        end_node_info = get_node_info(end_node)

        # Add start node and end node to relationships
        relationships_info.append(
            [start_node_info, end_node_info, relationship_type, rel_props]
        )

        # Add compounds to compound_info without repeating
        if start_node_info[1] == "Compound":
            compound_info[start_node_info[2]] = start_node_info[3]
            all_props.update(
                key for key in start_node_info[3].keys() if key != "CompoundName"
            )

        if end_node_info[1] == "Compound":
            compound_info[end_node_info[2]] = end_node_info[3]
            all_props.update(
                key for key in end_node_info[3].keys() if key != "CompoundName"
            )

    # Generate the HTML for compound summary tables for each group of nodes
    grouped_relationships = group_by_start_node(relationships_info)

    relationships_html = ""
    for group in grouped_relationships:
        start_node_info = group[0][0]  # Get the first relationship's start node info
        relationships_html += f"<h2 style='text-align: left;'>Similar Compounds to {start_node_info[2]}:</h2><ol>"
        for rel in group:
            relationships_html += f"<li>{rel[1][2]}</li>"  # rel[1][2] corresponds to the end node's name (CompoundName)
        relationships_html += "</ol>"

        # Generate compound summary table for each group
        table_html = "<table border='1' style='width:100%; border-collapse: collapse; word-wrap: break-word; table-layout: fixed;'>"

        # Create the header with compound names
        compound_names = list({start_node_info[2]} | {rel[1][2] for rel in group})
        table_html += (
            "<tr><th style='width: 200px;'>Property</th>"
            + "".join([f"<th>{name}</th>" for name in compound_names])
            + "</tr>"
        )

        # Populate the table by iterating over the properties and filling columns with the corresponding values for each compound
        for prop in all_props:
            row_html = f"<td>{prop}</td>"  # First column with property name
            for compound_name in compound_names:
                compound_props = compound_info.get(compound_name, {})
                value = compound_props.get(
                    prop, "N/A"
                )  # Get the property value or "N/A"
                row_html += f"<td>{value}</td>"
            table_html += f"<tr>{row_html}</tr>"

        table_html += "</table>"

        relationships_html += (
            f"<h2 style='text-align: left;'>Compounds Summary Table</h2>{table_html}"
        )

    # Combine the HTML report content
    report_html = f"""
    <html>
    <head>
        <title>CYP450-KG Summary Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            h1, h2, h5 {{ text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; word-wrap: break-word; table-layout: fixed; }}
            th, td {{ padding: 10px; border: 1px solid #dddddd; text-align: left; }}
            ul {{ list-style-type: none; padding-left: 0; }}
        </style>
    </head>
    <body>
        <h1>CYP450-KG Summary Report</h1>
        {relationships_html}
    </body>
    </html>
    """

    # elif action == "Show Related BioAssays":
    # elif action == "Co-Occurrence in Literature (Compound-Compound)":
    # elif action == "Co-Occurrence in Literature (Compound-Gene)":

    # elif action == "Compound-Gene Interactions (PubChem)":

    # elif action == "Compound-Gene Interactions (External Sources)":
    #     relationship_info = f"<h5>{start_node_info} -- studies --> </h5><h5>{end_node_info}</h5>"

    # Create a simple HTML structure for the report

    # Display the report in a new tab in Streamlit
    st.markdown(report_html, unsafe_allow_html=True)

    # Generate PDF from the report content
    download_html_as_pdf(report_html, "./CYP450-KG_Summary_Report.pdf")

    pdf_path = "./CYP450-KG_Summary_Report.pdf"
    # Read the PDF file into memory as bytes
    with open(pdf_path, "rb") as pdf_file:
        pdf_data = pdf_file.read()

    os.remove(pdf_path)

    # Provide the option to download the PDF in Streamlit
    st.markdown(
        """
        <style>
        /* Style the download button */
        div.stDownloadButton > button {
            background-color: #f0f0f0; /* Light gray */
            color: black; /* Button text color */
            border: 1px solid #d0d0d0; /* Border color */
            padding: 8px 20px; /* Padding */
            border-radius: 5px; /* Rounded corners */
            width: 100%; /* Full width */
            font-size: 24px; /* Font size */
            font-family: 'Arial', sans-serif; /* Font family */
        }

        /* Style the hover effect for the download button */
        div.stDownloadButton > button:hover {
            background-color: #b0b0b0; /* Darker gray on hover */
            color: white; /* Text color on hover */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Display the download button
    st.download_button(
        label="Download PDF Report",
        data=pdf_data,
        file_name="CYP450-KG_Summary_Report.pdf",
        mime="application/pdf",
    )


def display_neo4j_statistics(neo4j_conn):
    """
    Displays the Neo4j statistics in a table fixed at the bottom.

    Parameters:
        neo4j_conn (Neo4jConnection): Neo4j connection object.
    """
    # Query Neo4j for the statistics
    statistics = get_neo4j_statistics(neo4j_conn.driver)

    # Add the statistics section at the bottom of the graph with a fixed position
    fixed_statistics_html = f"""
    <div style="
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: white;
        border-top: 1px solid #f0f0f0;
        padding: 10px 0;
        z-index: 9999;
        box-shadow: 0px -4px 8px rgba(0,0,0,0.1);
    ">
        <div style="display: flex; justify-content: space-evenly; font-family: Arial, sans-serif;">
            <div>
                <h5 style="text-align: center;">Data</h5>
                <h5 style="text-align: center;">Statistics</h5>
            </div>
            <div>
                <br>
            </div>
            <div>
                <h5 style="text-align: center;">Compounds</h5>
                <h5 style="text-align: center;">{statistics['compounds_count']}</h5>
            </div>
            <div>
                <h5 style="text-align: center;">Genes</h5>
                <h5 style="text-align: center;">{statistics['genes_count']}</h5>
            </div>
            <div>
                <h5 style="text-align: center;">BioAssays</h5>
                <h5 style="text-align: center;">{statistics['bioassays_count']}</h5>
            </div>
            <div>
                <h5 style="text-align: center;">Relationships</h5>
                <h5 style="text-align: center;">{statistics['relationships_count']}</h5>
            </div>
        </div>
    </div>
    """

    # Display the fixed statistics at the bottom of the page
    st.markdown(fixed_statistics_html, unsafe_allow_html=True)
