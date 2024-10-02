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
