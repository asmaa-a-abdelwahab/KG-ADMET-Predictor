import streamlit as st
from py2neo import Graph
import networkx as nx
import plotly.graph_objects as go
import joblib
import numpy as np

# Neo4j connection (adjust credentials as needed)
graph_db = Graph("bolt://localhost:7687", auth=("neo4j", "your_password"))

# Load your deep learning model (adjust the path to your model)
model = joblib.load("path_to_model.pkl")

# Streamlit app layout
st.title("Neo4j Graph Visualization and Interaction Prediction")

# Input form for querying the graph
compound = st.text_input("Enter a compound:")
gene = st.text_input("Enter a gene:")

# Step 1: Query Neo4j to retrieve the nodes and edges
if st.button("Show Graph"):
    query = """
    MATCH (c:Compound {name: $compound})-[r:INTERACTS_WITH]->(g:Gene {name: $gene})
    RETURN c, g, r
    """
    results = graph_db.run(query, compound=compound, gene=gene).data()

    if results:
        # Step 2: Create a NetworkX graph
        G = nx.Graph()
        pos = {}
        labels = {}
        node_colors = []

        # Add nodes and edges to the NetworkX graph
        for result in results:
            compound_node = result["c"]["name"]
            gene_node = result["g"]["name"]
            G.add_node(compound_node, color="blue")
            G.add_node(gene_node, color="orange")
            G.add_edge(compound_node, gene_node)

            # Set positions for Plotly scatter plot
            pos[compound_node] = (np.random.rand(), np.random.rand())
            pos[gene_node] = (np.random.rand(), np.random.rand())
            labels[compound_node] = compound_node
            labels[gene_node] = gene_node

        # Step 3: Interactive Visualization using Plotly
        node_x = []
        node_y = []
        node_text = []
        node_color = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_color.append("blue" if G.nodes[node]["color"] == "blue" else "orange")

        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            text=node_text,
            marker=dict(showscale=False, color=node_color, size=15, line_width=2),
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Neo4j Graph Visualization",
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=0, l=0, r=0, t=40),
                annotations=[
                    dict(
                        text="Click to select nodes",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False),
                yaxis=dict(showgrid=False, zeroline=False),
            ),
        )

        # Show the graph with Plotly
        selected_points = st.plotly_chart(fig, use_container_width=True)

        # Step 4: Select nodes for interaction prediction
        st.subheader("Select Two Nodes to Predict Interaction")
        selected_nodes = st.multiselect("Select nodes:", options=node_text)

        if len(selected_nodes) == 2:
            # Step 5: Predict interaction using the deep learning model
            compound_node = selected_nodes[0]
            gene_node = selected_nodes[1]

            # Prepare input data for your deep learning model (adjust based on your model's input format)
            input_data = np.array([[compound_node, gene_node]])  # Adjust this line

            # Make prediction
            prediction = model.predict(input_data)
            confidence_score = model.predict_proba(input_data)[0][1]

            # Display the prediction and confidence score
            st.write(
                f"Predicted interaction between {compound_node} and {gene_node}: {prediction}"
            )
            st.write(f"Confidence Score: {confidence_score:.2f}")
        else:
            st.write("Please select exactly two nodes.")
    else:
        st.write("No data found for the given compound and gene.")
