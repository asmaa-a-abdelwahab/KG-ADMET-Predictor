import streamlit as st
import pandas as pd
from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
import joblib
from fpdf import FPDF

# Step 1: Set up Neo4j driver
def create_neo4j_driver(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver

# Step 2: Connect to Neo4j database
driver = create_neo4j_driver("neo4j://localhost:7687", "neo4j", "password")

# Step 3: Function to query compound-gene interactions from Neo4j
def get_compound_gene_interactions(driver, compound, gene):
    with driver.session() as session:
        query = """
        MATCH (c:Compound {name: $compound})-[r:INTERACTS_WITH]->(g:Gene {name: $gene})
        RETURN c.name as compound, g.name as gene, r.interaction_type as interaction_type
        """
        result = session.run(query, compound=compound, gene=gene)
        return result.data()

# Step 4: Function to visualize Neo4j graph data
def visualize_graph(interactions):
    G = nx.Graph()
    for interaction in interactions:
        compound = interaction['compound']
        gene = interaction['gene']
        G.add_node(compound, color='blue')
        G.add_node(gene, color='orange')
        G.add_edge(compound, gene)

    pos = nx.spring_layout(G)
    colors = ['blue' if G.nodes[n]['color'] == 'blue' else 'orange' for n in G.nodes]
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=1500)
    plt.show()

# Step 5: Function to generate a downloadable PDF report
def generate_pdf_report(compound, gene, prediction, confidence_score):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Compound-Gene Interaction Report", ln=True)
    pdf.cell(200, 10, txt=f"Compound: {compound}", ln=True)
    pdf.cell(200, 10, txt=f"Gene: {gene}", ln=True)
    pdf.cell(200, 10, txt=f"Prediction: {prediction}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence Score: {confidence_score}", ln=True)
    pdf.output("/mnt/data/report.pdf")

# Step 6: Load your trained deep learning model
model = joblib.load("path_to_model.pkl")

# Step 7: Streamlit app layout
st.title('CYP450-KG Application')

# Create tabs
tab1, tab2, tab3 = st.tabs(["Knowledge Graph", "Tabular Data", "Report"])

# Define a placeholder for the graph data and prediction report
graph_data = None
prediction_report = None

# Tab 1: Knowledge Graph Visualization and Interactions
with tab1:
    st.header("Knowledge Graph")
    
    # Dropdowns for selecting Compounds and Genes
    selected_compound = st.selectbox("Select Compound", ["Compound A", "Compound B"])
    selected_gene = st.selectbox("Select Gene", ["Gene 1", "Gene 2"])

    # Action buttons
    if st.button("Show Compound-Gene Interactions (PubChem)"):
        # Get the compound-gene interactions from Neo4j
        interactions = get_compound_gene_interactions(driver, selected_compound, selected_gene)
        st.write(interactions)
        graph_data = pd.DataFrame(interactions)  # Store the interactions for use in Tab 2
        
        # Visualize the interactions as a graph
        visualize_graph(interactions)
        st.pyplot(plt)

    # Button for predicting compound-gene interactions using the deep learning model
    if st.button("Predict Compound-Gene Interaction"):
        input_data = [selected_compound, selected_gene]  # Adjust based on your model's input format
        prediction = model.predict([input_data])[0]
        confidence_score = model.predict_proba([input_data])[0][1]
        st.write(f"Prediction: {prediction}")
        st.write(f"Confidence Score: {confidence_score}")
        
        # Generate report and save it
        generate_pdf_report(selected_compound, selected_gene, prediction, confidence_score)
        prediction_report = {
            "compound": selected_compound,
            "gene": selected_gene,
            "prediction": prediction,
            "confidence_score": confidence_score
        }

# Tab 2: Tabular Data View
with tab2:
    st.header("Tabular Data")
    # Display the dataframe of interactions in tabular form
    if graph_data is not None:
        st.dataframe(graph_data)
    else:
        st.write("No data to display yet.")

# Tab 3: Report
with tab3:
    st.header("Prediction Report")
    # Display the model prediction report if available
    if prediction_report is not None:
        st.write(f"Compound: {prediction_report['compound']}")
        st.write(f"Gene: {prediction_report['gene']}")
        st.write(f"Prediction: {prediction_report['prediction']}")
        st.write(f"Confidence Score: {prediction_report['confidence_score']}")
        
        # Provide link to download PDF report
        st.write("Download the report: [Download PDF](https://app.streamlit.io/mnt/data/report.pdf)")
    else:
        st.write("No report available yet.")

