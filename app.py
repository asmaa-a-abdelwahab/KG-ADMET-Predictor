import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Set Streamlit to use the wide layout
st.set_page_config(layout="wide")

# Custom CSS for controlling each tab's width and text style
st.markdown(
    """
    <style>
    /* Styling all tabs to be 33.33% wide */
    div[role="tablist"] > button {
        width: 40% !important;  /* Set the width of each tab to 33.33% */
        font-size: 60px !important;  /* Change the font size */
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
    </style>
    """,
    unsafe_allow_html=True,
)


# Streamlit app layout
# Create tabs with full page layout and custom widths and heights
tab1, tab2, tab3 = st.tabs(["Knowledge Graph", "Tabular Data", "Report"])

# Define a placeholder for the graph data and prediction report
graph_data = None
prediction_report = None

# Tab 1: Knowledge Graph Visualization and Interactions
with tab1:
    st.header("Knowledge Graph")

    # Sidebar for this tab only
    with st.sidebar:
        # Inject custom CSS for styling the title
        st.sidebar.markdown(
            """
            <style>
            .sidebar-title {
                font-size: 40px;  /* Increase font size */
                font-weight: bold;  /* Optional: Make the font bold */
                color: black;  /* Optional: Change the color */
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Use the class 'sidebar-title' in your markdown to apply the custom style
        st.sidebar.markdown(
            '<p class="sidebar-title">CYP450-KG</p>', unsafe_allow_html=True
        )

        # Sidebar for Configuration
        st.sidebar.image("images/kg_icon.webp", width=150, use_column_width=False)

        # Sidebar for input fields
        st.sidebar.write("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
        # Adding inputs in the sidebar
        neo4j_host = st.text_input("Neo4j Host", value="localhost")
        neo4j_port = st.text_input("Neo4j Port", value="7687")
        neo4j_user = st.text_input("Neo4j Username", value="neo4j")
        neo4j_password = st.text_input("Neo4j Password", type="password")

        # Dropdown for compound selection
        selected_compound = st.selectbox(
            "Select Compound", ["Compound A", "Compound B"]
        )

        # Dropdown for gene selection
        selected_gene = st.selectbox("Select Gene", ["Gene 1", "Gene 2"])

    # Action buttons
    if st.sidebar.button("Show Compound-Gene Interactions (PubChem)"):
        # Placeholder for showing interactions (Replace with actual interactions)
        st.write(f"Showing interactions for {selected_compound} and {selected_gene}")
        graph_data = pd.DataFrame(
            {
                "compound": [selected_compound],
                "gene": [selected_gene],
                "interaction_type": ["Sample Interaction"],
            }
        )
        st.write(graph_data)

        # Placeholder graph (Replace with actual graph logic)
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])  # Example graph
        st.pyplot(fig)

    # Button for predicting compound-gene interactions using the deep learning model
    if st.sidebar.button("Predict Compound-Gene Interaction"):
        # Placeholder for prediction logic (Replace with actual prediction logic)
        st.write(f"Predicted interaction for {selected_compound} and {selected_gene}")
        st.write("Confidence Score: 0.85")
        prediction_report = {
            "compound": selected_compound,
            "gene": selected_gene,
            "prediction": "Sample Prediction",
            "confidence_score": 0.85,
        }

st.sidebar.write("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
st.sidebar.markdown(
    """
    <div style="position: absolute; bottom: 10; width: 100%; text-align: left;">
        <p style="font-size:14px; color:black;">BY: 
            <a href="https://github.com/asmaa-a-abdelwahab" target="_blank" style="font-size:14px; color:black;">Asmaa A. Abdelwahab</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
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

        # Placeholder for PDF download link (Replace with actual PDF generation logic)
        st.write("Download the report: [Download PDF](#)")
    else:
        st.write("No report available yet.")
