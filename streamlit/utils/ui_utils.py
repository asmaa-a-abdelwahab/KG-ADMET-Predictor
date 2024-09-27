# ui_components.py

import streamlit as st
from utils.config import PAGE_ICON


def display_sidebar(compound_list, gene_list):
    # Sidebar title
    st.sidebar.markdown(
        '<p class="sidebar-title">CYP450-KG</p>', unsafe_allow_html=True
    )

    col1, col2, col3 = st.sidebar.columns([1, 1, 1])
    with col2:
        st.image(PAGE_ICON, width=120, use_column_width=False)

    # Sidebar dropdowns
    selected_compounds = st.sidebar.multiselect("Select Compound/s", compound_list)
    selected_genes = st.sidebar.multiselect("Select Gene/s", gene_list)

    return selected_compounds, selected_genes


def apply_custom_styles() -> None:
    """
    Apply custom CSS styles to the Streamlit app. This function adds styling for tabs, sidebar, and buttons.
    """
    st.markdown(
        """
        <style>
        /* Styling all tabs to be 33.33% wide */
        div[role="tablist"] > button {
            width: 33.33% !important;  /* Set the width of each tab to 33.33% */
            font-size: 20px !important;  /* Change the font size */
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

        /* Styling for the sidebar buttons */
        .sidebar-buttons button {
            margin-bottom: 10px !important;
            width: 100% !important;
            background-color: #ffffff !important;
            border: 2px solid #808080 !important;
            border-radius: 5px !important;
            color: #000000 !important;
        }

        .sidebar-buttons button:hover {
            background-color: #d3d3d3 !important;  /* Change the hover effect */
        }

        /* Sidebar title styling */
        .sidebar-title {
            font-size: 40px !important;  /* Increase font size */
            font-weight: bold !important;  /* Make the font bold */
            color: black !important;  /* Title color */
            text-align: center !important; /* Center the title */
        }
        
        /* Change the width of the sidebar */
        .css-1d391kg {
            width: 300px;  /* Adjust this to your desired width */
        }

        /* Change the width of the sidebar's content */
        .css-1d391kg .css-1lcbmhc {
            width: 300px;  /* Adjust this to your desired width */
        }
        </style>
    """,
        unsafe_allow_html=True,
    )
