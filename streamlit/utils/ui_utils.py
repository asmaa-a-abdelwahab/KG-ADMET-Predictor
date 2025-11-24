# ui_components.py
import base64
import streamlit as st
from utils.config import PAGE_ICON


def display_sidebar(compound_list, gene_list):
    # Sidebar title
    file_ = open(PAGE_ICON, "rb").read()
    base64_image = base64.b64encode(file_).decode("utf-8")
    st.sidebar.markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: center; padding-bottom: 10px;">
            <!-- Logo -->
            <div style="display: flex; align-items: center; margin-right: 10px;">
                <img src="data:image/png;base64,{base64_image}" alt="Logo" width="100" style="border-radius: 5px;">
            </div>
            <!-- Separator -->
            <div style="width: 4px; height: 50px; background-color: #ccc; margin-right: 10px;"></div>
            <!-- Text -->
            <div style="font-size: 34px; font-weight: bold; color: #112f5f;">
                CYP450-KG
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: center;">
            <a href="https://doi.org/10.5281/zenodo.15323478">
                <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.15323478.svg" alt="DOI">
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.write("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")
    # st.sidebar.markdown(
    #     '<p class="sidebar-title">CYP450-KG</p>', unsafe_allow_html=True
    # )

    # col1, col2, col3 = st.sidebar.columns([1, 1, 1])
    # with col2:
    #     st.image(PAGE_ICON, width=120, use_column_width=False)

    # Sidebar dropdowns
    selected_compounds = st.sidebar.multiselect(
        "Select up to 5 Compounds", compound_list, max_selections=5
    )
    selected_genes = st.sidebar.multiselect("Select Genes of Interest", gene_list)

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

        /* Increase font size for Select Compound/s and Select Gene/s */
        label[for="Select Compound/s"] {
            font-size: 30px !important;  /* Increase font size for compound selector */
            font-weight: bold !important; /* Make the font bold */
        }

        label[for="Select Gene/s"] {
            font-size: 30px !important;  /* Increase font size for gene selector */
            font-weight: bold !important; /* Make the font bold */
        }

        /* Change the width of the sidebar */
        .css-1d391kg {
            width: 460px;  /* Adjust this to your desired width */
        }

        /* Change the width of the sidebar's content */
        .css-1d391kg .css-1lcbmhc {
            width: 460px;  /* Adjust this to your desired width */
        }

        /* Custom submit button styling */
        div.stButton > button {
            background-color: #d3d3d3; /* Light gray */
            border: none;
            color: black;
            padding: 10px 20px;
            text-align: center;
            display: block;
            font-size: 30px;
            cursor: pointer;
            width: 100%; /* Full width */
            border-radius: 5px; /* Rounded corners */
        }
        
        div.download_button > button:hover {
            background-color: #b0b0b0; /* Darker gray on hover */
        }
        
        /* Add margin below gene multi-select */
        .element-container:nth-of-type(4), .element-container:nth-of-type(6) {
            margin-bottom: 25px !important;  /* Increase space below the multi-select */
        }
        
        .element-container:nth-of-type(3) {
            margin-top: 30px !important;
        }
        
        .st-emotion-cache-ue6h4q {
            font-size: 18px !important;
        }
        .sidebar-title {
            margin-top: 25px !important;
        }
        
        </style>
        """,
        unsafe_allow_html=True,
    )
