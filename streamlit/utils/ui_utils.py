# ui_components.py
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
from utils.config import PAGE_ICON


APP_NAME = "CYP450-KG Explorer"

ACTION_OPTIONS = [
    "Show Similar Compounds",
    "Show Related BioAssays",
    "Co-Occurrence in Literature (Compound-Compound)",
    "Evidence Co-Occurrence (Compound-Target)",
    "Compound-Target Interactions (PubChem)",
    "Compound-Target Enrichment Context",
]

TARGET_REQUIRED_ACTIONS = {
    "Evidence Co-Occurrence (Compound-Target)",
    "Compound-Target Interactions (PubChem)",
    "Compound-Target Enrichment Context",
}

ACTION_DESCRIPTIONS = {
    "Show Similar Compounds": "Find chemical neighbours connected by SIMILAR_TO. Use this to identify analogues for follow-up, not as direct CYP450 activity evidence.",
    "Show Related BioAssays": "Trace assay provenance for selected compounds through BioAssay, MeasureGrp, Endpoint, Substance, and Compound evidence paths.",
    "Co-Occurrence in Literature (Compound-Compound)": "Find compounds that share evidence context such as assays, references, interactions, or targets. Useful for grouping candidates before manual review.",
    "Evidence Co-Occurrence (Compound-Target)": "Inspect compound-target evidence paths for selected compounds and CYP450 proteins, including interaction assertions and support where available.",
    "Compound-Target Interactions (PubChem)": "Show direct PRING Interaction nodes asserting a selected compound-CYP450 target relationship and their supporting evidence.",
    "Compound-Target Enrichment Context": "Expand selected compound-target pairs with chemical properties, structures, GO/Reactome/InterPro/PDB/AlphaFold, and other feature context.",
}

ACTION_BADGES = {
    "Show Similar Compounds": "Compound only",
    "Show Related BioAssays": "Compound only",
    "Co-Occurrence in Literature (Compound-Compound)": "Compound only",
    "Evidence Co-Occurrence (Compound-Target)": "Requires targets",
    "Compound-Target Interactions (PubChem)": "Requires targets",
    "Compound-Target Enrichment Context": "Requires targets",
}

ACTION_QUESTION = {
    "Show Similar Compounds": "Which compounds are chemically close to my selection?",
    "Show Related BioAssays": "What assay evidence exists for these compounds?",
    "Co-Occurrence in Literature (Compound-Compound)": "Which compounds share evidence context?",
    "Evidence Co-Occurrence (Compound-Target)": "Do my compounds and targets appear together in evidence paths?",
    "Compound-Target Interactions (PubChem)": "Which direct compound-target interaction assertions are present?",
    "Compound-Target Enrichment Context": "What feature and annotation context surrounds the pair?",
}


def _short_option_label(value: str, max_length: int = 38) -> str:
    """Shorten long Streamlit option labels while preserving searchable values."""
    value = str(value)
    return value if len(value) <= max_length else value[: max_length - 1] + "…"


def _logo_data_uri() -> str:
    """Return the app icon as a data URI when available."""
    icon_path = Path(PAGE_ICON)
    if not icon_path.exists():
        return ""
    mime = "image/webp" if icon_path.suffix.lower() == ".webp" else "image/png"
    encoded = base64.b64encode(icon_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def display_sidebar(compound_list, gene_list):
    """Render sidebar branding and a guided single-submit query workflow.

    The analysis selector is intentionally placed before the form so users can
    read the action explanation and see whether target selection is required
    before choosing compounds/targets. The Neo4j query still runs only when the
    user clicks the Run analysis button inside the form.
    """
    logo_uri = _logo_data_uri()
    logo_html = (
        f'<img src="{logo_uri}" alt="CYP450-KG Explorer logo" class="kg-brand-logo">'
        if logo_uri
        else '<div class="kg-brand-logo-fallback">KG</div>'
    )

    st.sidebar.markdown(
        f"""
        <div class="kg-sidebar-brand">
            <div class="kg-brand-row">
                <div class="kg-brand-mark">{logo_html}</div>
                <div class="kg-brand-copy">
                    <div class="kg-brand-eyebrow">PRING knowledge graph</div>
                    <div class="kg-brand-title">{APP_NAME}</div>
                </div>
            </div>
            <a class="kg-doi-pill" href="https://doi.org/10.5281/zenodo.15323478" target="_blank">
                <span>DOI</span>10.5281/zenodo.15323478
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div class="kg-sidebar-step-title kg-step-analysis">
            <span>1</span><strong>Choose the question</strong>
        </div>
        <p class="kg-sidebar-help">
            Start with the scientific question. The short guide below helps users choose the right analysis before selecting targets.
        </p>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar.expander("Compare analysis actions", expanded=False):
        guide_cards = []
        for item in ACTION_OPTIONS:
            required_class = "kg-action-target" if item in TARGET_REQUIRED_ACTIONS else "kg-action-compound"
            guide_cards.append(
                f"""
                <div class="kg-action-guide-card {required_class}">
                    <div><strong>{ACTION_QUESTION[item]}</strong><span>{ACTION_BADGES[item]}</span></div>
                    <p>{ACTION_DESCRIPTIONS[item]}</p>
                </div>
                """
            )
        st.markdown("".join(guide_cards), unsafe_allow_html=True)

    action = st.sidebar.selectbox(
        "Select analysis action",
        ACTION_OPTIONS,
        key="form_action",
        format_func=lambda value: _short_option_label(value, 44),
        help="The graph query is not executed until you click Run analysis.",
    )
    target_required = action in TARGET_REQUIRED_ACTIONS

    if not target_required and st.session_state.get("form_selected_genes"):
        st.session_state["form_selected_genes"] = []

    with st.sidebar.form("kg_query_form", clear_on_submit=False):
        st.markdown(
            """
            <div class="kg-sidebar-step-title">
                <span>2</span><strong>Select compounds</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="kg-sidebar-help kg-sidebar-help-tight">Choose up to five PubChem compounds.</p>',
            unsafe_allow_html=True,
        )
        selected_compounds = st.multiselect(
            "Choose up to five PubChem compounds",
            compound_list,
            max_selections=5,
            key="form_selected_compounds",
            format_func=lambda value: _short_option_label(value, 42),
            placeholder="Search by compound name or CID",
        )

        selected_genes = []
        if target_required:
            st.markdown(
                """
                <div class="kg-sidebar-step-title kg-target-step-title">
                    <span>3</span><strong>Select CYP450 targets</strong>
                </div>
                <p class="kg-sidebar-help kg-sidebar-help-tight">Required for this compound-target analysis.</p>
                """,
                unsafe_allow_html=True,
            )
            selected_genes = st.multiselect(
                "Search CYP450 targets",
                gene_list,
                key="form_selected_genes",
                placeholder="Search by CYP symbol or UniProt accession",
                label_visibility="collapsed",
            )
        else:
            st.markdown(
                """
                <div class="kg-target-not-needed">
                    <strong>CYP450 targets are not needed for this action.</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        submitted = st.form_submit_button("Run analysis", use_container_width=True, type="primary")

    if not target_required:
        selected_genes = []

    return submitted, selected_compounds, selected_genes, action


def apply_custom_styles() -> None:
    """Apply a professional unified theme for the Streamlit app."""
    st.markdown(
        """
        <style>
        :root {
            --kg-red: #EF4444;
            --kg-charcoal: #505050;
            --kg-dark: #111827;
            --kg-mid: #808080;
            --kg-bg: #F3F6FA;
            --kg-panel: #FFFFFF;
            --kg-border: #E2E8F0;
            --kg-muted: #64748B;
            --kg-blue: #4B5563;
        }

        /* App background and typography */
        .stApp {
            background: linear-gradient(180deg, #F8FAFC 0%, #EEF2F7 100%);
            color: var(--kg-dark);
        }
        html, body, [class*="css"] {
            font-family: Arial, sans-serif;
        }


        /* Hide Streamlit development chrome for a cleaner deployed/demo view */
        #MainMenu,
        footer,
        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stDeployButton"],
        .stDeployButton {
            display: none !important;
            visibility: hidden !important;
        }
        .block-container {
            padding-top: 1.2rem !important;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #EEF2F7 !important;
            border-right: 1px solid var(--kg-border);
        }
        section[data-testid="stSidebar"] > div {
            padding-top: 1.4rem;
        }
        .kg-sidebar-brand {
            margin: 0 0 28px 0;
            padding: 18px 16px;
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(255,255,255,.94), rgba(248,250,252,.94));
            border: 1px solid var(--kg-border);
            box-shadow: 0 14px 36px rgba(17, 24, 39, .08);
        }
        .kg-brand-row {
            display: flex;
            gap: 14px;
            align-items: center;
        }
        .kg-brand-mark {
            flex: 0 0 auto;
            width: 72px;
            height: 72px;
            border-radius: 20px;
            background: #FFFFFF;
            border: 1px solid #D7DEE8;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: inset 0 0 0 4px #F8FAFC;
            overflow: hidden;
        }
        .kg-brand-logo {
            width: 58px;
            height: 58px;
            object-fit: contain;
            display: block;
        }
        .kg-brand-logo-fallback {
            width: 58px;
            height: 58px;
            border-radius: 16px;
            background: var(--kg-dark);
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            letter-spacing: .04em;
        }
        .kg-brand-copy { min-width: 0; }
        .kg-brand-eyebrow {
            color: var(--kg-red);
            font-size: 11px;
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: 3px;
        }
        .kg-brand-title {
            color: #4B5563;
            font-size: 21px;
            font-weight: 900;
            line-height: 1.12;
            letter-spacing: -.02em;
        }
        .kg-doi-pill {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            margin-top: 14px;
            padding: 10px 12px;
            width: 100%;
            box-sizing: border-box;
            border-radius: 14px;
            background: #FFFFFF;
            border: 1px solid #D7DEE8;
            color: #4B5563 !important;
            text-decoration: none !important;
            font-size: 12.5px;
            font-weight: 800;
            box-shadow: 0 3px 10px rgba(17,24,39,.05);
        }
        .kg-doi-pill span {
            background: #6B7280;
            color: #FFFFFF;
            padding: 3px 7px;
            border-radius: 999px;
            font-size: 11px;
        }

        .kg-sidebar-footer-compact {
            margin-top: 8px;
            padding: 8px 10px;
            border-radius: 14px;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 14px rgba(17, 24, 39, .04);
        }

        .kg-github-row {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 7px;
            text-decoration: none !important;
            margin-bottom: 6px;
        }

        .kg-github-row img {
            width: 24px;
            height: 24px;
            display: block;
        }

        .kg-github-row span {
            color: #111827;
            font-size: 13px;
            font-weight: 800;
            line-height: 1.2;
        }

        .kg-footer-note {
            margin: 0;
            padding-top: 6px;
            border-top: 1px solid #E5E7EB;
            text-align: center;
        }

        .kg-footer-note span {
            color: #475569;
            font-size: 11.5px;
            line-height: 1.35;
        }

        .kg-footer-note a {
            color: #EF4444 !important;
            text-decoration: none !important;
            font-weight: 900;
        }

        /* Sidebar form labels and controls */
        section[data-testid="stSidebar"] label {
            color: #334155 !important;
            font-weight: 700 !important;
            font-size: 14px !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] > div {
            border-radius: 12px !important;
            border-color: #E2E8F0 !important;
            background: #FFFFFF !important;
            min-height: 48px;
        }

        section[data-testid="stSidebar"] form {
            border: 0 !important;
            padding: 0 !important;
            background: transparent !important;
        }

        /* Result view selector: tab-like radio buttons with no default selection */
        div[data-testid="stRadio"] {
            margin: 0 0 22px 0;
            padding-bottom: 6px;
            border-bottom: 1px solid #D1D5DB;
        }
        div[data-testid="stRadio"] > label {
            display: none !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] {
            display: grid !important;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
            width: 100%;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label {
            background: var(--kg-mid) !important;
            color: #FFFFFF !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 13px 10px !important;
            min-height: 48px;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 16px !important;
            font-weight: 800 !important;
            border: 0 !important;
            cursor: pointer;
            transition: all .15s ease-in-out;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
            background: #666666 !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
            background: var(--kg-charcoal) !important;
            box-shadow: inset 0 -4px 0 var(--kg-red) !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label p {
            color: #FFFFFF !important;
            font-weight: 800 !important;
            text-align: center !important;
        }


        /* Landing page */
        .kg-landing-hero {
            margin: 28px 0 22px;
            padding: 34px 38px;
            border-radius: 24px;
            background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 52%, #EEF2F7 100%);
            border: 1px solid var(--kg-border);
            box-shadow: 0 18px 42px rgba(17,24,39,.08);
            text-align: center;
        }
        .kg-hero-eyebrow {
            color: var(--kg-red);
            text-transform: uppercase;
            letter-spacing: .14em;
            font-size: 12px;
            font-weight: 900;
            margin-bottom: 10px;
        }
        .kg-landing-hero h1 {
            margin: 0;
            color: #374151;
            font-size: clamp(32px, 4vw, 52px);
            font-weight: 900;
            letter-spacing: -.04em;
        }
        .kg-landing-hero p {
            margin: 14px auto 0;
            color: #374151;
            max-width: 980px;
            font-size: 18px;
            line-height: 1.7;
        }
        .kg-setup-callout {
            margin: 24px auto 0;
            max-width: 980px;
            padding: 16px 18px;
            border-radius: 14px;
            background: #DBEAFE;
            border: 1px solid #BFDBFE;
            color: #1E3A8A;
            text-align: left;
            line-height: 1.6;
        }
        .kg-setup-callout strong {
            display: block;
            margin-bottom: 4px;
        }
        .kg-howto-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 16px;
            margin: 20px 0;
        }
        .kg-howto-card {
            background: #FFFFFF;
            border: 1px solid var(--kg-border);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 10px 28px rgba(17,24,39,.06);
        }
        .kg-howto-card {
            position: relative;
            background: #FFFFFF;
            border: 1px solid var(--kg-border);
            border-radius: 18px;
            padding: 22px 22px 24px;
            box-shadow: 0 10px 28px rgba(17,24,39,.06);
        }

        /* Remove any old decorative empty circles */
        .kg-howto-card::before,
        .kg-howto-card::after,
        .kg-howto-card h3::before,
        .kg-howto-card h3::after {
            content: none !important;
            display: none !important;
        }

        .kg-howto-title {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        }

        .kg-step-badge {
            width: 38px;
            height: 38px;
            min-width: 38px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            background: #EF4444;
            color: #FFFFFF;
            font-weight: 900;
            font-size: 16px;
            line-height: 1;
            box-shadow: 0 8px 18px rgba(239, 68, 68, .22);
        }

        .kg-howto-title h3 {
            margin: 0;
            color: #111827;
            font-size: 19px;
            font-weight: 900;
            line-height: 1.25;
        }

        .kg-howto-card p {
            margin: 0;
            color: #475569;
            line-height: 1.65;
            font-size: 14.5px;
        }

        .kg-stat-panel {
            margin: 22px 0 8px;
            padding: 20px;
            background: #FFFFFF;
            border: 1px solid var(--kg-border);
            border-radius: 20px;
            box-shadow: 0 14px 34px rgba(17,24,39,.07);
            display: grid;
            grid-template-columns: 1.1fr 2.2fr;
            gap: 20px;
            align-items: center;
        }
        .kg-stat-panel h3 {
            margin: 0 0 6px;
            color: #111827;
            font-size: 22px;
        }
        .kg-stat-panel p {
            margin: 0;
            color: #64748B;
        }
        .kg-stat-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
        }
        .kg-stat-grid div {
            padding: 16px;
            border-radius: 16px;
            background: #F8FAFC;
            border: 1px solid #E5E7EB;
            text-align: center;
        }
        .kg-stat-grid strong {
            display: block;
            color: #111827;
            font-size: 24px;
            font-weight: 900;
            margin-bottom: 5px;
        }
        .kg-stat-grid span {
            color: #64748B;
            font-size: 13px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .04em;
        }
        @media (max-width: 1100px) {
            .kg-howto-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .kg-stat-panel { grid-template-columns: 1fr; }
        }
        @media (max-width: 760px) {
            div[data-testid="stRadio"] div[role="radiogroup"],
            .kg-howto-grid,
            .kg-stat-grid { grid-template-columns: 1fr; }
        }
        /* Native Streamlit result tabs. These do not run another Cypher query when the user switches views. */
        div[data-testid="stTabs"] {
            margin: 0 0 22px 0;
            padding-bottom: 6px;
            border-bottom: 1px solid #D1D5DB;
        }
        div[data-testid="stTabs"] button[role="tab"] {
            background: var(--kg-mid) !important;
            color: #FFFFFF !important;
            border-radius: 8px 8px 0 0 !important;
            min-height: 48px !important;
            font-size: 16px !important;
            font-weight: 800 !important;
            border: 0 !important;
            transition: all .15s ease-in-out;
        }
        div[data-testid="stTabs"] button[role="tab"] p {
            color: #FFFFFF !important;
            font-weight: 800 !important;
        }
        div[data-testid="stTabs"] button[role="tab"]:hover {
            background: #666666 !important;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            background: var(--kg-charcoal) !important;
            box-shadow: inset 0 -4px 0 var(--kg-red) !important;
        }
        div[data-testid="stTabs"] div[role="tablist"] {
            display: grid !important;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
        }


        /* Buttons */
        div.stButton > button,
        div[data-testid="stDownloadButton"] > button {
            background: #D1D5DB;
            border: 1px solid #CBD5E1;
            color: #111827;
            padding: 0.7rem 1rem;
            text-align: center;
            display: block;
            font-size: 16px;
            font-weight: 800;
            cursor: pointer;
            width: 100%;
            border-radius: 10px;
            transition: all .15s ease-in-out;
        }
        div.stButton > button:hover,
        div[data-testid="stDownloadButton"] > button:hover {
            background: var(--kg-charcoal);
            border-color: var(--kg-charcoal);
            color: #FFFFFF;
        }
        div.stButton > button:focus,
        div[data-testid="stDownloadButton"] > button:focus {
            box-shadow: 0 0 0 3px rgba(239,68,68,.18) !important;
            outline: none !important;
        }



        /* Guided sidebar analysis workflow */
        .kg-sidebar-step-title {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 8px 0 8px;
            color: #111827;
        }
        .kg-sidebar-step-title span {
            width: 28px;
            height: 28px;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: var(--kg-red);
            color: #FFFFFF;
            font-weight: 900;
            font-size: 13px;
            box-shadow: 0 8px 18px rgba(239,68,68,.22);
        }
        .kg-sidebar-step-title strong {
            font-size: 15px;
            font-weight: 900;
            color: #111827;
        }
        .kg-sidebar-help {
            margin: 0 0 12px;
            color: #64748B;
            font-size: 13px;
            line-height: 1.55;
        }
        .kg-action-guide-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #9CA3AF;
            border-radius: 14px;
            padding: 12px 12px;
            margin-bottom: 10px;
            box-shadow: 0 4px 14px rgba(17,24,39,.05);
        }
        .kg-action-guide-card.kg-action-target { border-left-color: var(--kg-red); }
        .kg-action-guide-card.kg-action-compound { border-left-color: #64748B; }
        .kg-action-guide-card div {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 8px;
        }
        .kg-action-guide-card strong {
            color: #111827;
            font-size: 13.5px;
            line-height: 1.35;
        }
        .kg-action-guide-card span {
            flex: 0 0 auto;
            border-radius: 999px;
            padding: 4px 7px;
            background: #F3F4F6;
            color: #4B5563;
            font-size: 10.5px;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: .03em;
        }
        .kg-action-guide-card.kg-action-target span {
            background: #FEE2E2;
            color: #991B1B;
        }
        .kg-action-guide-card p {
            margin: 8px 0 0;
            color: #475569;
            font-size: 12.8px;
            line-height: 1.52;
        }
        .kg-selected-action-card {
            margin: 12px 0 18px;
            padding: 14px;
            background: linear-gradient(180deg, #FFFFFF, #F8FAFC);
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            box-shadow: 0 8px 22px rgba(17,24,39,.06);
        }
        .kg-selected-action-card small {
            display: block;
            color: var(--kg-red);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: .08em;
            font-weight: 900;
            margin-bottom: 6px;
        }
        .kg-selected-action-card strong {
            display: block;
            color: #111827;
            line-height: 1.35;
            font-size: 14px;
        }
        .kg-selected-action-card p {
            margin: 8px 0 10px;
            color: #475569;
            line-height: 1.55;
            font-size: 13px;
        }
        .kg-selected-action-card span,
        .kg-compound-only,
        .kg-requires-target {
            display: inline-flex;
            width: fit-content;
            border-radius: 999px;
            padding: 5px 9px;
            font-size: 11px;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: .03em;
        }
        .kg-compound-only { background: #F3F4F6; color: #374151; }
        .kg-requires-target { background: #FEE2E2; color: #991B1B; }
        .kg-target-not-needed,
        .kg-run-panel,
        .kg-current-query {
            padding: 13px 14px;
            border-radius: 14px;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            box-shadow: 0 4px 14px rgba(17,24,39,.04);
            margin: 8px 0 14px;
        }
        .kg-target-not-needed strong,
        .kg-run-panel strong {
            display: block;
            color: #111827;
            font-size: 13.5px;
            margin-bottom: 4px;
        }
        .kg-target-not-needed span,
        .kg-run-panel span,
        .kg-run-panel small,
        .kg-current-query small {
            display: block;
            color: #64748B;
            font-size: 12.5px;
            line-height: 1.5;
        }
        .kg-current-query p {
            margin: 0 0 4px;
            color: #111827;
        }

        /* Give controls breathing room */
        section[data-testid="stSidebar"] .element-container {
            margin-bottom: 6px;
        }
        

        /* Balanced sidebar spacing: compact but non-overlapping. */
        section[data-testid="stSidebar"] > div {
            padding-top: 1.05rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        section[data-testid="stSidebar"] .element-container {
            margin-bottom: 8px !important;
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.45rem !important;
        }
        section[data-testid="stSidebar"] form,
        section[data-testid="stSidebar"] [data-testid="stForm"] {
            border: 1px solid #CBD5E1 !important;
            border-radius: 15px !important;
            background: rgba(255,255,255,0.30) !important;
            padding: 14px 14px 16px !important;
            margin-top: 12px !important;
            margin-bottom: 12px !important;
            box-shadow: none !important;
        }
        .kg-sidebar-brand {
            margin-bottom: 20px !important;
            padding: 16px 14px !important;
            border-radius: 20px !important;
        }
        .kg-brand-row { gap: 12px !important; }
        .kg-brand-mark {
            width: 64px !important;
            height: 64px !important;
            border-radius: 18px !important;
        }
        .kg-brand-logo,
        .kg-brand-logo-fallback {
            width: 52px !important;
            height: 52px !important;
        }
        .kg-brand-eyebrow {
            font-size: 9.5px !important;
            margin-bottom: 2px !important;
        }
        .kg-brand-title {
            font-size: 18px !important;
            line-height: 1.12 !important;
        }
        .kg-doi-pill {
            margin-top: 10px !important;
            padding: 8px 10px !important;
            font-size: 11px !important;
            border-radius: 12px !important;
        }
        .kg-sidebar-step-title {
            margin: 9px 0 7px !important;
            gap: 9px !important;
        }
        .kg-step-analysis {
            margin-top: 4px !important;
        }
        .kg-sidebar-step-title span {
            width: 26px !important;
            height: 26px !important;
            min-width: 26px !important;
            font-size: 12px !important;
            box-shadow: 0 5px 12px rgba(239,68,68,.18) !important;
        }
        .kg-sidebar-step-title strong {
            font-size: 13.8px !important;
            line-height: 1.25 !important;
        }
        .kg-sidebar-help {
            margin: 0 0 8px !important;
            font-size: 12px !important;
            line-height: 1.38 !important;
        }
        .kg-sidebar-help-tight {
            margin-bottom: 7px !important;
        }
        section[data-testid="stSidebar"] label {
            color: #334155 !important;
            font-size: 12.6px !important;
            font-weight: 700 !important;
            margin-bottom: 3px !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] > div {
            min-height: 43px !important;
            max-height: 104px !important;
            overflow-y: auto !important;
            align-items: flex-start !important;
            border-radius: 12px !important;
            border-color: #E2E8F0 !important;
            background: #FFFFFF !important;
            padding-top: 4px !important;
            padding-bottom: 4px !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="tag"] {
            max-width: 210px !important;
            min-height: 25px !important;
            margin: 2px 4px 2px 0 !important;
            border-radius: 7px !important;
            line-height: 1.1 !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="tag"] span {
            max-width: 170px !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stMultiSelect"] {
            margin-bottom: 14px !important;
        }
        .kg-target-step-title {
            margin-top: 12px !important;
        }
        .kg-target-not-needed {
            margin: 10px 0 12px !important;
            padding: 10px 12px !important;
            border-radius: 12px !important;
            box-shadow: none !important;
        }
        .kg-target-not-needed strong {
            font-size: 12.2px !important;
            line-height: 1.35 !important;
            margin: 0 !important;
        }
        .kg-action-guide-card {
            padding: 8px 9px !important;
            margin-bottom: 6px !important;
            border-radius: 10px !important;
        }
        .kg-action-guide-card strong { font-size: 12.1px !important; line-height: 1.25 !important; }
        .kg-action-guide-card p { margin-top: 5px !important; font-size: 11.5px !important; line-height: 1.32 !important; }
        .kg-action-guide-card span { padding: 3px 6px !important; font-size: 9px !important; }
        section[data-testid="stSidebar"] div.stButton > button[kind="primary"],
        section[data-testid="stSidebar"] div.stButton > button {
            min-height: 44px !important;
            padding: 0.55rem 0.8rem !important;
            font-size: 14px !important;
            border-radius: 10px !important;
            margin-top: 8px !important;
        }
        .kg-sidebar-footer-compact {
            margin-top: 12px !important;
            padding: 8px 9px !important;
            border-radius: 13px !important;
        }
        .kg-github-row { margin-bottom: 5px !important; gap: 6px !important; }
        .kg-github-row img { width: 20px !important; height: 20px !important; }
        .kg-github-row span { font-size: 11.7px !important; }
        .kg-footer-note { padding-top: 5px !important; }
        .kg-footer-note span { font-size: 10.5px !important; line-height: 1.28 !important; }
        .kg-table-intro {
            margin: 8px 0 16px;
            padding: 13px 15px;
            border: 1px solid #E2E8F0;
            border-left: 4px solid var(--kg-red);
            border-radius: 14px;
            background: #FFFFFF;
            box-shadow: 0 8px 22px rgba(17, 24, 39, .05);
        }
        .kg-table-intro strong {
            display: block;
            color: #111827;
            font-size: 15px;
            margin-bottom: 4px;
        }
        .kg-table-intro span {
            display: block;
            color: #475569;
            font-size: 13px;
            line-height: 1.45;
        }
</style>
        """,
        unsafe_allow_html=True,
    )
