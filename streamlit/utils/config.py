# config.py

import logging

# Streamlit app settings
PAGE_TITLE = "CYP450-KG Streamlit App"
PAGE_ICON = "images/kg_icon.webp"

# Neo4j configuration
NEO4J_URI = "bolt://neo4j:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "cyp450kg"

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
