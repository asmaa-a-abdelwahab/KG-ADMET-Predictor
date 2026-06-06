# config.py
from __future__ import annotations

import logging
import os
from pathlib import Path

# Streamlit app settings kept from the older CYP450-KG implementation.
PAGE_TITLE = os.getenv("STREAMLIT_PAGE_TITLE", "CYP450-KG Explorer")
PAGE_ICON = os.getenv("STREAMLIT_PAGE_ICON", "images/kg_icon.webp")
PRING_REPO_URL = os.getenv("PRING_REPO_URL", "https://github.com/asmaa-a-abdelwahab/PRING")

# Neo4j configuration. Defaults match docker-compose service names, but can be
# overridden when running the older app locally outside Docker.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "cyp450kg")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Query limits keep the interactive PyVis view responsive on large PRING runs.
APP_OPTION_LIMIT = int(os.getenv("APP_OPTION_LIMIT", "5000"))
APP_RESULT_LIMIT = int(os.getenv("APP_RESULT_LIMIT", "250"))
APP_GRAPH_LIMIT = int(os.getenv("APP_GRAPH_LIMIT", "150"))

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
