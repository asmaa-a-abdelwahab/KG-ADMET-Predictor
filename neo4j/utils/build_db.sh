#!/bin/bash

# Downloads nodes and relationships from Google Drive and imports them into
# the Cypher database.
#
# The `gdown` command is used to download the nodes and relationships CSV
# files from Google Drive. The `import_nodes.py` and `import_relationships.py`
# scripts are then used to import the nodes and relationships into the Cypher
# database.

readonly NODES_URL="https://drive.google.com/uc?id=1gfRcd1THN1KqlCB0Wi0i8X-DLjfcJulZ"
readonly RELATIONSHIPS_URL="https://drive.google.com/uc?id=1nM8d1TG3-ftKl4EgCs809aqAZUGNdO5O"
readonly NODES_FILE="/cypher/nodes.csv"
readonly RELATIONSHIPS_FILE="/cypher/relationships.csv"

# Download nodes and relationships from Google Drive
gdown "$NODES_URL" -O "$NODES_FILE"
gdown "$RELATIONSHIPS_URL" -O "$RELATIONSHIPS_FILE"

# Import nodes and relationships into the Cypher database
python3 "/usr/local/bin/import_nodes.py"
python3 "/usr/local/bin/import_relationships.py"