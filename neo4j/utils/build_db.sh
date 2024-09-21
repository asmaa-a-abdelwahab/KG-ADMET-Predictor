#!/bin/bash

# Downloads nodes and relationships from Google Drive and imports them into
# the Neo4j database using Cypher.
#
# The `gdown` command is used to download the nodes and relationships CSV
# files from Google Drive. The `import_nodes.py` and `import_relationships.py`
# scripts are then used to import the nodes and relationships into Neo4j.

readonly NODES_URL="https://drive.google.com/uc?id=1gfRcd1THN1KqlCB0Wi0i8X-DLjfcJulZ"
readonly RELATIONSHIPS_URL="https://drive.google.com/uc?id=1nM8d1TG3-ftKl4EgCs809aqAZUGNdO5O"
readonly NODES_FILE="/cypher/nodes.csv"
readonly RELATIONSHIPS_FILE="/cypher/relationships.csv"

# Check if Neo4j is running
bash /usr/local/bin/start_neo4j.sh

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "Error: gdown is not installed. Please install it using 'pip install gdown'."
    exit 1
fi

# Download nodes and relationships from Google Drive
echo "Downloading nodes from Google Drive..."
gdown "$NODES_URL" -O "$NODES_FILE"
check_command "Downloading nodes"

echo "Downloading relationships from Google Drive..."
gdown "$RELATIONSHIPS_URL" -O "$RELATIONSHIPS_FILE"
check_command "Downloading relationships"

# Check if the CSV files exist
if [ ! -f "$NODES_FILE" ]; then
    echo "Error: Nodes file not found at $NODES_FILE."
    exit 1
fi

if [ ! -f "$RELATIONSHIPS_FILE" ]; then
    echo "Error: Relationships file not found at $RELATIONSHIPS_FILE."
    exit 1
fi

# Import nodes and relationships into the Neo4j database
echo "Importing nodes into the Neo4j database..."
python3 "/usr/local/bin/import_nodes.py"
check_command "Importing nodes"

echo "Importing relationships into the Neo4j database..."
python3 "/usr/local/bin/import_relationships.py"
check_command "Importing relationships"

echo "Import completed successfully."