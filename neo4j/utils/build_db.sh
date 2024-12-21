#!/bin/bash

# Downloads nodes and relationships from Google Drive and imports them into Neo4j
#
# This script is used to populate the Neo4j database with the data from the CSV
# files available at the given URLs. The script downloads the CSV files, checks
# if they exist, and then imports them into Neo4j using the import_graph_data.py script.

user="neo4j"
old_password="neo4j"
new_password="cyp450kg"

# Wait for Neo4j to start by checking the HTTP port (7474)
until curl --silent --fail http://localhost:7474; do
    echo "Waiting for Neo4j to start..."
    sleep 5
done

echo "Neo4j is up and running."

# Change the default password
echo "Changing the default password..."
echo "ALTER CURRENT USER SET PASSWORD FROM '$old_password' TO '$new_password';" | cypher-shell -u "$user" -p "$old_password" -d system
if [ $? -ne 0 ]; then
    echo "Failed to change the password. Check permissions or Neo4j status."
    exit 1
fi

# Update the password variable to use the new password
password="$new_password"
echo "Password changed successfully."

# Confirm the default database is accessible
echo "Testing access to the default database 'neo4j'."
echo "MATCH (n) RETURN n LIMIT 1;" | cypher-shell -u "$user" -p "$password"
if [ $? -ne 0 ]; then
    echo "Failed to access the default database. Check permissions or Neo4j status."
    exit 1
fi

echo "Successfully accessed the default database 'neo4j' for project use."


# URLs can be passed via environment variables (set default values if not provided)
NodesUrl="${NODES_URL:-https://drive.google.com/uc?id=1gfRcd1THN1KqlCB0Wi0i8X-DLjfcJulZ}"
RelationshipsUrl="${RELATIONSHIPS_URL:-https://drive.google.com/uc?id=1nM8d1TG3-ftKl4EgCs809aqAZUGNdO5O}"
NodesFile="/cypher/nodes.csv"
RelationshipsFile="/cypher/relationships.csv"

# Ensure gdown is installed
if ! command -v gdown >/dev/null 2>&1; then
    echo "Error: gdown is not installed. Please install it using 'pip install gdown'." 1>&2
    exit 1
fi

# Download nodes and relationships from Google Drive
echo "Downloading nodes and relationships from Google Drive..." 1>&2

# Check if the CSV files exist
if [ ! -f "$NodesFile" ]; then
    gdown "$NodesUrl" -O "$NodesFile"
    echo "Finished downloading nodes." 1>&2
fi

if [ ! -f "$RelationshipsFile" ]; then
    gdown "$RelationshipsUrl" -O "$RelationshipsFile"
    echo "Finished downloading relationships." 1>&2
fi

# Import nodes and relationships into Neo4j
log_file="/cypher/import.log"

echo "Importing data into the Neo4j database..." 1>&2
python3 "/usr/local/bin/import_graph_data.py" >> "$log_file"
echo "Finished importing data." 1>&2

