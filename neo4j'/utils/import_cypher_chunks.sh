#!/bin/bash

# Directory where Cypher chunk files are stored
chunk_dir="/cypher/Cypher_Chunks"

# Neo4j credentials
user="neo4j"
password="cyp450kg"

# Wait for Neo4j to start
sleep 15

# Loop through all Cypher files in the chunk directory and import them using cypher-shell
for file in "$chunk_dir"/*.cypher; do
    echo "Importing $file..."
    cypher-shell -u $user -p $password -f "$file"
done

echo "All Cypher files have been imported successfully."
