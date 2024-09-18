#!/bin/bash

# Cypher chunk directory
chunk_dir="/cypher/Cypher_Chunks"

# Neo4j credentials (you can also retrieve these from environment variables if needed)
user="neo4j"
password="cyp450kg"

# Wait for Neo4j to start up (you can adjust the waiting time if needed)
sleep 15

# Loop through all Cypher files in the chunk directory and run cypher-shell on each
for file in "$chunk_dir"/*.cypher; do
    echo "Importing $file..."
    cypher-shell -u $user -p $password -f "$file"
done

echo "All Cypher files have been imported successfully."
