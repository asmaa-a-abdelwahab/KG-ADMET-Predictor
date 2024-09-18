#!/bin/bash

# Wait for Neo4j to start
/usr/local/bin/wait_for_neo4j.sh

# Neo4j credentials
user="neo4j"
password="cyp450kg"

# Create a new database using cypher-shell
echo "Creating a new database..."
echo "CREATE DATABASE my_new_database;" | cypher-shell -u "$user" -p "$password"

# Run the import script for Cypher files (if needed)
for file in "$chunk_dir"/*.cypher; do
    echo "Importing $file..."
    if cypher-shell -u "$user" -p "$password" -d my_new_database -f "$file"; then
        echo "$file imported successfully."
    else
        echo "Failed to import $file due to an authentication or connection error."
        exit 1
    fi
done

echo "Database and import process completed."