#!/bin/bash

# Wait for Neo4j to start
/usr/local/bin/wait_for_neo4j.sh

# Neo4j credentials
user="neo4j"
old_password="neo4j"
new_password="cyp450kg"

# Change the default password if necessary
echo "Changing the default password..."
echo "ALTER CURRENT USER SET PASSWORD FROM '$old_password' TO '$new_password';" | cypher-shell -u "$user" -p "$old_password" -d system
if [ $? -ne 0 ]; then
    echo "Failed to change the password. Check permissions or Neo4j status."
    exit 1
fi
echo "Password changed."

# Update the credentials in the script
password="$new_password"

# Use the default 'neo4j' database (no CREATE DATABASE)
echo "Using the default 'neo4j' database..."

# Wait for Neo4j to be ready
sleep 5

# Run the import script for Cypher files
chunk_dir="/cypher/Cypher_Chunks"
for file in "$chunk_dir"/*.cypher; do
    echo "Importing $file..."
    if cat $file | cypher-shell -u "$user" -p "$password" -d neo4j; then
        echo "$file imported successfully."
    else
        echo "Failed to import $file due to an authentication or connection error."
        exit 1
    fi
done

echo "Database and import process completed."
