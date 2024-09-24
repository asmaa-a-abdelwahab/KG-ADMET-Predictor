#!/bin/bash

# Start Neo4j in the background temporarily
# neo4j start

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

# # Stop the temporary Neo4j instance
# echo "Stopping Neo4j to restart in console mode..."
# neo4j stop

# # Ensure Neo4j stops completely before starting it again
# sleep 5

# # Start Neo4j in the foreground
# echo "Starting Neo4j in console mode..."
# exec neo4j console
