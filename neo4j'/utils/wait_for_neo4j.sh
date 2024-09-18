#!/bin/bash

# Wait for Neo4j to start by checking the HTTP port (7474)
until curl --silent --fail http://localhost:7474; do
    echo "Waiting for Neo4j to start..."
    sleep 5
done

echo "Neo4j is up and running."
