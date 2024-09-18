# Use the official Neo4j image as the base image
FROM neo4j:5.3

# Install Python3 and pip3 to use gdown
RUN apt-get update && apt-get install -y python3 python3-pip wget && \
    pip3 install gdown

# Download the Cypher file from Google Drive
RUN gdown "https://drive.google.com/uc?id=1ElYAtgyC5IJNZXXfgGvCLSXi9hcmInHo" -O /cypher/neo4j.cypher

# Copy the Bash scripts to the container
COPY split_cypher.sh /usr/local/bin/split_cypher.sh
COPY import_cypher_chunks.sh /usr/local/bin/import_cypher_chunks.sh

# Make the scripts executable
RUN chmod +x /usr/local/bin/split_cypher.sh
RUN chmod +x /usr/local/bin/import_cypher_chunks.sh

# Set the default password for Neo4j and the default database
ENV NEO4J_AUTH=neo4j/cyp450kg
ENV NEO4J_dbms_default__database=neo4j

# Set memory configuration (Optional)
ENV NEO4J_dbms_memory_heap_initial__size=8G
ENV NEO4J_dbms_memory_heap_max__size=8G
ENV NEO4J_dbms_memory_pagecache_size=6G

# Expose ports for Neo4j (7474 for HTTP and 7687 for Bolt)
EXPOSE 7474 7687

# Split the large Cypher file into chunks and import them into Neo4j
CMD /usr/local/bin/split_cypher.sh /cypher/neo4j.cypher /cypher/Cypher_Chunks 10000 && /usr/local/bin/import_cypher_chunks.sh
