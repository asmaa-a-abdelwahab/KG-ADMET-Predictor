# Use the official Neo4j image as the base image
FROM neo4j:5.3

# Install Python3 and pip3 to use gdown
RUN apt-get update && apt-get install -y python3 python3-pip wget && \
    pip3 install gdown

# Create a directory for Cypher files and set the ownership
RUN mkdir -p /cypher && chown -R neo4j:neo4j /cypher

# Download the Cypher file from Google Drive
RUN gdown "https://drive.google.com/uc?id=1ElYAtgyC5IJNZXXfgGvCLSXi9hcmInHo" -O /cypher/neo4j.cypher

# Copy the Bash scripts to the container
COPY utils/split_cypher.sh /usr/local/bin/split_cypher.sh
COPY utils/import_cypher_chunks.sh /usr/local/bin/import_cypher_chunks.sh

# Make the scripts executable as root (before switching to non-root)
RUN chmod +x /usr/local/bin/split_cypher.sh
RUN chmod +x /usr/local/bin/import_cypher_chunks.sh

# Switch to the non-root user (neo4j)
USER neo4j

# Set memory configuration (Optional)
ENV NEO4J_dbms_memory_heap_initial__size=8G
ENV NEO4J_dbms_memory_heap_max__size=8G
ENV NEO4J_dbms_memory_pagecache_size=6G

# Expose ports for Neo4j (7474 for HTTP and 7687 for Bolt)
EXPOSE 7474 7687

# Use the neo4j user to split and import the Cypher file
CMD ["/bin/bash", "-c", "/usr/local/bin/split_cypher.sh /cypher/neo4j.cypher /cypher/Cypher_Chunks 10000 && /usr/local/bin/import_cypher_chunks.sh"]

# to build the image
# docker build -t neo4j_docker '.\neo4j''\'

# to run it (with environment variable for NEO4J_AUTH)
# docker run -it -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/cyp450kg neo4j_docker
