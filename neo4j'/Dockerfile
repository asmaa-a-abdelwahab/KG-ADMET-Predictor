# Use the official Neo4j image as the base image
FROM neo4j:5.3

# Install Python3, pip3, wget, and curl to use gdown and to check Neo4j readiness
RUN apt-get update && apt-get install -y python3 python3-pip wget curl && \
    pip3 install gdown

# Create a directory for Cypher files and set the ownership
RUN mkdir -p /cypher && chown -R neo4j:neo4j /cypher

# Download the Cypher file from Google Drive
RUN gdown "https://drive.google.com/uc?id=1V4Cn9dX0VzDavQmTMrh93u9hXD-8NftR" -O /cypher/neo4j.cypher

# Copy the Bash scripts to the container
COPY utils/split_cypher.sh /usr/local/bin/split_cypher.sh
COPY utils/import_cypher_chunks.sh /usr/local/bin/import_cypher_chunks.sh
COPY utils/wait_for_neo4j.sh /usr/local/bin/wait_for_neo4j.sh

# Make the scripts executable as root (before switching to non-root)
RUN chmod +x /usr/local/bin/split_cypher.sh
RUN chmod +x /usr/local/bin/import_cypher_chunks.sh

# Switch to the non-root user (neo4j)
# USER neo4j

# Set memory configuration (Optional)
ENV NEO4J_dbms_memory_heap_initial__size=8G
ENV NEO4J_dbms_memory_heap_max__size=8G
ENV NEO4J_dbms_memory_pagecache_size=6G

# Expose ports for Neo4j (7474 for HTTP and 7687 for Bolt)
EXPOSE 7474 7687

# Start Neo4j, wait for it to be ready, and then run the import scripts
CMD ["/bin/bash", "-c", "neo4j start && /usr/local/bin/wait_for_neo4j.sh && /usr/local/bin/split_cypher.sh /cypher/neo4j.cypher /cypher/Cypher_Chunks 10000 && /usr/local/bin/import_cypher_chunks.sh"]

# to build the image
# docker build -t neo4j_docker '.\neo4j''\'

# to run it (with environment variable for NEO4J_AUTH)
# docker run -it -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/cyp450kg neo4j_docker
