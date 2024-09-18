# Use the official Neo4j image as the base image
FROM neo4j:5.3

# Install Python3 and pip3 to use gdown
RUN apt-get update && apt-get install -y python3 python3-pip wget && \
    pip3 install gdown

# Download the Cypher file from Google Drive
RUN gdown "https://drive.google.com/uc?id=1ElYAtgyC5IJNZXXfgGvCLSXi9hcmInHo" -O /cypher/neo4j.cypher

# Copy the custom neo4j.conf file to configure Neo4j memory settings
COPY neo4j.conf /var/lib/neo4j/conf/neo4j.conf

# Copy the Bash scripts to the container
COPY /utils/split_cypher.sh /usr/local/bin/split_cypher.sh
COPY /utils/import_cypher_chunks.sh /usr/local/bin/import_cypher_chunks.sh

# Make scripts executable
RUN chmod +x /usr/local/bin/split_cypher.sh
RUN chmod +x /usr/local/bin/import_cypher_chunks.sh

# Expose ports for Neo4j
EXPOSE 7474 7687

# Set the default password for Neo4j
ENV NEO4J_AUTH=neo4j/cyp450kg

# Split the large Cypher file into chunks and import them into Neo4j
CMD /usr/local/bin/split_cypher.sh /cypher/neo4j.cypher /cypher/Cypher_Chunks 10000 && /usr/local/bin/import_cypher_chunks.sh
