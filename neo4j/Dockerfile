# Use the official Neo4j image as the base image
FROM neo4j:latest

# Install Python3, pip3, wget, curl, and nano to use gdown and to check Neo4j readiness
RUN apt-get update && apt-get install -y python3 python3-pip wget curl nano && \
    pip3 install gdown docker

# Create a directory for Cypher files and set the ownership
RUN mkdir -p /cypher && chown -R neo4j:neo4j /cypher

# Copy the custom Neo4j configuration file to the appropriate location
COPY conf/neo4j.conf /var/lib/neo4j/conf/neo4j.conf

# Copy the Bash scripts and Python scripts to the container
COPY utils/start_neo4j.sh /usr/local/bin/start_neo4j.sh
COPY utils/import_nodes.py /usr/local/bin/import_nodes.py
COPY utils/import_relationships.py /usr/local/bin/import_relationships.py

# Make the scripts executable
RUN chmod +x /usr/local/bin/start_neo4j.sh /usr/local/bin/import_nodes.py /usr/local/bin/import_relationships.py

# Expose ports for Neo4j (7474 for HTTP and 7687 for Bolt)
EXPOSE 7474 7687

# Set up the entrypoint script to run the initialization
ENTRYPOINT ["/usr/local/bin/start_neo4j.sh"]

# Start Neo4j in console mode to keep the service running in the foreground
CMD ["/bin/bash", "-c", "neo4j console"]



# to build the image
# docker build -t neo4j_docker '.\neo4j'

# to run it (with environment variable for NEO4J_AUTH)
# docker run -it -p 7474:7474 -p 7687:7687 neo4j_docker

# To run it in interactive mode
# docker run -it -p 7474:7474 -p 7687:7687 -it neo4j_docker bash
# docker exec -it 56161cbc6cc1 bash

