# Use the official Neo4j image
FROM neo4j:latest

# Install Python3, pip3, wget, curl, and nano
RUN apt-get update && apt-get install -y python3 python3-pip wget curl nano && \
    pip3 install gdown

# Create directories and set ownership for Neo4j data and custom scripts
RUN mkdir -p /cypher && chown -R neo4j:neo4j /cypher

# Copy the Neo4j configuration file
COPY conf/neo4j.conf /var/lib/neo4j/conf/neo4j.conf
RUN chown neo4j:neo4j /var/lib/neo4j/conf/neo4j.conf

# Copy the Bash and Python scripts
COPY utils/start_neo4j.sh /usr/local/bin/start_neo4j.sh
COPY utils/build_db.sh /usr/local/bin/build_db.sh
COPY utils/import_graph_data.py /usr/local/bin/import_graph_data.py
RUN chmod +x /usr/local/bin/*.sh

# Expose ports for Neo4j
EXPOSE 7474 7687

# Use volumes to persist data
VOLUME /data
VOLUME /logs

# Default command to start Neo4j
CMD ["sh", "-c", "neo4j console"]






# # Use the official Neo4j image as the base image
# FROM neo4j:latest

# # Install Python3, pip3, wget, curl, and nano to use gdown and to check Neo4j readiness
# RUN apt-get update && apt-get install -y python3 python3-pip wget curl nano && \
#     pip3 install gdown

# # Create a directory for Cypher files and set the ownership
# RUN mkdir -p /cypher && chown -R neo4j:neo4j /cypher

# # Copy the custom Neo4j configuration file to the appropriate location => why not copied correctly in the container?
# COPY conf/neo4j.conf /var/lib/neo4j/conf/neo4j.conf 

# # Copy the Bash scripts and Python scripts to the container
# COPY utils/start_neo4j.sh /usr/local/bin/start_neo4j.sh
# COPY utils/build_db.sh /usr/local/bin/build_db.sh
# COPY utils/import_graph_data.py /usr/local/bin/import_graph_data.py

# # Make the scripts executable
# RUN chmod +x /usr/local/bin/build_db.sh 

# # Expose ports for Neo4j (7474 for HTTP and 7687 for Bolt)
# EXPOSE 7474 7687

# # Start Neo4j in console mode to keep the service running in the foreground
# CMD ["sh", "-c", "neo4j console"]


# CMD ["sh", "-c", "neo4j start && bash /usr/local/bin/build_db.sh"]

# RUN bash /usr/local/bin/build_db.sh &


# CMD ["sh", "/usr/local/bin/build_db.sh"]

# Set up the entrypoint script to run the initialization
# ENTRYPOINT ["sh", "/usr/local/bin/build_db.sh"]

# to build the image
# docker build -t neo4j_docker '.\neo4j'

# to run it (with environment variable for NEO4J_AUTH)
# docker run -it -p 7474:7474 -p 7687:7687 neo4j_docker

# To run it in interactive mode
# docker run -it -p 7474:7474 -p 7687:7687 -it neo4j_docker bash
# docker exec -it 56161cbc6cc1 bash