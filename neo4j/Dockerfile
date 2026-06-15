FROM neo4j:5.26-community

# Install Neo4j Graph Data Science (GDS) at container startup via the official
# Neo4j Docker plugin mechanism. This is required for CALL gds.* procedures
# used by Stage 1 GDS/FastRP/GraphSAGE feature export.
# If docker-compose supplies NEO4J_PLUGINS, it will override this value; make
# sure it still includes "graph-data-science".
ENV NEO4J_PLUGINS='["graph-data-science"]'


# The Neo4j service only runs the database. PRING loading is done by the
# separate pring-loader service so the database image stays clean and stable.
#
# IMPORTANT: neo4j.conf is baked into the image instead of bind-mounted as a
# single file. The official Neo4j entrypoint updates neo4j.conf when plugins
# such as APOC are enabled; a bind-mounted file can fail with:
#   sed: cannot rename /var/lib/neo4j/conf/sedXXXX: Device or resource busy
USER root
COPY conf/neo4j.conf /var/lib/neo4j/conf/neo4j.conf
RUN chown -R neo4j:neo4j /var/lib/neo4j/conf /var/lib/neo4j/plugins
USER neo4j

EXPOSE 7474 7687
CMD ["neo4j", "console"]
