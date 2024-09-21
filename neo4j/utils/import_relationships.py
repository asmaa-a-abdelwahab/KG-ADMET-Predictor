import csv
import json
import subprocess

relationships_file = (
    "/cypher/relationships.csv"  # Adjust based on where you copy the file
)
batch_size = 1000


def import_relationships(relationships_file, batch_size):
    """
    Reads a CSV file with relationship data and imports it into the Neo4j database
    using Cypher queries in batches.

    :param relationships_file: The path to the CSV file of relationships to import.
    :param batch_size: The number of relationships to include in each batch. This
        can be adjusted to control memory usage and performance.
    """
    with open(relationships_file, "r") as file:
        reader = csv.DictReader(file)
        batch = []
        for i, row in enumerate(reader, 1):
            # Read the relationship data from the CSV row
            relationship_data = json.loads(row["r"])
            rel_type = relationship_data["type"]
            start_node = relationship_data["start"]
            end_node = relationship_data["end"]
            properties = relationship_data["properties"]

            # Create a Cypher query to create the relationship
            properties_string = ", ".join(
                [f"{key}: {json.dumps(value)}" for key, value in properties.items()]
            )
            cypher_query = f"""
            MATCH (a), (b)
            WHERE id(a) = {start_node} AND id(b) = {end_node}
            CREATE (a)-[r:{rel_type} {{{properties_string}}}]->(b);
            """
            batch.append(cypher_query)

            # Execute the batch every 'batch_size' relationships or when we reach
            # the end of the file
            if i % batch_size == 0 or i == sum(1 for row in reader):
                execute_batch(batch)
                batch = []


def execute_batch(batch):
    """
    Execute a batch of Cypher queries using the cypher-shell command.

    :param batch: A list of Cypher queries to execute.
    """
    # Create a Cypher script with periodic commit
    cypher_script = "USING PERIODIC COMMIT " + str(len(batch)) + "\n" + "\n".join(batch)
    # Construct the command to run cypher-shell
    command = f'echo "{cypher_script}" | cypher-shell -u neo4j -p your_password'
    # Run the command using subprocess
    subprocess.run(command, shell=True)


import_relationships(relationships_file, batch_size)
