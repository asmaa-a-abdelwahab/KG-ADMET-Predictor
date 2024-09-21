import csv
import json
import subprocess
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Path to the relationships CSV file
relationships_file = (
    "/cypher/relationships.csv"  # Adjust based on where you copy the file
)
batch_size = 1000  # Adjust based on performance testing
max_workers = 4  # Number of parallel workers for executing batches


def import_relationships(relationships_file, batch_size):
    """
    Reads a CSV file with relationship data and imports it into the Neo4j database
    using Cypher queries in batches.

    :param relationships_file: The path to the CSV file of relationships to import.
    :param batch_size: The number of relationships to include in each batch.
    """
    with open(relationships_file, "r") as file:
        reader = csv.DictReader(file)
        batch = []
        count = 0

        # Create ThreadPoolExecutor for parallel execution of batches
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []  # To store future tasks

            for row in reader:
                # Increment row count
                count += 1

                # Read the relationship data from the CSV row
                try:
                    relationship_data = json.loads(row["r"])
                    rel_type = relationship_data["type"]
                    start_node = relationship_data["start"]
                    end_node = relationship_data["end"]
                    properties = relationship_data["properties"]
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON at row {count}: {e}")
                    continue  # Skip problematic rows

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

                # Execute the batch every 'batch_size' relationships
                if count % batch_size == 0:
                    # Submit batch execution to the executor
                    futures.append(executor.submit(execute_batch, batch))
                    batch = []

            # Execute any remaining queries in the batch
            if batch:
                futures.append(executor.submit(execute_batch, batch))

            # Wait for all futures to complete and check results
            for future in as_completed(futures):
                future.result()  # This will raise any exceptions caught during execution


def execute_batch(batch):
    """
    Execute a batch of Cypher queries using the cypher-shell command.

    :param batch: A list of Cypher queries to execute.
    """
    # Write the batch of queries to a temporary file
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".cypher"
    ) as temp_file:
        temp_file.write("\n".join(batch))
        temp_file_path = temp_file.name

    # Construct the command to run cypher-shell
    command = (
        f"cypher-shell -u neo4j -p your_password --format plain < {temp_file_path}"
    )

    # Run the command using subprocess
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Remove the temporary file after execution
    os.remove(temp_file_path)

    # Check for errors in the command execution
    if result.returncode != 0:
        print(f"Error executing batch: {result.stderr}")
    else:
        print("Batch executed successfully")


# Run the relationships import function
import_relationships(relationships_file, batch_size)
