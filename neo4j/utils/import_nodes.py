import csv
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

# Update path to CSV file inside the Docker container
nodes_file = "/cypher/nodes.csv"  # Ensure the path is correct
batch_size = 5000  # Adjust batch size based on performance testing
max_workers = 10  # Number of parallel workers for executing batches


def clean_json_string(json_str):
    """
    Cleans the JSON string by fixing common formatting issues. This is a
    pre-processing step before parsing the JSON string to ensure it is
    well-formed and can be successfully parsed by the json module.

    The function fixes the following common formatting issues:

        1. Replaces double-double quotes with single-double quotes
        2. Removes unescaped newlines or carriage returns
        3. Corrects misplaced commas or brackets if needed
        4. Attempts to fix misplaced double quotes or other formatting issues
    """
    # Replace double-double quotes with single-double quotes
    json_str = json_str.replace('""', '"')
    # Remove unescaped newlines or carriage returns
    json_str = re.sub(r"\\n|\\r", "", json_str)
    # Correct misplaced commas or brackets if needed
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)  # Remove trailing commas
    # Attempt to fix misplaced double quotes or other formatting issues
    json_str = re.sub(r'(?<=\w)"(?=\w)', "'", json_str)  # Replace misplaced quotes
    return json_str


def parse_json(json_str):
    """
    Parses the JSON string with error handling to manage problematic structures.

    This function will attempt to parse the provided JSON string and handle any
    decoding errors that may arise. If the decoding error is due to a minor issue
    such as misplaced commas or brackets, the function will attempt to clean the
    string and retry the parsing. If all attempts fail, the function will return
    None.
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Attempting to fix JSON decoding error: {e}")
        # Attempt minor fixes and retry
        try:
            # Try to fix specific errors by making adjustments to the string
            json_str = clean_json_string(json_str)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # Log parsing failure
            print(f"Failed to parse JSON after cleaning: {e}")
            return None


def wait_for_neo4j():
    """
    Waits for Neo4j to be fully up and running.

    This function will wait up to 30 attempts (150 seconds) for Neo4j to be
    ready by attempting to execute a simple Cypher query. If the query is
    successful, the function will return True. If the maximum number of attempts
    is reached without success, the function will return False.
    """
    print("Waiting for Neo4j to be ready...")
    for _ in range(30):  # Wait up to 30 attempts (150 seconds)
        try:
            # Attempt to execute a simple Cypher query
            result = subprocess.run(
                ["cypher-shell", "-u", "neo4j", "-p", "cyp450kg", "RETURN 1"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("Neo4j is ready.")
                return True
        except Exception as e:
            # Log the error and continue waiting
            print(f"Waiting for Neo4j... {e}")
        # Wait 5 seconds before attempting again
        time.sleep(5)
    print("Failed to connect to Neo4j after waiting.")
    return False


def import_nodes(nodes_file, batch_size):
    """
    Imports nodes from a CSV file into Neo4j using Cypher queries in batches.

    This function will wait for Neo4j to be ready before attempting to import
    nodes. If Neo4j is not ready after waiting, the function will exit.

    :param nodes_file: The path to the CSV file containing the nodes to import.
    :param batch_size: The number of nodes to include in each batch. This can be
        adjusted to control memory usage and performance.
    """
    if not wait_for_neo4j():
        print("Neo4j is not ready. Exiting.")
        return

    with open(nodes_file, "r") as file:
        reader = csv.DictReader(file)
        batch = []
        count = 0  # Keep track of total rows processed

        # Create ThreadPoolExecutor for parallel execution of batches
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []  # To store future tasks

            for row in reader:
                count += 1
                try:
                    # Clean and parse the JSON-like string
                    json_str = clean_json_string(row["n"])
                    node_data = parse_json(
                        json_str
                    )  # Use parse_json with improved error handling
                    if node_data is None:
                        continue  # Skip rows that still fail parsing
                except Exception as e:
                    print(f"Error parsing JSON at row {count}: {e}")
                    continue  # Skip the problematic row

                labels = ":".join(node_data["labels"])
                properties = node_data["properties"]
                properties_string = ", ".join(
                    [f"{key}: {json.dumps(value)}" for key, value in properties.items()]
                )
                cypher_query = f"CREATE (n:{labels} {{{properties_string}}});"
                batch.append(cypher_query)

                # Execute batch when the batch size is met
                if len(batch) == batch_size:
                    # Submit batch execution to the executor
                    futures.append(executor.submit(execute_batch, batch))
                    batch = []

            # Execute any remaining queries
            if batch:
                futures.append(executor.submit(execute_batch, batch))

            # Ensure all futures are completed
            for future in futures:
                future.result()


def execute_batch(batch):
    """
    Execute a batch of Cypher queries using the cypher-shell command.

    :param batch: A list of Cypher queries to execute.
    """
    # Create a temporary file to write the batch of queries
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".cypher"
    ) as temp_file:
        # Write the queries to the temporary file
        temp_file.write("\n".join(batch))
        temp_file_path = temp_file.name

    # Execute the queries using cypher-shell with the temporary file
    command = f"cypher-shell -u neo4j -p cyp450kg --format plain < {temp_file_path}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Remove the temporary file after execution
    os.remove(temp_file_path)

    # Check the result of the command
    if result.returncode != 0:
        # Print error message if the command failed
        print(f"Error executing batch: {result.stderr}")
        # Print additional information if the error is due to unauthorized access
        if "unauthorized" in result.stderr.lower():
            print("Check your Neo4j username and password.")
    else:
        # Print success message if the command executed successfully
        print("Batch executed successfully")


# Run the optimized node import function
import_nodes(nodes_file, batch_size)
