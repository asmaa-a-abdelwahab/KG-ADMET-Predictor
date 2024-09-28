import csv
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths to CSV files
nodes_file = "/cypher/nodes.csv"  # Adjust the path if necessary
relationships_file = "/cypher/relationships.csv"
batch_size = 5000  # Adjust based on performance testing
max_workers = 15  # Number of parallel workers for executing batches

# Dictionary to store node properties based on node ID
node_mapping = {}


def clean_json_string(json_str):
    """Cleans JSON strings to fix common formatting issues."""
    json_str = json_str.replace('""', '"')
    json_str = re.sub(r"\\n|\\r", "", json_str)
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)  # Remove trailing commas
    json_str = re.sub(r'(?<=\w)"(?=\w)', "'", json_str)  # Replace misplaced quotes
    return json_str


def parse_json(json_str):
    """Parses JSON string with error handling for problematic structures."""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}. Attempting to fix...")
        json_str = clean_json_string(json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON after cleaning: {e}")
            return None


def sanitize_key(key):
    """
    Sanitizes a property key to make it compatible with Neo4j Cypher.
    Replaces '%' with 'percentage' and removes other invalid characters.

    :param key: The original property key.
    :return: The sanitized property key.
    """
    # Replace '%' with 'percentage'
    key = key.replace("%", "percentage_")
    # Remove any other characters that are not allowed in identifiers
    key = re.sub(r"[^a-zA-Z0-9_]", "_", key)
    return key


def escape_quotes(value):
    """
    Escapes double quotes in a given value to make it Cypher-compatible.
    """
    if isinstance(value, str):
        return value.replace('"', '\\"').replace("'", "\\'")  # Escape double quotes
    return value


def import_nodes(nodes_file, batch_size):
    """
    Imports nodes from a CSV file into Neo4j using Cypher queries in batches.
    Also stores node properties in `node_mapping` for later use in relationships.
    """
    with open(nodes_file, "r") as file:
        reader = csv.DictReader(file)
        batch = []
        count = 0

        # Create ThreadPoolExecutor for parallel execution of batches
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            # Iterate over each row in the CSV file
            for row in reader:
                count += 1
                try:
                    json_str = clean_json_string(row["n"])
                    node_data = parse_json(json_str)
                    if node_data is None:
                        continue  # Skip rows that still fail parsing
                except Exception as e:
                    print(f"Error parsing JSON at row {count}: {e}")
                    continue  # Skip problematic rows

                # Extract labels and properties from the parsed JSON
                labels = ":".join(node_data["labels"])
                properties = node_data["properties"]

                # Store node properties in the mapping dictionary with integer node_id
                node_id = int(node_data["id"])  # Ensure node_id is an integer
                node_mapping[node_id] = {
                    "label": labels,
                    "properties": properties,
                }

                # Create Cypher query for node creation
                properties_string = ", ".join(
                    [
                        f"{key}: {json.dumps(escape_quotes(value))}"
                        for key, value in properties.items()
                    ]
                )
                cypher_query = f"CREATE (n:{labels} {{{properties_string}}});"
                batch.append(cypher_query)
                # print(cypher_query)

                # Execute batch when batch size is met
                if len(batch) == batch_size:
                    futures.append(executor.submit(execute_batch, batch))
                    batch = []

            # Execute any remaining queries
            if batch:
                futures.append(executor.submit(execute_batch, batch))

            # Ensure all batches are completed
            for future in futures:
                future.result()


def get_node_match_clause(node_id):
    """
    Generates a MATCH clause for the given node based on its ID using node_mapping.
    """
    node_id = int(node_id)  # Convert node_id to integer to match node_mapping keys
    if node_id not in node_mapping:
        raise ValueError(f"Node ID {node_id} not found in node_mapping.")

    node_data = node_mapping[node_id]
    label = node_data["label"]
    properties = node_data["properties"]

    # Escape quotes in the properties that will be used in the MATCH clause
    if label == "Compound":
        compound_name = escape_quotes(properties["CompoundName"])
        return f'MATCH (a:Compound {{CompoundName: "{compound_name}"}})'
    elif label == "Gene":
        gene_symbol = escape_quotes(properties["GeneSymbol"])
        return f'MATCH (a:Gene {{GeneSymbol: "{gene_symbol}"}})'
    elif label == "BioAssay":
        assay_name = escape_quotes(properties["AssayName"])
        return f'MATCH (a:BioAssay {{AssayName: "{assay_name}"}})'
    elif label == "Protein":
        protein_acc = escape_quotes(properties["ProteinRefSeqAccession"])
        return f'MATCH (a:Protein {{ProteinRefSeqAccession: "{protein_acc}"}})'
    else:
        raise ValueError(f"Unknown node label: {label}")


def import_relationships(relationships_file, batch_size):
    """
    Reads a CSV file with relationship data and imports it into Neo4j using
    Cypher queries in batches.
    """
    with open(relationships_file, "r") as file:
        reader = csv.DictReader(file)
        batch = []
        count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for row in reader:
                count += 1

                try:
                    relationship_data = json.loads(row["r"])
                    rel_type = relationship_data["type"]
                    start_node_id = int(
                        relationship_data["start"]
                    )  # Ensure IDs are integers
                    end_node_id = int(
                        relationship_data["end"]
                    )  # Ensure IDs are integers
                    properties = relationship_data["properties"]

                    # Get the MATCH clauses for the start and end nodes
                    start_match = get_node_match_clause(start_node_id)
                    end_match = get_node_match_clause(end_node_id).replace("a:", "b:")

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error processing row {count}: {e}")
                    continue  # Skip problematic rows

                # Sanitize property keys to ensure they are valid
                sanitized_properties = {
                    sanitize_key(k): v for k, v in properties.items()
                }

                # Create a Cypher query for relationship creation
                properties_string = ", ".join(
                    [
                        f"{key}: {json.dumps(value)}"
                        for key, value in sanitized_properties.items()
                    ]
                )
                cypher_query = f"""
                {start_match}
                {end_match}
                CREATE (a)-[r:{rel_type} {{{properties_string}}}]->(b);
                """
                # print(cypher_query)
                batch.append(cypher_query)

                if len(batch) == batch_size:
                    futures.append(executor.submit(execute_batch, batch))
                    batch = []

            if batch:
                futures.append(executor.submit(execute_batch, batch))

            for future in futures:
                future.result()


def execute_batch(batch):
    """
    Execute a batch of Cypher queries using the cypher-shell command.
    """
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".cypher"
    ) as temp_file:
        temp_file.write("\n".join(batch))
        temp_file_path = temp_file.name

    command = f"cypher-shell -u neo4j -p cyp450kg --format plain < {temp_file_path}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    os.remove(temp_file_path)

    if result.returncode != 0:
        print(f"Error executing batch: {result.stderr}")
    else:
        print("Batch executed successfully")


# Run the node and relationship import functions
import_nodes(nodes_file, batch_size)
import_relationships(relationships_file, batch_size)
