from utils.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    logger,
)
from utils.neo4j_utils import (
    Neo4jBase,
)
from neo4j import Driver

import pandas as pd
import json

logger.info("Retrieving data from Neo4j")


def retrieve_and_process_data(driver: Driver) -> None:
    """
    Retrieve and process data from the Neo4j database, then save the processed file.
    Extract all details for both Compound and Gene and save them as a flat CSV file.
    """
    export_query = """
    CALL apoc.export.csv.query(
        'MATCH (c:Compound)-[r]->(g:Gene)
        WHERE g.GeneSymbol IN ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP3A4", "CYP2E1", "CYP3A5"]
        RETURN c AS Compound, g AS Gene, type(r) AS InteractionType, properties(r) AS InteractionProperties',
        "compound_gene_interactions.csv",
        {batchSize: 10000, delimiter: ","}
    ) YIELD file, rows, done;
    """

    try:
        with driver.session() as session:
            # Execute export query and capture the file path
            logger.info("Starting data export to CSV...")
            result = session.run(export_query)
            for record in result:
                csv_path = record["file"]
                rows_exported = record["rows"]
                logger.info(
                    f"Data export completed successfully. CSV file path: {csv_path}"
                )
                logger.info(f"Rows exported: {rows_exported}")
                print(f"CSV file path: {csv_path}")

                # Load the exported CSV file
                logger.info("Loading exported data...")
                df = pd.read_csv("/import/compound_gene_interactions.csv")

                # Process the data
                logger.info("Processing data...")
                processed_data = []
                for _, row in df.iterrows():
                    # Parse the JSON-like structure in the Compound and Gene columns
                    compound = json.loads(row["Compound"])
                    gene = json.loads(row["Gene"])
                    interaction_type = row["InteractionType"]
                    interaction_properties = json.loads(row["InteractionProperties"])

                    # Extract all properties from Compound and Gene
                    compound_properties = compound.get("properties", {})
                    gene_properties = gene.get("properties", {})
                    processed_row = {
                        **{
                            f"Compound_{key}": value
                            for key, value in compound_properties.items()
                        },
                        **{
                            f"Gene_{key}": value
                            for key, value in gene_properties.items()
                        },
                        "InteractionType": interaction_type,
                        **{
                            f"Interaction_{key}": value
                            for key, value in interaction_properties.items()
                        },
                    }
                    processed_data.append(processed_row)

                # Create a DataFrame from the processed data
                processed_df = pd.DataFrame(processed_data)

                # Save the processed data to a new CSV file
                processed_file_path = "/import/processed_compound_gene_interactions.csv"
                logger.info(f"Saving processed data to {processed_file_path}...")
                processed_df.to_csv(processed_file_path, index=False)
                logger.info(
                    f"Processed data saved successfully to {processed_file_path}"
                )

    except Exception as e:
        logger.error(f"Error during data export or processing: {e}")
        print(f"Error during data export or processing: {e}")


neo4j_conn = Neo4jBase(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
neo4j_conn.connect_to_neo4j()

retrieve_and_process_data(neo4j_conn.driver)
