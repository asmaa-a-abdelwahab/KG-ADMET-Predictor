# neo4j_utils.py

from neo4j import GraphDatabase, Driver
from typing import List, Any
import logging
from utils.config import logger


class Neo4jConnectionError(Exception):
    pass


class Neo4jBase:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def connect_to_neo4j(self) -> None:
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            logger.info("Successfully connected to the Neo4j database.")
        except Exception as e:
            logger.error(f"Failed to connect to the Neo4j database: {e}")
            raise Neo4jConnectionError(
                "Failed to connect to the Neo4j database."
            ) from e

    def close(self) -> None:
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed successfully.")


def execute_query(driver: Driver, query: str) -> List[List[Any]]:
    try:
        with driver.session() as session:
            result = session.run(query)
            return result
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return []


def get_compound_names(driver: Driver) -> List[str]:
    """Retrieve a list of compound names from the Neo4j database."""
    query = "MATCH (c:Compound) RETURN c.CompoundName AS name"

    # Execute the query and ensure all results are consumed before processing
    with driver.session() as session:
        result = session.run(query)
        records = result.data()  # Fetch all records as a list of dictionaries

    # Return the names extracted from the result
    return [record["name"] for record in records]


def get_gene_symbols(driver: Driver) -> List[str]:
    """Retrieve a list of gene symbols from the Neo4j database."""
    query = "MATCH (g:Gene) RETURN g.GeneSymbol AS symbol"

    # Execute the query and ensure all results are consumed before processing
    with driver.session() as session:
        result = session.run(query)
        records = result.data()  # Fetch all records as a list of dictionaries

    # Return the symbols extracted from the result
    return [record["symbol"] for record in records]


def get_similar_compounds(driver: Driver, compound_names: List[str]) -> List[List[Any]]:
    """
    Retrieve similar compounds based on a list of compound names.
    :param driver: Neo4j driver object.
    :param compound_names: List of compound names to match.
    :return: List of results, each containing c1, r, and c2.
    """
    # Construct the Cypher query with parameters to avoid issues with string formatting
    query = """
    MATCH (c1:Compound)-[r:IS_SIMILAR_TO]->(c2:Compound)
    WHERE c1.CompoundName IN $compound_names
    AND c2.CompoundName IS NOT NULL AND c2.CompoundName <> ""
    RETURN c1, r, c2;
    """

    # Open a session and execute the query
    with driver.session() as session:
        result = session.run(query, compound_names=compound_names)

        # Fetch all records to consume the result immediately
        records = result.graph()

    return records
