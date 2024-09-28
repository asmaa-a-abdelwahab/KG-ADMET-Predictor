# neo4j_utils.py

import logging
from typing import Any, List

from neo4j import Driver, GraphDatabase
from utils.config import logger


class Neo4jConnectionError(Exception):
    """
    Exception raised when there is a problem connecting to the Neo4j database.
    """

    pass


class Neo4jBase:
    """
    Base class for handling connections to the Neo4j database.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        """
        Initialize the Neo4jBase object.

        :param uri: The URI of the Neo4j database.
        :param user: The username to use for the Neo4j connection.
        :param password: The password to use for the Neo4j connection.
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def connect_to_neo4j(self) -> None:
        """
        Connect to the Neo4j database using the provided credentials.
        """
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
        """
        Close the Neo4j connection.
        """
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed successfully.")


def create_indexes(driver: Driver) -> None:
    """
    Create indexes for the Compound and Gene nodes to optimize queries.
    """
    index_queries = [
        "CREATE INDEX IF NOT EXISTS FOR (c:Compound) ON (c.CompoundName)",
        "CREATE INDEX IF NOT EXISTS FOR (g:Gene) ON (g.GeneSymbol)",
    ]

    try:
        with driver.session() as session:
            for query in index_queries:
                session.run(query)
                logger.info(f"Index created successfully or already exists: {query}")
    except Exception as e:
        logger.error(f"Error creating indexes: {e}")


def execute_query(driver: Driver, query: str) -> List[List[Any]]:
    """
    Execute a query against the Neo4j database.

    :param driver: The Neo4j driver object.
    :param query: The Cypher query to execute.
    :return: A list of lists containing the results of the query.
    """
    try:
        with driver.session() as session:
            result = session.run(query)
            return result
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return []


def get_compound_names(driver: Driver) -> List[str]:
    """
    Retrieve a list of compound names from the Neo4j database.

    :param driver: The Neo4j driver object.
    :return: A list of compound names.
    """
    query = "MATCH (c:Compound) RETURN c.CompoundName AS name"

    # Execute the query and ensure all results are consumed before processing
    with driver.session() as session:
        result = session.run(query)
        records = result.data()  # Fetch all records as a list of dictionaries

    # Return the names extracted from the result
    return [record["name"] for record in records]


def get_gene_symbols(driver: Driver) -> List[str]:
    """
    Retrieve a list of gene symbols from the Neo4j database.

    :param driver: The Neo4j driver object.
    :return: A list of gene symbols.
    """
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

    :param driver: The Neo4j driver object.
    :param compound_names: List of compound names to match.
    :return: List of results, each containing c1, r, and c2.
    """
    # compound_names is a required parameter, so raise an error if it's not provided
    if not compound_names:
        raise ValueError("compound_names must be provided and cannot be None or empty.")

    # Construct the Cypher query with parameters to avoid issues with string formatting
    query = """
    PROFILE MATCH (c1:Compound)-[r:IS_SIMILAR_TO]->(c2:Compound)
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


def show_bioassays(
    driver: Driver, compound_names: List[str] = None, gene_symbols: List[str] = None
) -> List[List[Any]]:
    """
    Retrieve BioAssays that have a STUDIES relationship with any Gene in the gene_symbols list
    and/or BioAssays that have an EVALUATES relationship with any Compound in the compound_names list.

    :param driver: The Neo4j driver object.
    :param compound_names: List of compound names to match (can be None if not provided).
    :param gene_symbols: List of gene symbols to match (can be None if not provided).
    :return: List of results containing BioAssays and their relationships.
    """
    # Ensure at least one of gene_symbols or compound_names is provided
    if not (gene_symbols or compound_names):
        raise ValueError("Either gene_symbols or compound_names must be provided.")

    # Start building the Cypher query based on available input
    query_parts = []
    return_elements = ["ba"]  # Always return the BioAssay node
    params = {}

    if gene_symbols:
        query_parts.append("""
        MATCH (ba:BioAssay)-[r1:STUDIES]->(g:Gene)
        WHERE g.GeneSymbol IN $gene_symbols
        """)
        return_elements.extend(["g", "r1"])  # Return gene and STUDIES relationship
        params["gene_symbols"] = gene_symbols

    if compound_names:
        query_parts.append("""
        OPTIONAL MATCH (ba)-[r2:EVALUATES]->(c:Compound)
        WHERE c.CompoundName IN $compound_names
        """)
        return_elements.extend(
            ["c", "r2"]
        )  # Return compound and EVALUATES relationship
        params["compound_names"] = compound_names

    # Combine query parts and return the BioAssay nodes with relationships
    query = " ".join(query_parts) + f" RETURN {', '.join(return_elements)}"

    # Open a session and execute the query
    with driver.session() as session:
        result = session.run(query, **params)

        # Fetch all records and return them
        records = result.graph()

    return records


def show_cooccurrence(
    driver: Driver, compound_names: List[str], gene_symbols: List[str] = None
) -> List[List[Any]]:
    """
    Retrieve the co-occurrence of Compounds with other Compounds and/or Genes in the literature.

    :param driver: The Neo4j driver object.
    :param compound_names: List of compound names to match.
    :param gene_symbols: List of gene symbols to match (can be None if not provided).
    :return: List of results containing co-occurrences.
    """

    # compound_names is a required parameter, so raise an error if it's not provided
    if not compound_names:
        raise ValueError("compound_names must be provided and cannot be None or empty.")

    # Start building the Cypher query based on available input
    query_parts = []
    return_elements = ["c1"]  # Always return the Compound node (c1)
    params = {"compound_names": compound_names}

    # Add co-occurrence between compounds (compound-compound co-occurrence) first
    query_parts.append("""
    MATCH (c1:Compound)-[r2:CO_OCCURS_IN_LITERATURE]->(c2:Compound)
    WHERE c1.CompoundName IN $compound_names
    """)
    return_elements.extend(["c2", "r2"])  # Return compound-compound co-occurrence

    # Add co-occurrence between compounds and genes if gene symbols are provided
    if gene_symbols:
        query_parts.append("""
        MATCH (g:Gene)-[r1:CO_OCCURS_IN_LITERATURE]->(c2:Compound)
        WHERE g.GeneSymbol IN $gene_symbols
        """)
        return_elements.extend(
            ["g", "r1"]
        )  # Return gene and co-occurrence relationship
        params["gene_symbols"] = gene_symbols

    # Combine query parts and return the co-occurrence relationships
    query = " ".join(query_parts) + f" RETURN {', '.join(return_elements)}"

    # Open a session and execute the query
    with driver.session() as session:
        result = session.run(query, **params)

        # Fetch all records and return them
        records = result.graph()

    return records


def show_pubchem_interactions(
    driver: Driver, compound_names: List[str], gene_symbols: List[str]
) -> List[List[Any]]:
    """
    Retrieve the interactions between Compounds and Genes, excluding specific non-PubChem relationships.

    :param driver: The Neo4j driver object.
    :param compound_names: List of compound names to match.
    :param gene_symbols: List of gene symbols to match.
    :return: List of results containing co-occurrences.
    """

    # Ensure gene_symbols are mandatory
    if not gene_symbols:
        raise ValueError("gene_symbols must be provided and cannot be None or empty.")

    if not compound_names:
        raise ValueError("compound_names must be provided and cannot be None or empty.")

    # Start building the Cypher query based on available input
    query_parts = []
    return_elements = ["c1", "g", "r2"]  # Return compound, gene, and relationship
    params = {"compound_names": compound_names, "gene_symbols": gene_symbols}
    non_pubchem_rel = [
        "InhibitorORInducerORModulator",
        "CO_OCCURS_IN_LITERATURE",
        "INTERACTS_WITH",
        "InhibitorORSubstrate",
    ]

    # Add co-occurrence between compounds and genes, excluding specific relationships
    query_parts.append("""
    MATCH (c1:Compound)-[r2]->(g:Gene)
    WHERE c1.CompoundName IN $compound_names 
    AND g.GeneSymbol IN $gene_symbols
    AND NOT type(r2) IN $non_pubchem_rel
    """)

    params["non_pubchem_rel"] = non_pubchem_rel

    # Combine query parts and return the co-occurrence relationships
    query = " ".join(query_parts) + f" RETURN {', '.join(return_elements)}"

    # Open a session and execute the query
    with driver.session() as session:
        result = session.run(query, **params)

        # Fetch all records and return them
        records = result.graph()

    return records


def show_external_interactions(
    driver: Driver, compound_names: List[str], gene_symbols: List[str]
) -> List[List[Any]]:
    """
    Retrieve the interactions between Compounds and Genes from external sources, excluding specific PubChem relationships.

    :param driver: The Neo4j driver object.
    :param compound_names: List of compound names to match.
    :param gene_symbols: List of gene symbols to match.
    :return: List of results containing interactions.
    """

    # Ensure gene_symbols are mandatory
    if not gene_symbols:
        raise ValueError("gene_symbols must be provided and cannot be None or empty.")

    if not compound_names:
        raise ValueError("compound_names must be provided and cannot be None or empty.")

    # Start building the Cypher query based on available input
    query_parts = []
    return_elements = [
        "c1",
        "g",
        "r2",
    ]  # Return compound name, gene symbol, and relationship
    params = {"compound_names": compound_names, "gene_symbols": gene_symbols}

    # Add co-occurrence between compounds and genes
    query_parts.append("""
    MATCH (c1:Compound)-[r2:INTERACTS_WITH]->(g:Gene)
    WHERE c1.CompoundName IN $compound_names 
    AND g.GeneSymbol IN $gene_symbols
    """)

    # Combine query parts and return the co-occurrence relationships
    query = " ".join(query_parts) + f" RETURN {', '.join(return_elements)}"

    # Open a session and execute the query
    with driver.session() as session:
        result = session.run(query, **params)

        # Fetch all records and return them
        records = result.graph()

    return records
