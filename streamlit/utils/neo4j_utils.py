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
            logger.info(f"Connecting to Neo4j at {self.uri}")
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
            logger.info("Closing Neo4j connection")
            self.driver.close()
            logger.info("Neo4j connection closed successfully.")


def create_indexes(driver: Driver) -> None:
    """
    Ensure that all necessary indexes and constraints are created for efficient querying of the Neo4j database.
    If an index or constraint doesn't exist, it will be created.
    """
    index_queries = [
        # Indexes for Compound and Gene nodes
        "CREATE INDEX IF NOT EXISTS FOR (c:Compound) ON (c.CompoundName)",
        "CREATE INDEX IF NOT EXISTS FOR (c:Compound) ON (c.CompoundID)",
        "CREATE INDEX IF NOT EXISTS FOR (g:Gene) ON (g.GeneID)",
        "CREATE INDEX IF NOT EXISTS FOR (g:Gene) ON (g.GeneSymbol)",
        # Indexes for BioAssay and Protein nodes
        "CREATE INDEX IF NOT EXISTS FOR (ba:BioAssay) ON (ba.AssayID)",
        "CREATE INDEX IF NOT EXISTS FOR (ba:BioAssay) ON (ba.AssayName)",
        "CREATE INDEX IF NOT EXISTS FOR (p:Protein) ON (p.ProteinRefSeqAccession)",
    ]

    constraint_queries = [
        # Constraints to ensure uniqueness of the nodes based on their properties
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Compound) REQUIRE c.CompoundID IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gene) REQUIRE g.GeneID IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (ba:BioAssay) REQUIRE ba.AssayID IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Protein) REQUIRE p.ProteinRefSeqAccession IS UNIQUE",
    ]

    try:
        with driver.session() as session:
            # Execute index queries
            for query in index_queries:
                logger.info(f"Ensuring index: {query}")
                session.run(query)

            # Execute constraint queries
            for query in constraint_queries:
                logger.info(f"Ensuring constraint: {query}")
                session.run(query)
    except Exception as e:
        logger.error(f"Error creating indexes or constraints: {e}")


def execute_query(driver: Driver, query: str) -> List[List[Any]]:
    """
    Execute a query against the Neo4j database.

    :param driver: The Neo4j driver object.
    :param query: The Cypher query to execute.
    :return: A list of lists containing the results of the query.
    """
    try:
        logger.info(f"Executing query: {query}")
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
        logger.info("Retrieving compound names")
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
        logger.info("Retrieving gene symbols")
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

    logger.info("Retrieving similar compounds for %s", compound_names)

    # Open a session and execute the query
    with driver.session() as session:
        result = session.run(query, compound_names=compound_names)
        records = result.graph()

    return records


def show_bioassays(driver: Driver, compound_names: List[str]) -> List[dict]:
    """
    Retrieve BioAssays that have an EVALUATES relationship with any Compound in the compound_names list.

    This query will return a list of dictionaries, each containing a BioAssay, a Gene, and a Compound, along with
    their EVALUATES and STUDIES relationships.

    :param driver: The Neo4j driver object.
    :param compound_names: List of compound names to match.
    :return: List of results containing BioAssays, Genes, and Compounds with their relationships.
    """
    # Ensure compound_names is provided
    if not compound_names:
        raise ValueError(
            "You must select compounds to search. Please select one compound for better performance."
        )

    # Cypher query to retrieve BioAssays, Genes, and Compounds based on the compounds in the list
    query = """
    MATCH (ba:BioAssay)
    MATCH (ba)-[r2:EVALUATES]->(c:Compound)
    MATCH (ba)-[r1:STUDIES]->(g:Gene)
    WHERE c.CompoundName IN $compound_names
    RETURN ba, g, r1, c, r2
    """

    logger.info("Retrieving BioAssays for %s", compound_names)

    # Execute the query with compound_names as a parameter
    # This query will return a list of dictionaries, each containing a BioAssay, a Gene, and a Compound, along with
    # their EVALUATES and STUDIES relationships.
    params = {"compound_names": compound_names}

    with driver.session() as session:
        result = session.run(query, **params)
        records = result.graph()

    return records


def show_cooccurrence_cpd_cpd(
    driver: Driver, compound_names: List[str], gene_symbols: List[str] = None
) -> List[dict]:
    """
    Retrieve the co-occurrence of Compounds with other Compounds and/or Genes in the literature.

    :param driver: The Neo4j driver object.
    :param compound_names: List of compound names to match.
    :param gene_symbols: List of gene symbols to match (optional).
    :return: List of results containing co-occurrences.
    """
    # Ensure compound_names is provided
    if not compound_names:
        raise ValueError("compound_names must be provided and cannot be empty.")

    query = """
    MATCH (c1:Compound)-[r1:CO_OCCURS_IN_LITERATURE]->(c2:Compound)
    WHERE c1.CompoundName IN $compound_names
    RETURN c1, c2, r1;
    """

    params = {"compound_names": compound_names, "gene_symbols": gene_symbols}

    logger.info(
        "Retrieving co-occurrences between %s and other compounds",
        compound_names,
    )

    with driver.session() as session:
        result = session.run(query, **params)
        records = result.graph()

    return records


def show_cooccurrence_cpd_gene(
    driver: Driver, compound_names: List[str], gene_symbols: List[str]
) -> List[dict]:
    """
    Retrieve the co-occurrence of Compounds with Genes in the literature.

    :param driver: The Neo4j driver object.
    :param compound_names: List of compound names to match.
    :param gene_symbols: List of gene symbols to match.
    :return: List of results containing co-occurrences.
    """
    # Ensure both compound_names and gene_symbols are provided
    if not compound_names or not gene_symbols:
        raise ValueError("Both compound_names and gene_symbols must be provided.")

    query = """
    MATCH (g:Gene)-[r:CO_OCCURS_IN_LITERATURE]->(c:Compound)
    WHERE c.CompoundName IN $compound_names
    AND g.GeneSymbol IN $gene_symbols
    RETURN c, r, g;
    """

    params = {"compound_names": compound_names, "gene_symbols": gene_symbols}

    logger.info(
        "Retrieving co-occurrences between %s and %s",
        compound_names,
        gene_symbols,
    )

    with driver.session() as session:
        result = session.run(query, **params)

        # Extract data from the result
        records = result.graph()

    return records


def show_pubchem_interactions(
    driver: Driver, compound_names: List[str], gene_symbols: List[str]
) -> List[List[Any]]:
    """
    Retrieve the interactions between Compounds and Genes from PubChem, excluding specific non-PubChem relationships.

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
    non_pubchem_rel = [
        "InhibitorORInducerORModulator",
        "CO_OCCURS_IN_LITERATURE",
        "INTERACTS_WITH",
        "InhibitorORSubstrate",
    ]

    logger.info(
        "Retrieving interactions between %s and %s, excluding %s",
        compound_names,
        gene_symbols,
        non_pubchem_rel,
    )

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

    logger.info(
        "Retrieving external interactions between %s and %s",
        compound_names,
        gene_symbols,
    )

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


def get_neo4j_statistics(driver):
    """
    Retrieves statistics from the Neo4j database, including counts of compounds, genes, bioassays, and relationships.

    :param driver: The Neo4j driver object.
    :return: Dictionary containing the counts of different node and relationship types.
    """
    statistics = {}

    queries = {
        # Count the number of compounds
        "compounds_count": "MATCH (c:Compound) RETURN COUNT(c) AS count",
        # Count the number of genes
        "genes_count": "MATCH (g:Gene) RETURN COUNT(g) AS count",
        # Count the number of bioassays
        "bioassays_count": "MATCH (ba:BioAssay) RETURN COUNT(ba) AS count",
        # Count the number of relationships
        "relationships_count": "MATCH ()-[r]->() RETURN COUNT(r) AS count",
    }

    logger.info("Retrieving statistics from Neo4j database")

    with driver.session() as session:
        for stat_name, query in queries.items():
            result = session.run(query)
            statistics[stat_name] = result.single()["count"]

    return statistics
