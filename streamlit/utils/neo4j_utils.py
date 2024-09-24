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
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info("Successfully connected to the Neo4j database.")
        except Exception as e:
            logger.error(f"Failed to connect to the Neo4j database: {e}")
            raise Neo4jConnectionError("Failed to connect to the Neo4j database.") from e

    def close(self) -> None:
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed successfully.")


def execute_query(driver: Driver, query: str) -> List[List[Any]]:
    try:
        with driver.session() as session:
            result = session.run(query)
            return [record.values() for record in result] if result else []
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return []


def get_compound_names(driver: Driver) -> List[str]:
    query = "MATCH (c:Compound) RETURN c.CompoundName AS name"
    return [record[0] for record in execute_query(driver, query)]


def get_gene_symbols(driver: Driver) -> List[str]:
    query = "MATCH (g:Gene) RETURN g.GeneSymbol AS symbol"
    return [record[0] for record in execute_query(driver, query)]


def get_similar_compounds(driver: Driver, compound_names: List[str]) -> List[List[Any]]:
    query = f"""
    MATCH (c1:Compound)-[r:IS_SIMILAR_TO]->(c2:Compound)
    WHERE c1.CompoundName IN {compound_names}
    AND c2.CompoundName IS NOT NULL AND c2.CompoundName <> ""
    RETURN c1, r, c2;
    """
    return execute_query(driver, query)
