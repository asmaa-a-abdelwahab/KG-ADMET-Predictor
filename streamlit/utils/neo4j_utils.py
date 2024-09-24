import logging
from neo4j import Driver, GraphDatabase
from typing import Any, List
import streamlit as st


class Neo4jConnectionError(Exception):
    """Custom exception for Neo4j connection errors."""

    pass


class Neo4jBase:
    """
    Base class to manage connections with the Neo4j database.
    """

    def __init__(
        self,
        logger: logging.Logger = None,
        uri: str = "bolt://0.0.0.0:7687",
        user: str = "neo4j",
        password: str = "cyp450kg",
    ) -> None:
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logger or logging.getLogger(__name__)

    def connect_to_neo4j(self) -> None:
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            self.logger.info("Successfully connected to the Neo4j database.")
        except Exception as e:
            self.logger.error("Failed to connect to the Neo4j database: %s", e)
            raise Neo4jConnectionError(
                "Failed to connect to the Neo4j database."
            ) from e

    def close(self) -> None:
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed successfully.")


def execute_query(driver: Driver, query: str) -> List[List[Any]]:
    try:
        with driver.session() as session:
            result = session.run(query)
            return [record.values() for record in result] if result else []
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return []


def get_compound_names(driver: Driver) -> List[str]:
    query = "MATCH (c:Compound) RETURN c.CompoundName AS name"
    compounds = execute_query(driver, query)
    return [record[0] for record in compounds if record]


def get_gene_symbols(driver: Driver) -> List[str]:
    query = "MATCH (g:Gene) RETURN g.GeneSymbol AS symbol"
    genes = execute_query(driver, query)
    return [record[0] for record in genes if record]


def get_similar_compounds(driver: Driver, compound_names: List[str]) -> List[List[Any]]:
    query = f"""
            MATCH (c1:Compound)-[r:IS_SIMILAR_TO]->(c2:Compound)
            WHERE c1.CompoundName IN {compound_names}
            AND c2.CompoundName IS NOT NULL AND c2.CompoundName <> ""
            RETURN c1, r, c2;
            """
    return execute_query(driver, query)
