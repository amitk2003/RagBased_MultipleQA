import os
from neo4j import GraphDatabase
import chromadb

# Initialize Global Clients
_neo4j_driver = None
_chroma_client = None

def get_neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        try:
            _neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
            _neo4j_driver.verify_connectivity()
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            _neo4j_driver = None
    return _neo4j_driver

def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        try:
            # Persistent client in the api container
            _chroma_client = chromadb.PersistentClient(path="data/chroma_db")
        except Exception as e:
            print(f"Failed to connect to Chroma: {e}")
            _chroma_client = None
    return _chroma_client
