
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

uri = os.environ["NEO4J_URI"]
username = os.environ["NEO4J_USERNAME"]
password = os.environ["NEO4J_PASSWORD"]
database = os.getenv("NEO4J_DATABASE", "neo4j")

print("Connecting to Neo4j AuraDB...")

with GraphDatabase.driver(
    uri,
    auth=(username, password),
) as driver:

    driver.verify_connectivity()

    print("Neo4j connection successful!")

    records, _, _ = driver.execute_query(
        "RETURN 'Modular-RAG' AS project, "
        "'Connected' AS status",
        database_=database,
    )

    for record in records:
        print("Project:", record["project"])
        print("Status:", record["status"])

print("Neo4j connection test completed!")
