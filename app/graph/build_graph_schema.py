
import os
from pathlib import Path
from app.retrieval.document_context import DEFAULT_DOCUMENT_ID
from dotenv import load_dotenv
from neo4j import GraphDatabase
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DOCUMENT_ID = "a853569c-176a-4190-b92d-4d943d943bbd"

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(
        os.environ["NEO4J_USERNAME"],
        os.environ["NEO4J_PASSWORD"],
    ),
)

database = os.getenv("NEO4J_DATABASE", "neo4j")

search_client = SearchClient(
    endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
    index_name=os.environ["AZURE_SEARCH_INDEX"],
    credential=AzureKeyCredential(
        os.environ["AZURE_SEARCH_API_KEY"]
    ),
)


def create_schema():
    print("Creating Neo4j graph constraints...")

    constraints = [
        (
            "CREATE CONSTRAINT document_id_unique "
            "IF NOT EXISTS FOR (d:Document) "
            "REQUIRE d.id IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT chunk_id_unique "
            "IF NOT EXISTS FOR (c:Chunk) "
            "REQUIRE c.id IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT entity_key_unique "
            "IF NOT EXISTS FOR (e:Entity) "
            "REQUIRE e.key IS UNIQUE"
        ),
        (
            "CREATE CONSTRAINT fact_id_unique "
            "IF NOT EXISTS FOR (f:Fact) "
            "REQUIRE f.id IS UNIQUE"
        ),
    ]

    for query in constraints:
        driver.execute_query(
            query,
            database_=database,
        )

    print("Graph constraints created successfully!")



def load_chunks(document_id=DEFAULT_DOCUMENT_ID):
    print("Loading chunks for:", document_id)

    safe_id = document_id.replace("'", "''")

    results = search_client.search(
        search_text="*",
        filter=f"document_id eq '{safe_id}'",
        select=[
            "chunk_id",
            "document_id",
            "content",
            "file_name",
            "page_number",
        ],
        top=1000,
    )

    chunks = [dict(result) for result in results]

    if not chunks:
        raise RuntimeError(
            "No indexed chunks found for this document."
        )

    print("Retrieved chunks:", len(chunks))
    return chunks


def build_graph(chunks, document_id=DEFAULT_DOCUMENT_ID):
    print("Building document and chunk graph...")

    query = """
    MERGE (d:Document {id: $document_id})
    SET d.file_name = $file_name

    WITH d
    UNWIND $chunks AS chunk

    MERGE (c:Chunk {id: chunk.chunk_id})
    SET c.content = chunk.content,
        c.page_number = chunk.page_number,
        c.document_id = $document_id

    MERGE (d)-[:HAS_CHUNK]->(c)
    """

    driver.execute_query(
        query,
        document_id=document_id,
        file_name=chunks[0]["file_name"],
        chunks=chunks,
        database_=database,
    )

    print("Document graph created successfully!")


def verify_graph(document_id=DEFAULT_DOCUMENT_ID):
    records, _, _ = driver.execute_query(
        """
        MATCH (d:Document {id: $document_id})
              -[:HAS_CHUNK]->(c:Chunk)
        RETURN
            d.file_name AS file,
            count(c) AS chunks
        """,
        document_id=document_id,
        database_=database,
    )

    if not records:
        raise RuntimeError("Document graph was not found")

    record = records[0]

    print("\nGRAPH VERIFICATION")
    print("Document:", record["file"])
    print("Chunks:", record["chunks"])

    if record["chunks"] != 8:
        print(
            "Note: Expected 8 chunks from the previous "
            "ingestion. Check for indexing differences."
        )


def main():
    driver.verify_connectivity()

    create_schema()
    chunks = load_chunks()
    build_graph(chunks)
    verify_graph()

    print("\nStep 10.7 completed successfully!")


if __name__ == "__main__":
    try:
        main()
    finally:
        driver.close()
