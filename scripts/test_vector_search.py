
import os
from pathlib import Path

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.identity import (
    DefaultAzureCredential,
    get_bearer_token_provider,
)
from openai import OpenAI

# Load environment configuration
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Azure OpenAI configuration
openai_endpoint = os.environ["AZURE_OPENAI_BASE_URL"]
embedding_deployment = os.environ["AZURE_EMBEDDING_DEPLOYMENT"]

# Azure AI Search configuration
search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
search_index = os.environ["AZURE_SEARCH_INDEX"]
search_key = os.environ["AZURE_SEARCH_API_KEY"]

# Authenticate with Microsoft Foundry
credential = DefaultAzureCredential()

token_provider = get_bearer_token_provider(
    credential,
    "https://ai.azure.com/.default"
)

openai_client = OpenAI(
    base_url=openai_endpoint,
    api_key=token_provider
)

# Connect to Azure AI Search
search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=search_index,
    credential=AzureKeyCredential(search_key)
)

def generate_embedding(text):
    response = openai_client.embeddings.create(
        model=embedding_deployment,
        input=text
    )
    return response.data[0].embedding


# Step 1: Prepare a small test document
document_text = (
    "Modular-RAG is a document question-answering "
    "system that supports vector search, RAG Fusion, "
    "knowledge graph retrieval, and agent-based "
    "retrieval tool selection."
)

print("Generating document embedding...")

document_vector = generate_embedding(document_text)

assert len(document_vector) == 1536

# Step 2: Upload one document chunk
document = {
    "chunk_id": "azure-rag-test-001",
    "document_id": "test-document-001",
    "content": document_text,
    "content_vector": document_vector,
    "file_name": "test-document.txt",
    "page_number": 1
}

print("Uploading document to Azure AI Search...")

upload_result = search_client.upload_documents(
    documents=[document]
)

if not upload_result[0].succeeded:
    raise RuntimeError("Document indexing failed")

print("Document uploaded successfully!")

import time

print("Checking document visibility in Azure AI Search...")

indexed_document = None

for attempt in range(10):
    try:
        indexed_document = search_client.get_document(
            key="azure-rag-test-001"
        )
        break
    except Exception as exc:
        print(f"Waiting for indexing, attempt {attempt + 1}: {type(exc).__name__}")
        time.sleep(2)

if indexed_document is None:
    raise RuntimeError(
        "Document was accepted but is not visible in the index."
    )

print("Document is indexed successfully!")
print("Indexed chunk:", indexed_document["chunk_id"])
print("Indexed content:", indexed_document["content"])


# Step 3: Generate query embedding
question = "Which retrieval methods does Modular-RAG support?"

print("Generating query embedding...")

query_vector = generate_embedding(question)

# Step 4: Perform vector similarity search
vector_query = VectorizedQuery(
    vector=query_vector,
    k_nearest_neighbors=3,
    fields="content_vector"
)

results = search_client.search(
    search_text=None,
    vector_queries=[vector_query],
    select=[
        "chunk_id",
        "content",
        "file_name",
        "page_number"
    ],
    top=3
)

# Step 5: Display results
print("\nVECTOR SEARCH RESULTS")
print("-" * 50)

found = False

for result in results:
    found = True
    print("Chunk ID:", result["chunk_id"])
    print("Content:", result["content"])
    print("File:", result["file_name"])
    print("Page:", result["page_number"])
    print("Search score:", result["@search.score"])
    print("-" * 50)

if not found:
    raise RuntimeError("No search results returned")

print("Vector retrieval test completed!")
