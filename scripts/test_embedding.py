
import os
from pathlib import Path

from dotenv import load_dotenv
from azure.identity import (
    DefaultAzureCredential,
    get_bearer_token_provider,
)
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

endpoint = os.getenv("AZURE_OPENAI_BASE_URL", "").strip()
deployment = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "").strip()

if not endpoint or not deployment:
    raise ValueError(
        "Missing Azure endpoint or embedding deployment in .env"
    )

endpoint = endpoint.rstrip("/") + "/"

if not endpoint.endswith("/openai/v1/"):
    raise ValueError(
        "AZURE_OPENAI_BASE_URL must end with /openai/v1/"
    )

print("Endpoint:", endpoint)
print("Deployment:", deployment)

credential = DefaultAzureCredential()

token_provider = get_bearer_token_provider(
    credential,
    "https://ai.azure.com/.default"
)

client = OpenAI(
    base_url=endpoint,
    api_key=token_provider,
)

print("Sending embedding request...")

response = client.embeddings.create(
    model=deployment,
    input="Modular-RAG uses Azure AI Search for vector retrieval.",
)

embedding = response.data[0].embedding

print("Embedding request successful!")
print("Vector dimensions:", len(embedding))

assert len(embedding) == 1536
