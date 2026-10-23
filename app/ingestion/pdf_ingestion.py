
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from azure.identity import (
    get_bearer_token_provider,
)
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
from app.azure_auth import create_credential

credential = create_credential()

token_provider = get_bearer_token_provider(
    credential,
    "https://ai.azure.com/.default",
)

openai_client = OpenAI(
    base_url=os.environ["AZURE_OPENAI_BASE_URL"],
    api_key=token_provider,
)

search_client = SearchClient(
    endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
    index_name=os.environ["AZURE_SEARCH_INDEX"],
    credential=AzureKeyCredential(
        os.environ["AZURE_SEARCH_API_KEY"]
    ),
)

blob_service = BlobServiceClient(
    account_url=(
        f"https://{os.environ['AZURE_STORAGE_ACCOUNT']}"
        ".blob.core.windows.net"
    ),
    credential=credential,
)

def upload_pdf(pdf_path, document_id):
    container = blob_service.get_container_client(
        os.environ["AZURE_BLOB_CONTAINER"]
    )

    blob_name = f"{document_id}/{pdf_path.name}"

    with pdf_path.open("rb") as file:
        container.upload_blob(
            name=blob_name,
            data=file,
            overwrite=False,
        )

    print("PDF uploaded to Blob Storage:", blob_name)


def extract_pages(pdf_path):
    reader = PdfReader(str(pdf_path))

    for page_number, page in enumerate(
        reader.pages, start=1
    ):
        text = page.extract_text() or ""

        if text.strip():
            yield page_number, text


def chunk_text(text, chunk_size=1000, overlap=150):
    if chunk_size <= overlap or overlap < 0:
        raise ValueError("Invalid chunk settings")

    text = " ".join(text.split())
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size]

        if chunk.strip():
            yield chunk


def embed(text):
    response = openai_client.embeddings.create(
        model=os.environ["AZURE_EMBEDDING_DEPLOYMENT"],
        input=text,
    )

    vector = response.data[0].embedding

    if len(vector) != 1536:
        raise ValueError(
            f"Unexpected embedding dimensions: {len(vector)}"
        )

    return vector


def ingest_pdf(pdf_path):
    pdf_path = Path(pdf_path).resolve()

    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    document_id = str(uuid.uuid4())

    chunks = []

    for page_number, text in extract_pages(pdf_path):
        for chunk in chunk_text(text):
            chunks.append((page_number, chunk))

    if not chunks:
        raise ValueError(
            "No extractable text found. "
            "Scanned PDFs require OCR."
        )

    print("Document:", pdf_path.name)
    print("Total chunks:", len(chunks))

    # Store original document before indexing.
    upload_pdf(pdf_path, document_id)

    # Keep test batches small for the Free tier.
    batch = []

    for number, (page_number, content) in enumerate(
        chunks, start=1
    ):
        print(f"Embedding chunk {number}/{len(chunks)}")

        batch.append({
            "chunk_id": f"{document_id}-{number}",
            "document_id": document_id,
            "content": content,
            "content_vector": embed(content),
            "file_name": pdf_path.name,
            "page_number": page_number,
        })

        if len(batch) >= 10:
            upload_batch(batch)
            batch.clear()

    if batch:
        upload_batch(batch)

    print("PDF ingestion completed successfully!")
    print("Document ID:", document_id)
    return document_id


def upload_batch(batch):
    result = search_client.upload_documents(
        documents=batch
    )

    failures = [
        item.key for item in result
        if not item.succeeded
    ]

    if failures:
        raise RuntimeError(
            f"Failed to index chunks: {failures}"
        )

    print(f"Indexed {len(batch)} chunks")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m app.ingestion.pdf_ingestion "
            "data/documents/sample.pdf"
        )

    ingest_pdf(sys.argv[1])
