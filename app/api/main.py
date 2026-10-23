

import re
import tempfile
from pathlib import Path
from uuid import UUID

from app.graph.document_graph_service import (
    process_document_graph,
)

from app.api.document_registry import (
    claim_graph_build,
    update_document_status,
)
from fastapi import File, UploadFile
from starlette.concurrency import run_in_threadpool

from app.ingestion.pdf_ingestion import ingest_pdf

from app.api.document_registry import (
    register_document,
    get_document,
    list_documents,
)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.retrieval.document_context import use_document
from app.retrieval.standard_rag import DOCUMENT_ID
from app.api.schemas import (
    QuestionRequest,
    QuestionResponse,
    EvidenceResponse,
)

from app.agents.verified_agent import run_verified_agent


app = FastAPI(
    title="Modular-RAG Agentic API",
    description=(
        "Agentic Retrieval-Augmented Generation "
        "using Microsoft Azure and Neo4j"
    ),
    version="1.0.0",
)

# Only allow local frontend origins during development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/")
def root():
    return {
        "application": "Modular-RAG Agentic API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    # Process health only. Does not check Azure services.
    return {
        "status": "healthy",
        "service": "modular-rag-backend",
    }



@app.post(
    "/api/ask",
    response_model=QuestionResponse,
)
def ask_question(request: QuestionRequest):
    import logging

    # Until all retrieval tools are fully scoped, accept
    # only the already indexed default document.
    requested_id = (
        str(request.document_id)
        if request.document_id else DOCUMENT_ID
    )


    document = get_document(requested_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document is not registered.",
        )

    if document["status"] not in {
        "search_ready",
        "graph_processing",
        "graph_failed",
        "graph_ready", }:
            
        raise HTTPException(
            status_code=409,
            detail="Document is not ready for retrieval.",
        )

    try:
        with use_document(requested_id):
            result = run_verified_agent(
                question=request.question,
                top_k=5,
            )

        return QuestionResponse(
            question=request.question,
            answer=result.answer,
            grounded=result.grounded,
            successful_tool=result.successful_tool,
            evidence=[
                EvidenceResponse(
                    content=item.content,
                    chunk_id=item.chunk_id,
                    file_name=item.file_name,
                    page_number=item.page_number,
                    tool=item.tool,
                )
                for item in result.evidence
            ],
            retrieval_trace=result.retrieval_trace,
            verification_trace=result.verification_trace,
        )

    except Exception as exc:
        logging.exception("Agentic RAG request failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question.",
        ) from exc



MAX_PDF_SIZE = 10 * 1024 * 1024


@app.post("/api/documents/upload", status_code=201)
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a text-based PDF."""

    original_name = file.filename or ""

    # Prevent path traversal and unsafe blob filenames.
    safe_name = Path(
        original_name.replace("\\", "/")
    ).name

    safe_name = re.sub(
        r"[^A-Za-z0-9._-]",
        "_",
        safe_name,
    )

    if (
        not safe_name
        or safe_name in {".", ".."}
        or not safe_name.lower().endswith(".pdf")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid PDF filename.",
        )

    try:
        contents = await file.read(MAX_PDF_SIZE + 1)
    finally:
        await file.close()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(contents) > MAX_PDF_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the 10 MB upload limit.",
        )

    if not contents.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The file does not have a valid PDF header.",
        )

    try:
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / safe_name
            pdf_path.write_bytes(contents)

            # Run blocking Azure ingestion outside
            # the FastAPI event loop.
            document_id = await run_in_threadpool(
                ingest_pdf,
                pdf_path,
            )

        register_document(document_id, safe_name)

        return {
            "document_id": document_id,
            "file_name": safe_name,
            "status": "search_ready",
            "message": (
                "PDF uploaded and indexed successfully. "
                "Graph processing is pending."
            ),
        }

    except Exception as exc:
        import logging

        logging.exception("PDF ingestion failed")

        raise HTTPException(
            status_code=500,
            detail=(
                "Document ingestion failed. "
                "The document has not been registered as ready."
            ),
        ) from exc


@app.get("/api/documents")
def get_documents():
    """List locally registered uploaded documents."""
    return {
        "documents": list_documents()
    }


@app.get("/api/documents/{document_id}")
def get_document_details(document_id: str):
    try:
        document = get_document(document_id)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Invalid document ID.",
        )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return document

@app.post("/api/documents/{document_id}/build-graph")
async def build_document_graph(document_id: str):
    import logging

    try:
        document_id = str(UUID(document_id))
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Invalid document ID.",
        )

    document = get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    if document_id == DOCUMENT_ID:
        raise HTTPException(
            status_code=409,
            detail="Original sample graph already exists.",
        )

    if document["status"] == "graph_ready":
        return {
            "document_id": document_id,
            "status": "graph_ready",
            "message": "Graph already built.",
        }

    if not claim_graph_build(document_id):
        raise HTTPException(
            status_code=409,
            detail=(
                "Document is not eligible for graph building "
                "or processing is already in progress."
            ),
        )

    try:
        result = await run_in_threadpool(
            process_document_graph,
            document_id,
        )

        update_document_status(
            document_id,
            "graph_ready",
        )

        return {
            "status": "graph_ready",
            **result,
        }

    except Exception as exc:
        logging.exception("Graph processing failed")

        try:
            update_document_status(
                document_id,
                "graph_failed",
            )
        except Exception:
            logging.exception(
                "Unable to record graph failure status"
            )

        raise HTTPException(
            status_code=500,
            detail=(
                "Graph processing failed. "
                "The document remains searchable."
            ),
        ) from exc
