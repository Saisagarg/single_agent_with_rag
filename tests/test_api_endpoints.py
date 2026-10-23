
from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.api.main as api
from app.retrieval.document_context import get_document_id
from app.agents.agent_controller import Evidence


client = TestClient(api.app)

TEST_DOCUMENT_ID = (
    "11111111-1111-4111-8111-111111111111"
)


def sample_document(status="graph_ready"):
    return {
        "document_id": TEST_DOCUMENT_ID,
        "file_name": "test_document.pdf",
        "status": status,
    }


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_document_list(monkeypatch):
    monkeypatch.setattr(
        api,
        "list_documents",
        lambda: [sample_document()],
    )

    response = client.get("/api/documents")

    assert response.status_code == 200

    documents = response.json()["documents"]

    assert len(documents) == 1
    assert documents[0]["document_id"] == TEST_DOCUMENT_ID


def test_ask_uses_selected_document(monkeypatch):
    monkeypatch.setattr(
        api,
        "get_document",
        lambda document_id: sample_document(),
    )

    def fake_verified_agent(question, top_k=5):
        # Ensure document context reaches the agent.
        assert get_document_id() == TEST_DOCUMENT_ID

        return SimpleNamespace(
            answer="Agentic AI uses planning [1].",
            grounded=True,
            successful_tool="vector",
            evidence=[
                Evidence(
                    content="Agentic AI uses planning.",
                    chunk_id="test-chunk-1",
                    file_name="test_document.pdf",
                    page_number=1,
                    tool="vector",
                )
            ],
            retrieval_trace=[],
            verification_trace=[],
        )

    monkeypatch.setattr(
        api,
        "run_verified_agent",
        fake_verified_agent,
    )

    response = client.post(
        "/api/ask",
        json={
            "question": "What is Agentic AI?",
            "document_id": TEST_DOCUMENT_ID,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["grounded"] is True
    assert data["successful_tool"] == "vector"
    assert len(data["evidence"]) == 1


def test_unknown_document_rejected(monkeypatch):
    monkeypatch.setattr(
        api,
        "get_document",
        lambda document_id: None,
    )

    response = client.post(
        "/api/ask",
        json={
            "question": "What is Agentic AI?",
            "document_id": TEST_DOCUMENT_ID,
        },
    )

    assert response.status_code == 404


def test_invalid_document_id_rejected():
    response = client.post(
        "/api/ask",
        json={
            "question": "What is Agentic AI?",
            "document_id": "invalid-uuid",
        },
    )

    assert response.status_code == 422


def test_invalid_pdf_upload_rejected():
    response = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "invalid.pdf",
                b"This is not a real PDF",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400


def test_existing_graph_is_not_rebuilt(monkeypatch):
    monkeypatch.setattr(
        api,
        "get_document",
        lambda document_id: sample_document(),
    )

    def forbidden_build(document_id):
        raise AssertionError(
            "Graph should not be rebuilt"
        )

    monkeypatch.setattr(
        api,
        "process_document_graph",
        forbidden_build,
    )

    response = client.post(
        f"/api/documents/{TEST_DOCUMENT_ID}/build-graph"
    )

    assert response.status_code == 200
    assert response.json()["status"] == "graph_ready"
