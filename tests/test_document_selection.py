
from uuid import uuid4

import pytest

from app.retrieval.document_context import (
    DEFAULT_DOCUMENT_ID,
    get_document_id,
    use_document,
)

import app.retrieval.standard_rag as standard
import app.retrieval.hybrid_rag as hybrid
import app.retrieval.rag_fusion as fusion
import app.graph.graph_rag as graph


def test_default_document():
    assert get_document_id() == DEFAULT_DOCUMENT_ID


def test_document_context_is_restored():
    document_a = str(uuid4())
    document_b = str(uuid4())

    with use_document(document_a):
        assert get_document_id() == document_a

        with use_document(document_b):
            assert get_document_id() == document_b

        assert get_document_id() == document_a

    assert get_document_id() == DEFAULT_DOCUMENT_ID


def test_invalid_document_id_rejected():
    with pytest.raises(ValueError):
        with use_document("invalid-document"):
            pass


@pytest.mark.parametrize(
    "module,function_name",
    [
        (standard, "retrieve"),
        (hybrid, "hybrid_retrieve"),
        (fusion, "retrieve_for_query"),
    ],
)
def test_search_uses_selected_document(
    monkeypatch, module, function_name
):
    document_id = str(uuid4())
    captured = {}

    def fake_search(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        module,
        "get_embedding",
        lambda question: [0.0] * 1536,
    )

    monkeypatch.setattr(
        module.search_client,
        "search",
        fake_search,
    )

    with use_document(document_id):
        getattr(module, function_name)(
            "Test question",
            top_k=3,
        )

    assert captured["filter"] == (
        f"document_id eq '{document_id}'"
    )


def test_graph_uses_selected_document(monkeypatch):
    document_id = str(uuid4())
    captured = {}

    monkeypatch.setattr(
        graph,
        "identify_entities",
        lambda question: ["agentic ai"],
    )

    def fake_execute_query(query, **kwargs):
        captured.update(kwargs)
        return [], None, None

    monkeypatch.setattr(
        graph.driver,
        "execute_query",
        fake_execute_query,
    )

    with use_document(document_id):
        graph.retrieve_graph_facts(
            "What is Agentic AI?"
        )

    assert captured["document_id"] == document_id
