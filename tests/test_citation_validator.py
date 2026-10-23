
from app.agents.citation_validator import (
    validate_citations
)

from app.agents.agent_controller import Evidence


def evidence():
    return [
        Evidence(
            content="Agentic AI uses planning.",
            chunk_id="chunk-1",
            file_name="sample.pdf",
            page_number=1,
            tool="vector",
        ),
        Evidence(
            content="Agentic AI uses tools.",
            chunk_id="chunk-2",
            file_name="sample.pdf",
            page_number=2,
            tool="hybrid",
        ),
    ]


def test_valid_citations():
    result = validate_citations(
        "Agentic AI uses planning [1] and tools [2].",
        evidence(),
    )

    assert result.valid
    assert result.cited_indices == [1, 2]


def test_invalid_citation():
    result = validate_citations(
        "Agentic AI uses planning [5].",
        evidence(),
    )

    assert not result.valid
    assert result.invalid_indices == [5]


def test_missing_citations():
    result = validate_citations(
        "Agentic AI uses planning.",
        evidence(),
    )

    assert not result.valid


def test_no_evidence():
    result = validate_citations(
        "Agentic AI uses planning [1].",
        [],
    )

    assert not result.valid
