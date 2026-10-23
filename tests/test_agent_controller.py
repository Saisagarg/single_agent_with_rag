
import pytest

import app.agents.agent_controller as agent

from app.agents.agent_controller import (
    Evidence,
    run_agent,
    deduplicate,
    make_context,
)


def sample_evidence(chunk_id="test-chunk-1"):
    return Evidence(
        content=(
            "Agentic AI systems use planning "
            "and tools to accomplish goals."
        ),
        chunk_id=chunk_id,
        file_name="sample.pdf",
        page_number=1,
        tool="vector",
    )


def test_evidence_deduplication():
    """Duplicate chunks must not appear twice."""

    evidence = [
        sample_evidence(),
        sample_evidence(),
        sample_evidence("test-chunk-2"),
    ]

    unique = deduplicate(evidence)

    assert len(unique) == 2
    assert unique[0].chunk_id == "test-chunk-1"
    assert unique[1].chunk_id == "test-chunk-2"


def test_context_contains_source_metadata():
    """Evidence context must retain source details."""

    context = make_context([sample_evidence()])

    assert "sample.pdf" in context
    assert "Page: 1" in context
    assert "test-chunk-1" in context
    assert "[1]" in context


def test_successful_first_tool(monkeypatch):
    """Agent should stop when the first tool is sufficient."""

    calls = []

    monkeypatch.setattr(
        agent,
        "select_tool",
        lambda question: ("vector", "General question"),
    )

    def fake_tool(tool, question, top_k=5):
        calls.append(tool)
        return [sample_evidence()]

    monkeypatch.setattr(agent, "run_tool", fake_tool)

    monkeypatch.setattr(
        agent,
        "evaluate_evidence",
        lambda question, evidence: (
            True,
            "Relevant evidence is available",
        ),
    )

    monkeypatch.setattr(
        agent,
        "generate_answer",
        lambda question, evidence: (
            "Agentic AI uses planning and tools [1]."
        ),
    )

    result = run_agent("What is Agentic AI?")

    assert result.selected_tool == "vector"
    assert result.successful_tool == "vector"
    assert len(result.trace) == 1
    assert calls == ["vector"]
    assert len(result.evidence) == 1


def test_fallback_to_second_tool(monkeypatch):
    """Agent must retry if the first tool is insufficient."""

    calls = []

    monkeypatch.setattr(
        agent,
        "select_tool",
        lambda question: ("vector", "Semantic question"),
    )

    def fake_tool(tool, question, top_k=5):
        calls.append(tool)

        if tool == "vector":
            return []

        if tool == "hybrid":
            return [sample_evidence()]

        return []

    monkeypatch.setattr(agent, "run_tool", fake_tool)

    monkeypatch.setattr(
        agent,
        "evaluate_evidence",
        lambda question, evidence: (
            bool(evidence),
            "Sufficient" if evidence else "Insufficient",
        ),
    )

    monkeypatch.setattr(
        agent,
        "generate_answer",
        lambda question, evidence: "Answer [1]",
    )

    result = run_agent("Explain planning")

    assert calls == ["vector", "hybrid"]
    assert result.successful_tool == "hybrid"
    assert len(result.trace) == 2
    assert result.trace[0]["sufficient"] is False
    assert result.trace[1]["sufficient"] is True


def test_tool_failure_recovery(monkeypatch):
    """A failed tool must not crash the controller."""

    monkeypatch.setattr(
        agent,
        "select_tool",
        lambda question: ("graph", "Relationship question"),
    )

    def fake_tool(tool, question, top_k=5):
        if tool == "graph":
            raise RuntimeError("Simulated Neo4j failure")

        return [sample_evidence()]

    monkeypatch.setattr(agent, "run_tool", fake_tool)

    monkeypatch.setattr(
        agent,
        "evaluate_evidence",
        lambda question, evidence: (
            True,
            "Evidence found",
        ),
    )

    monkeypatch.setattr(
        agent,
        "generate_answer",
        lambda question, evidence: "Answer [1]",
    )

    result = run_agent("How are entities related?")

    assert result.selected_tool == "graph"
    assert result.successful_tool == "hybrid"
    assert len(result.trace) == 2
    assert "error" in result.trace[0]


def test_unsupported_question(monkeypatch):
    """No sufficient evidence means no fabricated answer."""

    monkeypatch.setattr(
        agent,
        "select_tool",
        lambda question: ("vector", "General question"),
    )

    monkeypatch.setattr(
        agent,
        "run_tool",
        lambda tool, question, top_k=5: [],
    )

    monkeypatch.setattr(
        agent,
        "evaluate_evidence",
        lambda question, evidence: (
            False,
            "No relevant evidence",
        ),
    )

    def forbidden_answer(question, evidence):
        pytest.fail(
            "Answer generation must not run "
            "with insufficient evidence"
        )

    monkeypatch.setattr(
        agent,
        "generate_answer",
        forbidden_answer,
    )

    result = run_agent("What is the population of Mars?")

    assert result.successful_tool == ""
    assert len(result.trace) == 4
    assert result.evidence == []
    assert "could not find sufficient evidence" in (
        result.answer.lower()
    )


def test_maximum_four_tool_attempts(monkeypatch):
    """Agent must not execute more than four retrieval tools."""

    attempted = []

    monkeypatch.setattr(
        agent,
        "select_tool",
        lambda question: ("fusion", "Broad question"),
    )

    def fake_tool(tool, question, top_k=5):
        attempted.append(tool)
        return []

    monkeypatch.setattr(agent, "run_tool", fake_tool)

    monkeypatch.setattr(
        agent,
        "evaluate_evidence",
        lambda question, evidence: (
            False,
            "Insufficient evidence",
        ),
    )

    result = run_agent("Unsupported question")

    assert len(attempted) == 4
    assert len(set(attempted)) == 4
    assert len(result.trace) == 4
    assert result.successful_tool == ""
