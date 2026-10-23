
from types import SimpleNamespace

import app.agents.verified_agent as verified
from app.agents.agent_controller import Evidence


def sample_result():
    return SimpleNamespace(
        answer="Agentic AI uses planning [1].",
        successful_tool="vector",
        evidence=[
            Evidence(
                content="Agentic AI uses planning.",
                chunk_id="chunk-1",
                file_name="sample.pdf",
                page_number=1,
                tool="vector",
            )
        ],
        trace=[{"tool": "vector", "sufficient": True}],
    )


def test_grounded_answer_is_returned(monkeypatch):
    monkeypatch.setattr(
        verified,
        "run_agent",
        lambda question, top_k=5: sample_result(),
    )

    monkeypatch.setattr(
        verified,
        "evaluate_groundedness",
        lambda **kwargs: SimpleNamespace(
            grounded=True,
            reason="Claims supported",
        ),
    )

    result = verified.run_verified_agent("What is Agentic AI?")

    assert result.grounded is True
    assert "[1]" in result.answer
    assert len(result.verification_trace) == 1


def test_unsupported_answer_is_rejected(monkeypatch):
    monkeypatch.setattr(
        verified,
        "run_agent",
        lambda question, top_k=5: sample_result(),
    )

    monkeypatch.setattr(
        verified,
        "evaluate_groundedness",
        lambda **kwargs: SimpleNamespace(
            grounded=False,
            reason="Unsupported claim",
        ),
    )

    monkeypatch.setattr(
        verified,
        "generate_answer",
        lambda question, evidence: "Unsupported [1].",
    )

    result = verified.run_verified_agent("What is Agentic AI?")

    assert result.grounded is False
    assert result.evidence == []
    assert "could not produce" in result.answer.lower()
    assert len(result.verification_trace) == 2
