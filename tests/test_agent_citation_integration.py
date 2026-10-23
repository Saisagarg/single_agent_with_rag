
from types import SimpleNamespace

import app.agents.agent_controller as agent


def mock_response(text):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text)
            )
        ]
    )


def sample_evidence():
    return [
        agent.Evidence(
            content="Agentic AI uses planning and tools.",
            chunk_id="test-chunk-1",
            file_name="sample.pdf",
            page_number=1,
            tool="vector",
        )
    ]


def test_valid_answer_citations(monkeypatch):
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return mock_response(
            "Agentic AI uses planning and tools [1]."
        )

    monkeypatch.setattr(
        agent.ai_client.chat.completions,
        "create",
        fake_create,
    )

    answer = agent.generate_answer(
        "What does Agentic AI use?",
        sample_evidence(),
    )

    assert "[1]" in answer
    assert len(calls) == 1


def test_invalid_citation_triggers_retry(monkeypatch):
    responses = iter([
        "Agentic AI uses planning [9].",
        "Agentic AI uses planning and tools [1].",
    ])

    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return mock_response(next(responses))

    monkeypatch.setattr(
        agent.ai_client.chat.completions,
        "create",
        fake_create,
    )

    answer = agent.generate_answer(
        "What does Agentic AI use?",
        sample_evidence(),
    )

    assert "[1]" in answer
    assert "[9]" not in answer
    assert len(calls) == 2


def test_repeated_invalid_citations(monkeypatch):
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return mock_response(
            "Unsupported answer [99]."
        )

    monkeypatch.setattr(
        agent.ai_client.chat.completions,
        "create",
        fake_create,
    )

    answer = agent.generate_answer(
        "What does Agentic AI use?",
        sample_evidence(),
    )

    assert len(calls) == 2
    assert "could not generate an answer" in answer


def test_no_evidence_skips_generation(monkeypatch):
    def forbidden_create(**kwargs):
        raise AssertionError(
            "LLM must not run without evidence"
        )

    monkeypatch.setattr(
        agent.ai_client.chat.completions,
        "create",
        forbidden_create,
    )

    answer = agent.generate_answer(
        "Unsupported question",
        [],
    )

    assert "could not find sufficient evidence" in answer.lower()