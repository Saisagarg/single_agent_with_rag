
import sys
from dataclasses import dataclass, field

from app.agents.agent_controller import (
    run_agent,
    generate_answer,
    print_result,
)
from app.agents.citation_validator import validate_citations
from app.agents.groundedness_evaluator import evaluate_groundedness


REFUSAL = (
    "I could not produce a sufficiently grounded "
    "answer from the uploaded document."
)


@dataclass
class VerifiedResult:
    answer: str
    grounded: bool
    successful_tool: str
    evidence: list = field(default_factory=list)
    retrieval_trace: list = field(default_factory=list)
    verification_trace: list = field(default_factory=list)


def run_verified_agent(question, top_k=5):
    """
    Run Agentic RAG and validate its final answer.

    Allows one correction attempt if the answer
    fails citation or groundedness evaluation.
    """

    result = run_agent(question, top_k=top_k)

    output = VerifiedResult(
        answer=REFUSAL,
        grounded=False,
        successful_tool=result.successful_tool,
        evidence=[],
        retrieval_trace=list(result.trace),
    )

    if not result.successful_tool or not result.evidence:
        output.verification_trace.append({
            "stage": "retrieval",
            "status": "insufficient_evidence",
        })
        return output

    candidate_answer = result.answer

    for attempt in range(2):
        print(f"\nVerification attempt {attempt + 1}/2")

        citations = validate_citations(
            candidate_answer,
            result.evidence,
        )

        if not citations.valid:
            reason = citations.reason
            passed = False
        else:
            evaluation = evaluate_groundedness(
                question=question,
                answer=candidate_answer,
                evidence=result.evidence,
            )

            passed = evaluation.grounded
            reason = evaluation.reason

        output.verification_trace.append({
            "attempt": attempt + 1,
            "citations_valid": citations.valid,
            "grounded": passed,
            "reason": reason,
        })

        if passed:
            output.answer = candidate_answer
            output.grounded = True
            output.evidence = result.evidence
            return output

        if attempt == 0:
            print("Verification failed. Regenerating answer.")

            correction_prompt = (
                f"{question}\n\n"
                "Correction requirements:\n"
                "Use only directly supported claims.\n"
                "Remove unsupported statements.\n"
                "Cite the exact evidence for each claim.\n"
                "If the evidence cannot answer the "
                "question, state that limitation.\n"
                f"Previous verification issue: {reason[:500]}"
            )

            try:
                candidate_answer = generate_answer(
                    correction_prompt,
                    result.evidence,
                )
            except Exception as exc:
                output.verification_trace.append({
                    "stage": "regeneration",
                    "error": type(exc).__name__,
                })
                break

    # Fail closed: do not return an unverified answer.
    return output


def display_verified_result(result):
    print("\n" + "=" * 60)
    print("VERIFIED AGENTIC RAG RESULT")
    print("=" * 60)

    print("\nGrounded:", result.grounded)
    print("Successful tool:", result.successful_tool or "None")

    print("\nFINAL ANSWER")
    print(result.answer)

    print("\nVERIFICATION TRACE")
    for entry in result.verification_trace:
        print(entry)

    if result.grounded:
        print("\nSOURCE REFERENCES")
        for index, item in enumerate(result.evidence, 1):
            print(
                f"[{index}] {item.file_name} "
                f"- Page {item.page_number} "
                f"- Chunk {item.chunk_id}"
            )


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = "What is Agentic AI?"

    verified_result = run_verified_agent(question)
    display_verified_result(verified_result)
