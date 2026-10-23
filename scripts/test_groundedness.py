
from app.agents.agent_controller import run_agent
from app.agents.groundedness_evaluator import (
    evaluate_groundedness,
)

question = "What is Agentic AI?"

print("Running Agentic RAG...")
result = run_agent(question)

print("\nGENERATED ANSWER")
print(result.answer)

if not result.successful_tool:
    print("\nNo sufficient evidence. Skipping evaluation.")
else:
    print("\nEvaluating claim-level groundedness...")

    evaluation = evaluate_groundedness(
        question=result.question,
        answer=result.answer,
        evidence=result.evidence,
    )

    print("\nGROUNDEDNESS RESULTS")
    print("Grounded:", evaluation.grounded)
    print("Total claims:", len(evaluation.claims))
    print(
        "Unsupported claims:",
        len(evaluation.unsupported_claims),
    )
    print("Reason:", evaluation.reason)

    for index, claim in enumerate(evaluation.claims, 1):
        print(f"\nClaim {index}: {claim['claim']}")
        print("Supported:", claim["supported"])
        print("Citations:", claim["citations"])
        print("Reason:", claim.get("reason", ""))

    if not evaluation.grounded:
        print(
            "\nWARNING: Answer failed the groundedness "
            "evaluation and should not be treated as verified."
        )
