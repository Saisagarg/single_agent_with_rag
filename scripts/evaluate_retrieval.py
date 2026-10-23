
import csv
import json
import time
from pathlib import Path
from statistics import mean

from app.retrieval.standard_rag import retrieve
from app.retrieval.hybrid_rag import hybrid_retrieve
from app.retrieval.rag_fusion import (
    generate_queries,
    retrieve_for_query,
    reciprocal_rank_fusion,
)
from app.graph.graph_rag import (
    retrieve_graph_facts,
    driver as graph_driver,
)

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "evaluation" / "questions.json"
OUTPUT = ROOT / "evaluation" / "retrieval_results.csv"

TOP_K = 5


def run_vector(question):
    return retrieve(question, top_k=TOP_K)


def run_hybrid(question):
    return hybrid_retrieve(question, top_k=TOP_K)


def run_fusion(question):
    queries = generate_queries(question)

    results = [
        retrieve_for_query(query, top_k=TOP_K)
        for query in queries
    ]

    return reciprocal_rank_fusion(results)[:TOP_K]


def run_graph(question):
    facts = retrieve_graph_facts(
        question,
        top_k=TOP_K,
    )

    # Graph results are facts rather than chunks.
    # Map them back to distinct supporting chunks.
    ids = []
    for fact in facts:
        if fact["chunk_id"] not in ids:
            ids.append(fact["chunk_id"])

    return [{"chunk_id": chunk_id} for chunk_id in ids]


TOOLS = {
    "vector": run_vector,
    "hybrid": run_hybrid,
    "fusion": run_fusion,
    "graph": run_graph,
}


def calculate_metrics(retrieved, relevant):
    """
    Compute retrieval metrics only when manually
    reviewed relevance labels are available.
    """

    if not relevant:
        return {
            "precision_at_k": None,
            "recall_at_k": None,
            "mrr": None,
        }

    relevant = set(relevant)
    hits = [
        chunk_id for chunk_id in retrieved
        if chunk_id in relevant
    ]

    first_rank = next(
        (
            rank
            for rank, chunk_id in enumerate(
                retrieved, start=1
            )
            if chunk_id in relevant
        ),
        None,
    )

    return {
        "precision_at_k": len(hits) / TOP_K,
        "recall_at_k": len(hits) / len(relevant),
        "mrr": (
            1 / first_rank if first_rank else 0.0
        ),
    }


def main():
    dataset = json.loads(
        DATASET.read_text(encoding="utf-8")
    )

    rows = []

    for item in dataset:
        question = item["question"]
        relevant = item["relevant_chunk_ids"]

        print(f"\nQuestion: {question}")

        for tool_name, tool_fn in TOOLS.items():
            print(f"Testing {tool_name}...")

            start = time.perf_counter()

            try:
                results = tool_fn(question)
                latency = time.perf_counter() - start

                ids = list(dict.fromkeys(
                    doc["chunk_id"]
                    for doc in results
                ))[:TOP_K]

                metrics = calculate_metrics(
                    ids,
                    relevant,
                )

                row = {
                    "question_id": item["id"],
                    "question": question,
                    "tool": tool_name,
                    "status": "success",
                    "latency_seconds": round(latency, 3),
                    "retrieved_count": len(ids),
                    "retrieved_chunk_ids": json.dumps(ids),
                    **metrics,
                    "error": "",
                }

            except Exception as exc:
                row = {
                    "question_id": item["id"],
                    "question": question,
                    "tool": tool_name,
                    "status": "failed",
                    "latency_seconds": None,
                    "retrieved_count": 0,
                    "retrieved_chunk_ids": "[]",
                    "precision_at_k": None,
                    "recall_at_k": None,
                    "mrr": None,
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    )[:300],
                }

            rows.append(row)

            print(
                "Status:", row["status"],
                "| Latency:", row["latency_seconds"],
                "| Results:", row["retrieved_count"],
            )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nEVALUATION SUMMARY")
    print("-" * 60)

    for tool_name in TOOLS:
        successful = [
            row for row in rows
            if row["tool"] == tool_name
            and row["status"] == "success"
        ]

        print(f"\nTool: {tool_name}")
        print("Successful queries:", len(successful))

        if successful:
            print(
                "Average latency (seconds):",
                round(mean(
                    row["latency_seconds"]
                    for row in successful
                ), 3),
            )

            for key in [
                "precision_at_k",
                "recall_at_k",
                "mrr",
            ]:
                measured = [
                    row[key] for row in successful
                    if row[key] is not None
                ]

                print(
                    key,
                    round(mean(measured), 4)
                    if measured else "Not labeled",
                )

    print("\nResults saved to:", OUTPUT)


if __name__ == "__main__":
    try:
        main()
    finally:
        graph_driver.close()
