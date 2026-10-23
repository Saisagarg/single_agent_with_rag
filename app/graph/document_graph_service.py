
from uuid import UUID

from app.graph import build_graph_schema as builder
from app.graph import extract_knowledge as knowledge


def process_document_graph(document_id):
    """Build a graph and extract source-backed facts."""

    document_id = str(UUID(str(document_id)))

    # Load exactly the selected document's chunks.
    chunks = builder.load_chunks(document_id)

    builder.create_schema()
    builder.build_graph(chunks, document_id)
    builder.verify_graph(document_id)

    # Extract facts from Neo4j chunks.
    graph_chunks = knowledge.load_chunks(document_id)

    if len(graph_chunks) != len(chunks):
        raise RuntimeError(
            "Graph chunk count does not match Search index."
        )

    total_facts = 0
    rejected_facts = 0

    for index, chunk in enumerate(graph_chunks, 1):
        print(
            f"Extracting facts from chunk "
            f"{index}/{len(graph_chunks)}"
        )

        # If extraction fails, propagate the error.
        # The document must not be marked graph_ready.
        extracted = knowledge.extract_facts(chunk)

        valid = []

        for fact in extracted:
            checked = knowledge.validate_fact(
                fact,
                chunk,
                document_id=document_id,
            )

            if checked:
                valid.append(checked)
            else:
                rejected_facts += 1

        knowledge.save_facts(valid)
        total_facts += len(valid)

    knowledge.verify_graph(document_id)

    return {
        "document_id": document_id,
        "chunk_count": len(graph_chunks),
        "accepted_facts": total_facts,
        "rejected_facts": rejected_facts,
    }
