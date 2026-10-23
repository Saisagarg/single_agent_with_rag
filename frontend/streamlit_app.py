
import os

import requests
import streamlit as st


API_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

st.set_page_config(
    page_title="Modular-RAG AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def api_request(method, path, timeout=30, **kwargs):
    """Call the FastAPI backend with readable errors."""
    try:
        response = requests.request(
            method,
            f"{API_URL}{path}",
            timeout=timeout,
            **kwargs,
        )

        response.raise_for_status()
        return response.json()

    except requests.HTTPError as exc:
        detail = response.text[:500]
        try:
            detail = response.json().get("detail", detail)
        except ValueError:
            pass

        raise RuntimeError(
            f"Backend returned HTTP {response.status_code}: "
            f"{detail}"
        ) from exc

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach the FastAPI backend: {exc}"
        ) from exc


def get_documents():
    return api_request(
        "GET",
        "/api/documents",
    ).get("documents", [])


def upload_pdf(uploaded_file):
    return api_request(
        "POST",
        "/api/documents/upload",
        timeout=300,
        files={
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf",
            )
        },
    )


def build_graph(document_id):
    return api_request(
        "POST",
        f"/api/documents/{document_id}/build-graph",
        timeout=600,
    )


def ask_question(question, document_id):
    return api_request(
        "POST",
        "/api/ask",
        timeout=300,
        json={
            "question": question,
            "document_id": document_id,
        },
    )


def show_answer(response):
    st.markdown(response.get("answer", "No answer returned."))

    grounded = response.get("grounded", False)
    tool = response.get("successful_tool") or "None"

    if grounded:
        st.success("Groundedness evaluation passed")
    else:
        st.warning(
            "The response was not verified as grounded."
        )

    st.caption(f"Retrieval tool: {tool}")

    evidence = response.get("evidence") or []

    with st.expander(
        f"Source evidence ({len(evidence)})"
    ):
        if not evidence:
            st.info("No verified source evidence returned.")

        for index, item in enumerate(evidence, start=1):
            st.markdown(
                f"**[{index}] {item.get('file_name', 'Unknown')}**"
            )

            st.caption(
                f"Page {item.get('page_number', '?')} "
                f"| Chunk {item.get('chunk_id', '?')}"
            )

            st.write(item.get("content", ""))
            st.divider()

    with st.expander("Agent execution trace"):
        st.subheader("Retrieval")
        st.json(response.get("retrieval_trace") or [])

        st.subheader("Verification")
        st.json(response.get("verification_trace") or [])


# -----------------------------
# Session state
# -----------------------------

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "selected_document" not in st.session_state:
    st.session_state.selected_document = None


# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.title("🧠 Modular-RAG AI")
    st.caption("Microsoft Azure + Neo4j")

    st.divider()
    st.subheader("Upload document")

    uploaded = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        max_upload_size=10,
    )

    if st.button(
        "Upload and index",
        use_container_width=True,
        disabled=uploaded is None,
    ):
        try:
            with st.spinner(
                "Uploading PDF and generating embeddings..."
            ):
                result = upload_pdf(uploaded)

            st.session_state.selected_document = (
                result["document_id"]
            )

            st.success("PDF indexed successfully!")
            st.rerun()

        except RuntimeError as exc:
            st.error(str(exc))

    st.divider()
    st.subheader("Document library")

    try:
        documents = get_documents()
    except RuntimeError as exc:
        st.error(str(exc))
        documents = []

    # Distinguish documents that share the same filename.
    document_map = {
        (
            f"{item['file_name']} "
            f"({item['document_id'][:8]}) "
            f"— {item['status']}"
        ): item
        for item in documents
    }

    if document_map:
        labels = list(document_map.keys())

        previous = st.session_state.selected_document

        selected_index = next(
            (
                index
                for index, label in enumerate(labels)
                if document_map[label]["document_id"]
                == previous
            ),
            0,
        )

        selected_label = st.selectbox(
            "Select document",
            options=labels,
            index=selected_index,
        )

        selected = document_map[selected_label]

        document_id = selected["document_id"]
        st.session_state.selected_document = document_id

        st.caption(
            f"Document ID: {document_id}"
        )

        status = selected["status"]
        st.write(f"**Status:** {status}")

        if status in {"search_ready", "graph_failed"}:
            if st.button(
                "Build knowledge graph",
                use_container_width=True,
            ):
                try:
                    with st.spinner(
                        "Building Neo4j graph and extracting facts..."
                    ):
                        result = build_graph(document_id)

                    st.success(
                        "Knowledge graph processing completed."
                    )
                    st.json(result)
                    st.rerun()

                except RuntimeError as exc:
                    st.error(str(exc))

        if status == "graph_processing":
            st.info(
                "Graph processing is in progress. "
                "Refresh to check its status."
            )

        if st.button(
            "Refresh documents",
            use_container_width=True,
        ):
            st.rerun()

    else:
        document_id = None
        st.info("Upload a PDF to get started.")

    st.divider()
    st.caption("Local development interface")


# -----------------------------
# Main chat area
# -----------------------------

st.title("Agentic Retrieval-Augmented Generation")

st.write(
    "Ask questions about your selected PDF. "
    "The system retrieves evidence, selects retrieval "
    "tools, validates citations, and evaluates groundedness."
)

if not document_id:
    st.info(
        "Select or upload a document in the sidebar."
    )
    st.stop()

history = st.session_state.chats.setdefault(
    document_id, []
)

col1, col2 = st.columns([4, 1])

with col1:
    st.caption(
        f"Active document: {selected['file_name']}"
    )

with col2:
    if st.button(
        "Clear chat",
        use_container_width=True,
    ):
        history.clear()
        st.rerun()


for message in history:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            show_answer(message["response"])
        else:
            st.markdown(message["content"])


prompt = st.chat_input(
    "Ask a question about this document..."
)

if prompt:
    history.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner(
                "Searching documents and verifying the answer..."
            ):
                response = ask_question(
                    prompt,
                    document_id,
                )

            show_answer(response)

            history.append({
                "role": "assistant",
                "response": response,
            })

        except RuntimeError as exc:
            st.error(str(exc))
