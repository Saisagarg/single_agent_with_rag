
import sqlite3
from pathlib import Path
from uuid import UUID

from app.retrieval.document_context import DEFAULT_DOCUMENT_ID


ROOT = Path(__file__).resolve().parents[2]
DATABASE = ROOT / "data" / "document_registry.db"


def connect():
    DATABASE.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE, timeout=30)
    connection.row_factory = sqlite3.Row

    connection.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    return connection


def register_document(document_id, file_name):
    document_id = str(UUID(str(document_id)))

    with connect() as connection:
        connection.execute(
            """
            INSERT INTO documents
                (document_id, file_name, status)
            VALUES (?, ?, ?)
            """,
            (document_id, file_name, "search_ready"),
        )


def get_document(document_id):
    document_id = str(UUID(str(document_id)))

    if document_id == DEFAULT_DOCUMENT_ID:
        return {
            "document_id": DEFAULT_DOCUMENT_ID,
            "file_name": "sample.pdf",
            "status": "search_ready",
        }

    with connect() as connection:
        row = connection.execute(
            """
            SELECT document_id, file_name, status
            FROM documents
            WHERE document_id = ?
            """,
            (document_id,),
        ).fetchone()

    return dict(row) if row else None


def list_documents():
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT document_id, file_name,
                   status, created_at
            FROM documents
            ORDER BY created_at DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]

def claim_graph_build(document_id):
    """Atomically claim a document for graph processing."""
    document_id = str(UUID(str(document_id)))

    with connect() as connection:
        cursor = connection.execute(
            """
            UPDATE documents
            SET status = 'graph_processing'
            WHERE document_id = ?
              AND status IN ('search_ready', 'graph_failed')
            """,
            (document_id,),
        )
        return cursor.rowcount == 1


def update_document_status(document_id, status):
    document_id = str(UUID(str(document_id)))

    allowed = {"graph_ready", "graph_failed"}

    if status not in allowed:
        raise ValueError("Invalid document status")

    with connect() as connection:
        cursor = connection.execute(
            """
            UPDATE documents
            SET status = ?
            WHERE document_id = ?
              AND status = 'graph_processing'
            """,
            (status, document_id),
        )

        if cursor.rowcount != 1:
            raise RuntimeError(
                "Document was not in graph_processing state"
            )
