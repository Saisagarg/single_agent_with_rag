
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import UUID


DEFAULT_DOCUMENT_ID = (
    "a853569c-176a-4190-b92d-4d943d943bbd"
)

_active_document_id = ContextVar(
    "active_document_id",
    default=None,
)


def get_document_id():
    """Get the document selected for this request."""
    selected = _active_document_id.get()

    return selected or DEFAULT_DOCUMENT_ID


@contextmanager
def use_document(document_id):
    """Temporarily select a document for retrieval."""

    if document_id is None:
        selected = DEFAULT_DOCUMENT_ID
    else:
        # Normalize and validate the UUID.
        selected = str(UUID(str(document_id)))

    token = _active_document_id.set(selected)

    try:
        yield
    finally:
        _active_document_id.reset(token)
