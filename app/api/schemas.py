
from typing import Any

from pydantic import BaseModel, Field



from typing import Optional
from uuid import UUID

class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=2000
    )
    document_id: Optional[UUID] = None

class EvidenceResponse(BaseModel):
    content: str
    chunk_id: str
    file_name: str
    page_number: int
    tool: str


class QuestionResponse(BaseModel):
    question: str
    answer: str
    grounded: bool
    successful_tool: str
    evidence: list[EvidenceResponse]
    retrieval_trace: list[dict[str, Any]]
    verification_trace: list[dict[str, Any]]
