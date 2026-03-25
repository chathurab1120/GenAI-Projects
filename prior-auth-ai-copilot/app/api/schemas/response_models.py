from pydantic import BaseModel, Field
from app.core.constants import Decision


class CriterionResultResponse(BaseModel):
    criterion: str
    status: str
    evidence: str
    chunk_id: str


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    content: str
    source_file: str
    similarity_score: float


class PAReviewResponse(BaseModel):
    """
    Full structured response for a prior authorization review.
    """
    case_id: str
    decision: str
    confidence: float
    case_summary: str
    rationale: str
    missing_information: list[str] = Field(default_factory=list)
    criteria_results: list[CriterionResultResponse] = Field(
        default_factory=list
    )
    retrieved_chunks: list[RetrievedChunkResponse] = Field(
        default_factory=list
    )
    reviewer_note: str
    citations: list[str] = Field(default_factory=list)
    disclaimer: str
    prompt_tokens_total: int
    completion_tokens_total: int


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
