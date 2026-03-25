from typing import TypedDict, Any
from app.core.constants import Decision


class PAReviewState(TypedDict, total=False):
    """
    Shared state object passed between all LangGraph workflow nodes.
    Each node reads from and writes to this state.
    total=False means all fields are optional so nodes can
    add fields incrementally as the workflow progresses.
    """

    # ── Input fields (set by intake node) ──────────────────
    case_id: str
    member_id: str
    patient_age: int
    diagnosis: str
    requested_service: str
    provider_specialty: str
    clinical_note_text: str
    policy_name: str | None

    # ── Retrieval output (set by retrieve_policy node) ──────
    retrieved_chunks: list[dict[str, Any]]

    # ── Summarization output (set by summarize node) ────────
    case_summary: str
    key_clinical_facts: list[str]
    urgency_indicators: list[str]

    # ── Evidence output (set by extract_evidence node) ──────
    extracted_evidence: dict[str, Any]

    # ── Criteria output (set by compare_criteria node) ──────
    criteria_results: list[dict[str, Any]]
    overall_assessment: str

    # ── Recommendation (set by recommend node) ──────────────
    decision: str
    confidence: float
    rationale: str
    missing_information: list[str]
    citations: list[str]
    disclaimer: str

    # ── Reviewer note (set by reviewer_note node) ───────────
    reviewer_note: str

    # ── Metadata (set across nodes) ─────────────────────────
    prompt_tokens_total: int
    completion_tokens_total: int
    errors: list[str]
