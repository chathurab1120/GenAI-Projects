import json
from app.core.constants import TOP_K_RESULTS
from app.core.logging_config import get_logger
from app.llm.llm_factory import LLMClient
from app.llm.output_parsers import (
    parse_llm_output,
    CaseSummaryOutput,
    EvidenceOutput,
    CriteriaComparisonOutput,
    RecommendationOutput,
    ReviewerNoteOutput,
)
from app.llm.prompts import (
    SYSTEM_SUMMARIZE_CASE,
    SYSTEM_EXTRACT_EVIDENCE,
    SYSTEM_COMPARE_CRITERIA,
    SYSTEM_RECOMMEND,
    SYSTEM_REVIEWER_NOTE,
    build_summarize_prompt,
    build_extract_evidence_prompt,
    build_compare_criteria_prompt,
    build_recommend_prompt,
    build_reviewer_note_prompt,
)
from app.retrieval.retriever import PolicyRetriever
from app.workflows.state import PAReviewState

logger = get_logger(__name__)

# Shared clients — initialised once, reused across nodes
_llm_client: LLMClient | None = None
_retriever: PolicyRetriever | None = None


def _get_llm() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def _get_retriever() -> PolicyRetriever:
    global _retriever
    if _retriever is None:
        _retriever = PolicyRetriever()
    return _retriever


def intake_node(state: PAReviewState) -> dict:
    """
    Node 1 — Intake.
    Validates required input fields are present.
    Initialises token counters and error list.
    """
    logger.info(f"[intake] Processing case: {state.get('case_id')}")
    errors = []

    required = ["case_id", "diagnosis", "requested_service", "clinical_note_text"]
    for field in required:
        if not state.get(field):
            errors.append(f"Missing required field: {field}")

    return {
        "errors": errors,
        "prompt_tokens_total": 0,
        "completion_tokens_total": 0,
    }


def retrieve_policy_node(state: PAReviewState) -> dict:
    """
    Node 2 — Retrieve policy.
    Searches the vector store for policy chunks relevant
    to the diagnosis and requested service.
    """
    logger.info(f"[retrieve_policy] case: {state.get('case_id')}")

    query = (
        f"{state.get('diagnosis', '')} "
        f"{state.get('requested_service', '')} "
        f"{state.get('clinical_note_text', '')[:200]}"
    )

    retriever = _get_retriever()
    chunks = retriever.retrieve(query=query, top_k=TOP_K_RESULTS)

    retrieved = [
        {
            "chunk_id": c.chunk_id,
            "content": c.content,
            "source_file": c.source_file,
            "similarity_score": c.similarity_score,
        }
        for c in chunks
    ]

    logger.info(
        f"[retrieve_policy] Retrieved {len(retrieved)} chunks"
    )
    return {"retrieved_chunks": retrieved}


def summarize_case_node(state: PAReviewState) -> dict:
    """
    Node 3 — Summarize case.
    Generates a concise clinical summary of the request.
    """
    logger.info(f"[summarize_case] case: {state.get('case_id')}")

    llm = _get_llm()
    user_prompt = build_summarize_prompt(
        diagnosis=state.get("diagnosis", ""),
        requested_service=state.get("requested_service", ""),
        patient_age=state.get("patient_age", 0),
        provider_specialty=state.get("provider_specialty", ""),
        clinical_note=state.get("clinical_note_text", ""),
    )

    response = llm.complete(
        system_prompt=SYSTEM_SUMMARIZE_CASE,
        user_prompt=user_prompt,
    )
    parsed: CaseSummaryOutput = parse_llm_output(
        response.content, CaseSummaryOutput
    )

    return {
        "case_summary": parsed.case_summary,
        "key_clinical_facts": parsed.key_clinical_facts,
        "urgency_indicators": parsed.urgency_indicators,
        "prompt_tokens_total": (
            state.get("prompt_tokens_total", 0) + response.prompt_tokens
        ),
        "completion_tokens_total": (
            state.get("completion_tokens_total", 0) + response.completion_tokens
        ),
    }


def extract_evidence_node(state: PAReviewState) -> dict:
    """
    Node 4 — Extract evidence.
    Pulls structured clinical facts from the note text.
    """
    logger.info(f"[extract_evidence] case: {state.get('case_id')}")

    llm = _get_llm()
    user_prompt = build_extract_evidence_prompt(
        clinical_note=state.get("clinical_note_text", ""),
        diagnosis=state.get("diagnosis", ""),
        requested_service=state.get("requested_service", ""),
    )

    response = llm.complete(
        system_prompt=SYSTEM_EXTRACT_EVIDENCE,
        user_prompt=user_prompt,
    )
    parsed: EvidenceOutput = parse_llm_output(
        response.content, EvidenceOutput
    )

    return {
        "extracted_evidence": parsed.model_dump(),
        "prompt_tokens_total": (
            state.get("prompt_tokens_total", 0) + response.prompt_tokens
        ),
        "completion_tokens_total": (
            state.get("completion_tokens_total", 0) + response.completion_tokens
        ),
    }


def compare_criteria_node(state: PAReviewState) -> dict:
    """
    Node 5 — Compare criteria.
    Matches extracted evidence against retrieved policy criteria.
    """
    logger.info(f"[compare_criteria] case: {state.get('case_id')}")

    llm = _get_llm()

    chunks = state.get("retrieved_chunks", [])
    policy_text = "\n\n".join(
        f"[{c['chunk_id']}]\n{c['content']}" for c in chunks
    )

    user_prompt = build_compare_criteria_prompt(
        extracted_evidence=json.dumps(
            state.get("extracted_evidence", {}), indent=2
        ),
        retrieved_policy_chunks=policy_text,
    )

    response = llm.complete(
        system_prompt=SYSTEM_COMPARE_CRITERIA,
        user_prompt=user_prompt,
    )
    parsed: CriteriaComparisonOutput = parse_llm_output(
        response.content, CriteriaComparisonOutput
    )

    return {
        "criteria_results": [r.model_dump() for r in parsed.criteria_results],
        "overall_assessment": parsed.overall_assessment,
        "prompt_tokens_total": (
            state.get("prompt_tokens_total", 0) + response.prompt_tokens
        ),
        "completion_tokens_total": (
            state.get("completion_tokens_total", 0) + response.completion_tokens
        ),
    }


def recommend_decision_node(state: PAReviewState) -> dict:
    """
    Node 6 — Recommend decision.
    Generates APPROVE / DENY / NEED_MORE_INFO with rationale.
    """
    logger.info(f"[recommend_decision] case: {state.get('case_id')}")

    llm = _get_llm()
    user_prompt = build_recommend_prompt(
        criteria_results=json.dumps(
            state.get("criteria_results", []), indent=2
        ),
        case_summary=state.get("case_summary", ""),
    )

    response = llm.complete(
        system_prompt=SYSTEM_RECOMMEND,
        user_prompt=user_prompt,
    )
    parsed: RecommendationOutput = parse_llm_output(
        response.content, RecommendationOutput
    )

    return {
        "decision": parsed.decision.value,
        "confidence": parsed.confidence,
        "rationale": parsed.rationale,
        "missing_information": parsed.missing_information,
        "citations": parsed.citations,
        "disclaimer": parsed.disclaimer,
        "prompt_tokens_total": (
            state.get("prompt_tokens_total", 0) + response.prompt_tokens
        ),
        "completion_tokens_total": (
            state.get("completion_tokens_total", 0) + response.completion_tokens
        ),
    }


def generate_reviewer_note_node(state: PAReviewState) -> dict:
    """
    Node 7 — Generate reviewer note.
    Writes a professional clinical reviewer note for the decision.
    """
    logger.info(
        f"[generate_reviewer_note] case: {state.get('case_id')}"
    )

    llm = _get_llm()
    user_prompt = build_reviewer_note_prompt(
        decision=state.get("decision", ""),
        rationale=state.get("rationale", ""),
        criteria_results=json.dumps(
            state.get("criteria_results", []), indent=2
        ),
        missing_information=state.get("missing_information", []),
    )

    response = llm.complete(
        system_prompt=SYSTEM_REVIEWER_NOTE,
        user_prompt=user_prompt,
    )
    parsed: ReviewerNoteOutput = parse_llm_output(
        response.content, ReviewerNoteOutput
    )

    return {
        "reviewer_note": parsed.reviewer_note,
        "prompt_tokens_total": (
            state.get("prompt_tokens_total", 0) + response.prompt_tokens
        ),
        "completion_tokens_total": (
            state.get("completion_tokens_total", 0) + response.completion_tokens
        ),
    }


def audit_log_node(state: PAReviewState) -> dict:
    """
    Node 8 — Audit log.
    Saves the completed review to the SQLite audit database.
    """
    import time
    from datetime import datetime, UTC
    from app.db.init_db import init_db
    from app.db.session import get_db_session
    from app.db.models import AuditLog

    logger.info(f"[audit_log] Saving case: {state.get('case_id')}")

    try:
        init_db()
        with get_db_session() as session:
            record = AuditLog(
                case_id=state.get("case_id", "UNKNOWN"),
                timestamp=datetime.now(UTC),
                member_id=state.get("member_id"),
                patient_age=state.get("patient_age"),
                diagnosis=state.get("diagnosis"),
                requested_service=state.get("requested_service"),
                provider_specialty=state.get("provider_specialty"),
                decision=state.get("decision", "UNKNOWN"),
                confidence=state.get("confidence"),
                rationale=state.get("rationale"),
                missing_information=str(
                    state.get("missing_information", [])
                ),
                llm_model="gpt-4o-mini",
                prompt_tokens=state.get("prompt_tokens_total"),
                completion_tokens=state.get("completion_tokens_total"),
            )
            session.add(record)
        logger.info(f"[audit_log] Saved case: {state.get('case_id')}")
    except Exception as e:
        logger.error(f"[audit_log] Failed to save audit record: {e}")

    return {}
