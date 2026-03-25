import time
from fastapi import APIRouter, HTTPException
from app.api.schemas.request_models import PAReviewRequest
from app.api.schemas.response_models import (
    PAReviewResponse,
    CriterionResultResponse,
    RetrievedChunkResponse,
)
from app.core.logging_config import get_logger
from app.workflows.graph import pa_review_graph

router = APIRouter()
logger = get_logger(__name__)


@router.post("/review", response_model=PAReviewResponse)
def run_pa_review(request: PAReviewRequest) -> PAReviewResponse:
    """
    Run a full prior authorization review through the LangGraph workflow.

    Accepts a prior auth request with clinical note, runs it through
    all 8 workflow nodes, and returns a structured decision with
    citations, rationale, and reviewer note.

    This output is decision support only — not a final coverage
    determination. Human review is required.
    """
    logger.info(f"Received PA review request: {request.case_id}")
    start_time = time.time()

    try:
        initial_state = {
            "case_id": request.case_id,
            "member_id": request.member_id,
            "patient_age": request.patient_age,
            "diagnosis": request.diagnosis,
            "requested_service": request.requested_service,
            "provider_specialty": request.provider_specialty,
            "clinical_note_text": request.clinical_note_text,
            "policy_name": request.policy_name,
        }

        result = pa_review_graph.invoke(initial_state)

        elapsed = round(time.time() - start_time, 2)
        logger.info(
            f"Review complete: {request.case_id} | "
            f"decision={result.get('decision')} | "
            f"elapsed={elapsed}s"
        )

        criteria = [
            CriterionResultResponse(**c)
            for c in result.get("criteria_results", [])
        ]

        chunks = [
            RetrievedChunkResponse(**c)
            for c in result.get("retrieved_chunks", [])
        ]

        return PAReviewResponse(
            case_id=request.case_id,
            decision=result.get("decision", "NEED_MORE_INFO"),
            confidence=result.get("confidence", 0.0),
            case_summary=result.get("case_summary", ""),
            rationale=result.get("rationale", ""),
            missing_information=result.get("missing_information", []),
            criteria_results=criteria,
            retrieved_chunks=chunks,
            reviewer_note=result.get("reviewer_note", ""),
            citations=result.get("citations", []),
            disclaimer=result.get(
                "disclaimer",
                "AI-generated draft — human review required"
            ),
            prompt_tokens_total=result.get("prompt_tokens_total", 0),
            completion_tokens_total=result.get(
                "completion_tokens_total", 0
            ),
        )

    except Exception as e:
        logger.error(f"Review failed for {request.case_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Review processing failed: {str(e)}"
        )
