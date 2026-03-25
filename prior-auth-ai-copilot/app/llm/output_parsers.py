import json
from typing import Any
from pydantic import BaseModel, Field, field_validator
from app.core.constants import Decision
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class CaseSummaryOutput(BaseModel):
    """Parsed output from the case summarization step."""
    case_summary: str
    key_clinical_facts: list[str] = Field(default_factory=list)
    urgency_indicators: list[str] = Field(default_factory=list)


class EvidenceOutput(BaseModel):
    """Parsed output from the evidence extraction step."""
    symptom_duration: str | None = None
    conservative_treatments_tried: list[str] = Field(default_factory=list)
    clinical_findings: list[str] = Field(default_factory=list)
    lab_results: list[str] = Field(default_factory=list)
    specialist_involvement: str | None = None
    contraindications_noted: list[str] = Field(default_factory=list)
    additional_relevant_facts: list[str] = Field(default_factory=list)


class CriterionResult(BaseModel):
    """A single criterion comparison result."""
    criterion: str
    status: str
    evidence: str = ""
    chunk_id: str = ""

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"MET", "NOT_MET", "UNKNOWN"}
        if v.upper() not in allowed:
            return "UNKNOWN"
        return v.upper()


class CriteriaComparisonOutput(BaseModel):
    """Parsed output from the criteria comparison step."""
    criteria_results: list[CriterionResult] = Field(default_factory=list)
    overall_assessment: str = ""


class RecommendationOutput(BaseModel):
    """Parsed output from the recommendation step."""
    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    missing_information: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    disclaimer: str = "AI-generated draft — human review required"

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 2)


class ReviewerNoteOutput(BaseModel):
    """Parsed output from the reviewer note step."""
    reviewer_note: str
    disclaimer: str = "AI-generated draft — human review required"


def parse_llm_output(raw: str, model_class: type[BaseModel]) -> Any:
    """
    Parse raw LLM string output into a Pydantic model.
    Strips markdown code fences if present.
    Falls back to a safe default if parsing fails.

    Args:
        raw: Raw string content from the LLM.
        model_class: The Pydantic model class to parse into.

    Returns:
        An instance of model_class, or a safe fallback on failure.
    """
    cleaned = raw.strip()

    # Strip markdown code fences if present
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            line for line in lines
            if not line.startswith("```")
        ).strip()

    try:
        data = json.loads(cleaned)
        return model_class(**data)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            f"Failed to parse LLM output as {model_class.__name__}: {e}"
        )
        logger.debug(f"Raw output was: {raw[:200]}")
        return _get_fallback(model_class)


def _get_fallback(model_class: type[BaseModel]) -> Any:
    """Return a safe fallback instance when parsing fails."""
    if model_class is RecommendationOutput:
        return RecommendationOutput(
            decision=Decision.NEED_MORE_INFO,
            confidence=0.0,
            rationale="Unable to parse LLM output — manual review required",
            missing_information=["LLM output parsing failed"],
            disclaimer="AI-generated draft — human review required",
        )
    if model_class is CaseSummaryOutput:
        return CaseSummaryOutput(
            case_summary="Unable to parse case summary"
        )
    if model_class is EvidenceOutput:
        return EvidenceOutput()
    if model_class is CriteriaComparisonOutput:
        return CriteriaComparisonOutput()
    if model_class is ReviewerNoteOutput:
        return ReviewerNoteOutput(
            reviewer_note="Unable to parse reviewer note — manual review required"
        )
    raise ValueError(f"No fallback defined for {model_class.__name__}")
