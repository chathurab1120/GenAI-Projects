import pytest
from app.llm.output_parsers import (
    parse_llm_output,
    RecommendationOutput,
    CaseSummaryOutput,
    EvidenceOutput,
    CriteriaComparisonOutput,
    ReviewerNoteOutput,
)
from app.core.constants import Decision


def test_parse_valid_recommendation():
    """Valid recommendation JSON parses correctly."""
    raw = """
    {
        "decision": "APPROVE",
        "confidence": 0.92,
        "rationale": "All criteria met",
        "missing_information": [],
        "citations": ["policy::chunk_0001"],
        "disclaimer": "AI-generated draft — human review required"
    }
    """
    result = parse_llm_output(raw, RecommendationOutput)
    assert result.decision == Decision.APPROVE
    assert result.confidence == 0.92
    assert result.citations == ["policy::chunk_0001"]


def test_parse_deny_decision():
    """DENY decision parses correctly."""
    raw = """
    {
        "decision": "DENY",
        "confidence": 0.85,
        "rationale": "Criteria not met",
        "missing_information": ["Conservative therapy not documented"],
        "citations": [],
        "disclaimer": "AI-generated draft — human review required"
    }
    """
    result = parse_llm_output(raw, RecommendationOutput)
    assert result.decision == Decision.DENY
    assert len(result.missing_information) == 1


def test_parse_need_more_info_decision():
    """NEED_MORE_INFO decision parses correctly."""
    raw = """
    {
        "decision": "NEED_MORE_INFO",
        "confidence": 0.60,
        "rationale": "Incomplete documentation",
        "missing_information": ["Duration not documented", "No lab results"],
        "citations": [],
        "disclaimer": "AI-generated draft — human review required"
    }
    """
    result = parse_llm_output(raw, RecommendationOutput)
    assert result.decision == Decision.NEED_MORE_INFO
    assert len(result.missing_information) == 2


def test_parse_recommendation_with_markdown_fences():
    """Parser strips markdown code fences from LLM output."""
    raw = """```json
    {
        "decision": "APPROVE",
        "confidence": 0.88,
        "rationale": "Criteria met",
        "missing_information": [],
        "citations": [],
        "disclaimer": "AI-generated draft — human review required"
    }
```"""
    result = parse_llm_output(raw, RecommendationOutput)
    assert result.decision == Decision.APPROVE


def test_parse_invalid_json_returns_fallback():
    """Malformed JSON returns safe NEED_MORE_INFO fallback."""
    raw = "This is not valid JSON at all {{ broken"
    result = parse_llm_output(raw, RecommendationOutput)
    assert result.decision == Decision.NEED_MORE_INFO
    assert result.confidence == 0.0
    assert "parsing failed" in result.missing_information[0].lower()


def test_parse_case_summary():
    """Case summary parses correctly."""
    raw = """
    {
        "case_summary": "Patient needs MRI",
        "key_clinical_facts": ["8 weeks pain", "PT completed"],
        "urgency_indicators": []
    }
    """
    result = parse_llm_output(raw, CaseSummaryOutput)
    assert result.case_summary == "Patient needs MRI"
    assert len(result.key_clinical_facts) == 2


def test_parse_case_summary_fallback():
    """Invalid case summary JSON returns fallback."""
    raw = "not json"
    result = parse_llm_output(raw, CaseSummaryOutput)
    assert "unable" in result.case_summary.lower()


def test_confidence_clamped():
    """Confidence value above 1.0 raises validation error."""
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        RecommendationOutput(
            decision=Decision.APPROVE,
            confidence=1.5,
            rationale="Test",
        )
