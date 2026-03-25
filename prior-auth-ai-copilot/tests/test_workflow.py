import pytest
from unittest.mock import patch, MagicMock
from app.core.constants import Decision


@patch("app.workflows.nodes._get_retriever")
@patch("app.workflows.nodes._get_llm")
def test_workflow_approve_case(mock_get_llm, mock_get_retriever):
    """Full workflow returns APPROVE for a well-documented case."""
    from app.llm.llm_factory import LLMResponse
    from app.retrieval.vectorstore import RetrievedChunk
    from app.workflows.graph import build_pa_review_graph

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        RetrievedChunk(
            chunk_id="mri_policy::chunk_0001",
            content="Duration 6 weeks, conservative therapy required",
            source_file="mri_lumbar_spine_policy.md",
            similarity_score=0.85,
            chunk_index=1,
        )
    ]
    mock_get_retriever.return_value = mock_retriever

    approve_response = LLMResponse(
        content='{"decision":"APPROVE","confidence":0.95,'
                '"rationale":"All criteria met","missing_information":[],'
                '"citations":["mri_policy::chunk_0001"],'
                '"disclaimer":"AI-generated draft — human review required"}',
        prompt_tokens=300,
        completion_tokens=100,
        total_tokens=400,
        model="gpt-4o-mini",
    )
    summary_response = LLMResponse(
        content='{"case_summary":"Test summary",'
                '"key_clinical_facts":["8 weeks pain"],'
                '"urgency_indicators":[]}',
        prompt_tokens=200,
        completion_tokens=80,
        total_tokens=280,
        model="gpt-4o-mini",
    )
    evidence_response = LLMResponse(
        content='{"symptom_duration":"8 weeks",'
                '"conservative_treatments_tried":["PT","NSAIDs"],'
                '"clinical_findings":["positive SLR"],'
                '"lab_results":[],"specialist_involvement":null,'
                '"contraindications_noted":[],'
                '"additional_relevant_facts":[]}',
        prompt_tokens=250,
        completion_tokens=90,
        total_tokens=340,
        model="gpt-4o-mini",
    )
    criteria_response = LLMResponse(
        content='{"criteria_results":[{"criterion":"Duration",'
                '"status":"MET","evidence":"8 weeks",'
                '"chunk_id":"mri_policy::chunk_0001"}],'
                '"overall_assessment":"Criteria met"}',
        prompt_tokens=400,
        completion_tokens=120,
        total_tokens=520,
        model="gpt-4o-mini",
    )
    note_response = LLMResponse(
        content='{"reviewer_note":"Approved based on criteria",'
                '"disclaimer":"AI-generated draft — human review required"}',
        prompt_tokens=200,
        completion_tokens=60,
        total_tokens=260,
        model="gpt-4o-mini",
    )

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = [
        summary_response,
        evidence_response,
        criteria_response,
        approve_response,
        note_response,
    ]
    mock_get_llm.return_value = mock_llm

    graph = build_pa_review_graph()
    result = graph.invoke({
        "case_id": "TEST-APPROVE-001",
        "member_id": "SYNTHETIC-001",
        "patient_age": 52,
        "diagnosis": "Low back pain with radiculopathy",
        "requested_service": "MRI Lumbar Spine",
        "provider_specialty": "Orthopedics",
        "clinical_note_text": (
            "Patient has 8 weeks of low back pain. "
            "Completed PT and NSAIDs. Positive SLR."
        ),
    })

    assert result["decision"] == "APPROVE"
    assert result["confidence"] == 0.95
    assert result["case_summary"] == "Test summary"
    assert len(result["criteria_results"]) == 1


def test_intake_node_missing_fields():
    """Intake node flags missing required fields in errors."""
    from app.workflows.nodes import intake_node
    state = {
        "case_id": "",
        "diagnosis": "",
        "requested_service": "",
        "clinical_note_text": "",
    }
    result = intake_node(state)
    assert len(result["errors"]) > 0


def test_intake_node_valid_input():
    """Intake node passes with all required fields present."""
    from app.workflows.nodes import intake_node
    state = {
        "case_id": "TEST-001",
        "diagnosis": "Low back pain",
        "requested_service": "MRI Lumbar Spine",
        "clinical_note_text": "Patient note here",
    }
    result = intake_node(state)
    assert result["errors"] == []
    assert result["prompt_tokens_total"] == 0
    assert result["completion_tokens_total"] == 0
