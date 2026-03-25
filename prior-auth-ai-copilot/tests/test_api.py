import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.api.main import app

client = TestClient(app)


def test_health_check():
    """Health endpoint returns 200 and correct fields."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app_name"] == "prior-auth-ai-copilot"
    assert "version" in data


def test_root_endpoint():
    """Root endpoint returns app info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "app" in data
    assert "docs" in data


def test_review_missing_fields():
    """Review endpoint rejects incomplete requests."""
    response = client.post("/review", json={
        "case_id": "TEST-001",
        "patient_age": 45,
    })
    assert response.status_code == 422


def test_review_invalid_age():
    """Review endpoint rejects invalid patient age."""
    response = client.post("/review", json={
        "case_id": "TEST-001",
        "patient_age": 999,
        "diagnosis": "Test diagnosis",
        "requested_service": "Test service",
        "clinical_note_text": "Test note text here",
    })
    assert response.status_code == 422


@patch("app.api.routes.auth_review.pa_review_graph")
def test_review_returns_valid_response(mock_graph):
    """Review endpoint returns correctly structured response."""
    mock_graph.invoke.return_value = {
        "case_id": "TEST-001",
        "decision": "APPROVE",
        "confidence": 0.95,
        "case_summary": "Test summary",
        "rationale": "All criteria met",
        "missing_information": [],
        "criteria_results": [
            {
                "criterion": "Duration of symptoms",
                "status": "MET",
                "evidence": "8 weeks documented",
                "chunk_id": "policy::chunk_0001",
            }
        ],
        "retrieved_chunks": [],
        "reviewer_note": "Test reviewer note",
        "citations": ["policy::chunk_0001"],
        "disclaimer": "AI-generated draft — human review required",
        "prompt_tokens_total": 500,
        "completion_tokens_total": 200,
    }

    response = client.post("/review", json={
        "case_id": "TEST-001",
        "patient_age": 45,
        "diagnosis": "Low back pain",
        "requested_service": "MRI Lumbar Spine",
        "clinical_note_text": "Patient has 8 weeks of low back pain.",
    })

    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "APPROVE"
    assert data["confidence"] == 0.95
    assert len(data["criteria_results"]) == 1
    assert data["criteria_results"][0]["status"] == "MET"


@patch("app.api.routes.auth_review.pa_review_graph")
def test_review_handles_workflow_error(mock_graph):
    """Review endpoint returns 500 when workflow raises exception."""
    mock_graph.invoke.side_effect = Exception("Workflow failed")

    response = client.post("/review", json={
        "case_id": "TEST-ERR-001",
        "patient_age": 45,
        "diagnosis": "Low back pain",
        "requested_service": "MRI Lumbar Spine",
        "clinical_note_text": "Patient has 8 weeks of low back pain.",
    })

    assert response.status_code == 500
