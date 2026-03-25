# scripts\verify_api.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging_config import setup_logging
setup_logging()

from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

print("=== Testing FastAPI backend ===")
print()

print("--- Test 1: Health check ---")
response = client.get("/health")
print(f"Status: {response.status_code}")
print(f"Body:   {response.json()}")

print()
print("--- Test 2: Root endpoint ---")
response = client.get("/")
print(f"Status: {response.status_code}")
print(f"Body:   {response.json()}")

print()
print("--- Test 3: Full PA review via API ---")
payload = {
    "case_id": "PA-API-TEST-001",
    "member_id": "SYNTHETIC-MBR-9001",
    "patient_age": 52,
    "diagnosis": "Low back pain with radiculopathy",
    "requested_service": "MRI Lumbar Spine without contrast",
    "provider_specialty": "Orthopedic Surgery",
    "clinical_note_text": (
        "Patient is a 52-year-old male with 8 weeks of low back pain "
        "radiating to the left leg. Completed 8 PT sessions and naproxen "
        "500mg BID. Positive straight leg raise at 45 degrees. "
        "Decreased sensation L4-L5 distribution."
    ),
    "policy_name": "mri_lumbar_spine_policy"
}

response = client.post("/review", json=payload)
print(f"Status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    print(f"Decision:    {data['decision']}")
    print(f"Confidence:  {data['confidence']}")
    print(f"Summary:     {data['case_summary'][:100]}...")
    print(f"Criteria:    {len(data['criteria_results'])} items")
    print(f"Citations:   {data['citations']}")
    print(f"Tokens:      {data['prompt_tokens_total'] + data['completion_tokens_total']}")
else:
    print(f"ERROR: {response.text}")

print()
print("API layer OK")
