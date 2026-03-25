# scripts\verify_workflow.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging_config import setup_logging
from app.workflows.graph import pa_review_graph

setup_logging()

print("=== Testing full PA review workflow ===")
print("Running sample case PA-2024-001 through all 8 nodes...")
print()

initial_state = {
    "case_id": "PA-2024-001",
    "member_id": "SYNTHETIC-MBR-1001",
    "patient_age": 52,
    "diagnosis": "Low back pain with radiculopathy",
    "requested_service": "MRI Lumbar Spine without contrast",
    "provider_specialty": "Orthopedic Surgery",
    "clinical_note_text": (
        "Patient is a 52-year-old male presenting with 8 weeks of low back "
        "pain radiating down the left leg to the knee. Patient reports "
        "numbness in the left foot. Physical exam shows decreased sensation "
        "L4-L5 distribution and positive straight leg raise at 45 degrees "
        "on the left. Patient completed 6 weeks of physical therapy "
        "(8 sessions) and has been on naproxen 500mg BID for 8 weeks "
        "with minimal improvement. Requesting MRI to evaluate for disc "
        "herniation or nerve root compression."
    ),
    "policy_name": "mri_lumbar_spine_policy",
}

result = pa_review_graph.invoke(initial_state)

print("=== WORKFLOW RESULT ===")
print(f"Decision:     {result.get('decision')}")
print(f"Confidence:   {result.get('confidence')}")
print(f"Summary:      {result.get('case_summary', '')[:120]}...")
print(f"Rationale:    {result.get('rationale', '')[:120]}...")
print(f"Missing info: {result.get('missing_information')}")
print(f"Citations:    {result.get('citations')}")
print(f"Reviewer note:{result.get('reviewer_note', '')[:120]}...")
print(f"Total tokens: {result.get('prompt_tokens_total', 0) + result.get('completion_tokens_total', 0)}")
print()
print("Workflow OK")
