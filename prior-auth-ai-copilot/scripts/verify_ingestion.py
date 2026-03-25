# scripts\verify_ingestion.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging_config import setup_logging
from app.ingestion.policy_ingestor import (
    ingest_policy_documents,
    ingest_clinical_note,
)

setup_logging()

print("=== Testing policy document ingestion ===")
chunks = ingest_policy_documents()
print(f"Total policy chunks: {len(chunks)}")
print(f"First chunk ID: {chunks[0].chunk_id}")
print(f"First chunk preview: {chunks[0].content[:80]}...")

print()
print("=== Testing clinical note ingestion ===")
sample_note = (
    "Patient is a 52-year-old male with 8 weeks of low back pain "
    "radiating to the left leg. Completed physical therapy 8 sessions "
    "and naproxen 500mg BID. Positive straight leg raise at 45 degrees."
)
note_chunks = ingest_clinical_note(sample_note, "TEST-001")
print(f"Clinical note chunks: {len(note_chunks)}")
print(f"Note chunk preview: {note_chunks[0].content[:80]}...")

print()
print("Ingestion layer OK")
