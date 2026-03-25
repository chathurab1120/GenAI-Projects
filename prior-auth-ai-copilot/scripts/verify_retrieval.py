# scripts\verify_retrieval.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging_config import setup_logging
from app.retrieval.retriever import PolicyRetriever

setup_logging()

print("=== Testing retrieval layer ===")
print("Initialising PolicyRetriever...")
retriever = PolicyRetriever()

print()
print("--- Query 1: Lumbar spine MRI ---")
results = retriever.retrieve(
    query="low back pain radiculopathy MRI lumbar spine medical necessity",
    top_k=3,
)
for r in results:
    print(f"  [{r.similarity_score:.3f}] {r.source_file} | {r.content[:80]}...")

print()
print("--- Query 2: Sleep study ---")
results = retriever.retrieve(
    query="sleep apnea polysomnography daytime sleepiness snoring",
    top_k=3,
)
for r in results:
    print(f"  [{r.similarity_score:.3f}] {r.source_file} | {r.content[:80]}...")

print()
print("--- Query 3: Biologic therapy ---")
results = retriever.retrieve(
    query="rheumatoid arthritis biologic TNF inhibitor DMARD failure",
    top_k=3,
)
for r in results:
    print(f"  [{r.similarity_score:.3f}] {r.source_file} | {r.content[:80]}...")

print()
print("Retrieval layer OK")
