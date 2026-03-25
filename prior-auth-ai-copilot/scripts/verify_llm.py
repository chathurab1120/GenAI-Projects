# scripts\verify_llm.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging_config import setup_logging
from app.llm.llm_factory import LLMClient
from app.llm.prompts import build_summarize_prompt
from app.llm.output_parsers import parse_llm_output, CaseSummaryOutput

setup_logging()

print("=== Testing LLM layer ===")

client = LLMClient()

system_prompt = "You are a clinical reviewer. Output valid JSON only."
user_prompt = build_summarize_prompt(
    diagnosis="Low back pain with radiculopathy",
    requested_service="MRI Lumbar Spine",
    patient_age=52,
    provider_specialty="Orthopedics",
    clinical_note=(
        "Patient has 8 weeks of low back pain radiating to left leg. "
        "Completed 8 PT sessions and naproxen 500mg BID. "
        "Positive straight leg raise at 45 degrees."
    ),
)

print("Sending test summarization request to OpenAI...")
response = client.complete(
    system_prompt=system_prompt,
    user_prompt=user_prompt,
)

print(f"Tokens used: {response.total_tokens}")
print(f"Raw response preview: {response.content[:120]}...")

parsed = parse_llm_output(response.content, CaseSummaryOutput)
print(f"Parsed case_summary: {parsed.case_summary}")
print(f"Key facts count: {len(parsed.key_clinical_facts)}")
print()
print("LLM layer OK")
